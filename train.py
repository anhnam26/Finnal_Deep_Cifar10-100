"""One Python entry point; uses the current interpreter, never a new venv."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import signal
import site
import sys

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / '.deps'))
sys.path.insert(0, str(ROOT / '.deps'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))


def parser():
    p = argparse.ArgumentParser(description='CIFAR exam: train, auto-resume, search, frozen final evaluation')
    p.add_argument('--dataset', choices=['both', 'cifar10', 'cifar100'], default='both')
    p.add_argument('--model', choices=['compactmix', 'wrn', 'preactresnet', 'convnext'])
    p.add_argument('--config', type=Path)
    for flag, kind in [('epochs', int), ('batch-size', int), ('workers', int), ('seed', int),
                       ('lr', float), ('weight-decay', float), ('warmup', int), ('log-every', int),
                       ('limit-train', int), ('limit-val', int)]:
        p.add_argument('--' + flag, type=kind)
    p.add_argument('--optimizer', choices=['sgd', 'adam', 'adamw'])
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--run-name', default='default')
    p.add_argument('--output-root', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--smoke', action='store_true', help='Synthetic 48-image checks; no benchmark claims')
    p.add_argument('--matrix', action='store_true', help='SGD/Adam with three LRs each, per dataset')
    p.add_argument('--profile', action='store_true', help='Print budgets, then exit without training')
    p.add_argument('--fast', action='store_true', help='Allow nondeterministic CUDA algorithms')
    p.add_argument('--no-tensorboard', action='store_true')
    p.add_argument('--stop-after-epochs', type=int, help='Stop after N completed epochs this invocation')
    p.add_argument('--search', action='store_true')
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--study-name')
    p.add_argument('--multi-objective', action='store_true')
    p.add_argument('--finalize', type=Path, help='Completed selection run: freeze recipe, retrain on 50k, test once')
    p.add_argument('--evaluate', type=Path, help='Evaluate a completed frozen FINAL run, once')
    p.add_argument('--report', action='store_true', help='Rebuild CSV, leaderboard and final-report PDF')
    p.add_argument('--selection-only', action='store_true', help='Train/validate only; defer frozen retraining and test')
    return p


def configs_from_args(args):
    from src.config import default_config, merge_config
    import yaml
    datasets = ['cifar10', 'cifar100'] if args.dataset == 'both' else [args.dataset]
    supplied = yaml.safe_load(args.config.read_text(encoding='utf-8')) if args.config else {}
    if not isinstance(supplied, dict):
        raise ValueError('YAML configuration must be a mapping')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.run_name):
        raise ValueError('run-name must contain only letters, digits, dot, dash or underscore')
    result = []
    for dataset in datasets:
        cfg = merge_config(default_config(dataset), supplied)
        cfg['dataset'] = dataset
        for field in ('epochs', 'batch_size', 'workers', 'seed', 'lr', 'weight_decay', 'warmup',
                      'log_every', 'limit_train', 'limit_val', 'optimizer', 'device'):
            value = getattr(args, field)
            if value is not None:
                cfg[field] = value
        if args.optimizer in ('adam', 'adamw'):
            if args.lr is None and 'lr' not in supplied:
                cfg['lr'] = .001
            if args.weight_decay is None and 'weight_decay' not in supplied:
                cfg['weight_decay'] = .0001
        if args.model:
            cfg['model'] = {'family': args.model}
        cfg['smoke'] = args.smoke
        if args.smoke:
            cfg['epochs'] = args.epochs if args.epochs is not None else 2
            cfg['batch_size'] = args.batch_size if args.batch_size is not None else 8
            cfg['workers'] = args.workers if args.workers is not None else 0
        if args.fast:
            cfg['deterministic'] = False
        if args.no_tensorboard:
            cfg['tensorboard'] = False
        if args.data_root:
            cfg['data_root'] = str(args.data_root.resolve())
        root = args.output_root or (ROOT / 'experiments' / 'smoke' if args.smoke else ROOT / 'checkpoints')
        recipes = [('sgd', lr) for lr in (.05, .1, .15)] + [('adam', lr) for lr in (.0003, .001, .003)] if args.matrix else [(cfg['optimizer'], cfg['lr'])]
        for optimizer, lr in recipes:
            job = copy.deepcopy(cfg)
            job['optimizer'], job['lr'] = optimizer, lr
            if args.matrix:
                job['weight_decay'] = .0005 if optimizer == 'sgd' else .0001
            label = f"{args.run_name}_{dataset}_{job['model']['family']}_{optimizer}_lr{lr:g}_seed{job['seed']}"
            job['run_dir'] = str((root / label).resolve())
            result.append(job)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    from src import engine
    from src.models import build_model, profile_model
    def stop_handler(signum, frame):
        if engine.REQUEST_STOP:
            raise KeyboardInterrupt
        engine.REQUEST_STOP = True
        print('\nSTOP requested: finishing this epoch, then saving last/best. Press Ctrl+C again to stop immediately.', flush=True)
    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, stop_handler)
    try:
        print(f'Python: {sys.executable}', flush=True)
        if args.report:
            from src.reporting import build_reports
            build_reports()
            return 0
        if args.finalize or args.evaluate:
            from src.finalize import finalize, evaluate_final
            if args.finalize:
                finalize(args.finalize, args.stop_after_epochs)
            else:
                evaluate_final(args.evaluate)
            from src.reporting import build_reports
            build_reports()
            return 0
        configurations = configs_from_args(args)
        if args.profile:
            for config in configurations:
                classes = 10 if config['dataset'] == 'cifar10' else 100
                stats = profile_model(build_model(config['model'], classes))
                print(json.dumps({'dataset': config['dataset'], **stats}, indent=2))
            return 0
        if args.search:
            from src.search import search
            for config in configurations:
                search(config, args.trials, args.study_name, args.multi_objective, args.stop_after_epochs)
                if engine.REQUEST_STOP:
                    break
            from src.reporting import build_reports
            build_reports()
            return 0
        completed = []
        for index, config in enumerate(configurations, 1):
            print(f"\nQUEUE {index}/{len(configurations)}: {config['dataset']} {config['optimizer']} LR={config['lr']}\n", flush=True)
            result = engine.run_experiment(config, args.stop_after_epochs)
            if result['status'] == 'interrupted' or engine.REQUEST_STOP:
                break
            completed.append(result)
        if len(completed) == len(configurations) and not args.selection_only and not args.smoke and not args.limit_train and not args.limit_val:
            from src.finalize import finalize
            for dataset in sorted({r['dataset'] for r in completed}):
                candidates = [r for r in completed if r['dataset'] == dataset and not r['synthetic'] and not r['subset']]
                if not candidates:
                    continue
                winner = max(candidates, key=lambda r: r['best_val_accuracy'])
                final_result = finalize(Path(winner['config']['run_dir']), args.stop_after_epochs)
                if final_result['status'] == 'interrupted' or engine.REQUEST_STOP:
                    break
        if (ROOT / 'src' / 'reporting.py').exists():
            from src.reporting import build_reports
            build_reports()
        return 0
    except KeyboardInterrupt:
        print('\nStopped immediately. Repeat the same command to replay the unfinished epoch from last.pt.', flush=True)
        return 130
    except Exception as error:
        print(f'ERROR: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
