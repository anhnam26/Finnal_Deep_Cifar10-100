"""Freeze validation decisions, retrain from scratch, evaluate test once."""
import copy
import hashlib
import json
from pathlib import Path

from torchvision import datasets

from . import engine
from .datasets import make_loader
from .models import build_model


def document(directory):
    path = Path(directory).resolve() / 'result.json'
    if not path.exists():
        raise ValueError('Run must be completed before finalization/evaluation')
    return json.loads(path.read_text(encoding='utf-8'))


def finalize(selection_directory, stop_after_epochs=None, verbose=False):
    selection_directory = Path(selection_directory).resolve()
    result = document(selection_directory)
    if result['phase'] != 'selection' or result['status'] != 'complete':
        raise ValueError('Require a completed selection experiment')
    if result['synthetic'] or result['subset']:
        raise ValueError('Cannot finalize synthetic/subset checks as benchmark evidence')
    config = copy.deepcopy(result['config'])
    config['phase'] = 'final'
    config['epochs'] = result['best_epoch']
    config['final_weights'] = result['best_kind']
    directory = selection_directory.with_name(selection_directory.name + '_final')
    config['run_dir'] = str(directory)
    freeze = {'selection_run': str(selection_directory), 'selection_validation_accuracy': result['best_val_accuracy'],
              'selection_best_epoch': result['best_epoch'], 'config': config,
              'source_sha256': engine.source_fingerprint()}
    directory.mkdir(parents=True, exist_ok=True)
    freeze_path = directory / 'frozen.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text(encoding='utf-8')) != freeze:
            raise ValueError('Frozen final recipe changed; existing final run preserved')
    else:
        engine.atomic_json(freeze_path, freeze)
    if verbose:
        print(f'FINAL FREEZE {config["dataset"]}: retrain all 50k for {config["epochs"]} frozen epochs; weights={config["final_weights"]}', flush=True)
    trained = engine.run_experiment(config, stop_after_epochs)
    if trained['status'] == 'interrupted':
        return trained
    return evaluate_final(directory, verbose=verbose)


def evaluate_final(directory, verbose=True):
    directory = Path(directory).resolve()
    with engine.run_lock(directory):
        result = document(directory)
        config = result['config']
        if result['phase'] != 'final' or not (directory / 'frozen.json').exists():
            raise ValueError('Official test requires completed frozen final retraining')
        if result['synthetic'] or result['subset']:
            raise ValueError('Synthetic/subset runs cannot be official test results')
        digest = hashlib.sha256((directory / 'best.pt').read_bytes()).hexdigest()
        if result['test_accuracy'] is not None:
            if result['test_checkpoint_sha256'] != digest:
                raise ValueError('Test checkpoint changed after evaluation; refusing reuse')
            if verbose:
                print(f'CACHED TEST {config["dataset"]}: {result["test_accuracy"]:.2f}%', flush=True)
            return result
        freeze = json.loads((directory / 'frozen.json').read_text(encoding='utf-8'))
        if freeze['source_sha256'] != engine.source_fingerprint() or freeze['config'] != config:
            raise ValueError('Frozen source/config no longer matches')
        classes = 10 if config['dataset'] == 'cifar10' else 100
        model = build_model(config['model'], classes).to(config['device'])
        model.load_state_dict(engine.load_checkpoint(directory / 'best.pt')['model'])
        constructor = datasets.CIFAR10 if classes == 10 else datasets.CIFAR100
        source = constructor(root=config['data_root'], train=False, download=config['download'])
        loader = make_loader(source, list(range(len(source))), config, 0)
        metrics = engine.evaluate(model, loader, config['device'], extended=True)
        result.update(test_accuracy=metrics['accuracy'], test_metrics=metrics, test_checkpoint_sha256=digest)
        target = 98. if classes == 10 else 90.
        result['target_met'] = metrics['accuracy'] >= target
        engine.atomic_json(directory / 'result.json', result)
        if verbose:
            print(f'TEST {config["dataset"]} top1={metrics["accuracy"]:.2f}% top5={metrics["top5_accuracy"]:.2f}% macroF1={metrics["macro_f1"]:.4f} target_met={result["target_met"]}', flush=True)
        return result
