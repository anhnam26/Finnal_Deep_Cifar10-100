"""Persistent ask/tell Optuna studies; interrupted trials reuse their checkpoints."""
import copy
from pathlib import Path
import re
import time

import optuna
from optuna.trial import TrialState

from .config import ROOT
from . import engine
from .models import build_model, profile_model


def sample_config(trial, base):
    config = copy.deepcopy(base)
    family = trial.suggest_categorical('family', ['compactmix'] if base['smoke'] else base['search_families'])
    model = {'family': family}
    if family == 'compactmix':
        widths = {'small': [24, 64, 160], 'medium': [32, 80, 192], 'full': [40, 112, 256]}
        model.update(widths=widths[trial.suggest_categorical('width', list(widths))],
                     depths=[trial.suggest_int('depth_0', 1, 3), trial.suggest_int('depth_1', 2, 4), trial.suggest_int('depth_2', 4, 8)],
                     context_kernel=trial.suggest_categorical('context_kernel', [3, 5, 7]),
                     attention=trial.suggest_categorical('attention', ['none', 'se', 'eca']),
                     norm=trial.suggest_categorical('norm', ['bn', 'gn', 'ln']),
                     activation=trial.suggest_categorical('activation', ['relu', 'silu', 'gelu']),
                     stem=trial.suggest_categorical('stem', ['simple', 'stacked']),
                     expansion=trial.suggest_categorical('expansion', [2, 4]),
                     residual_scale=trial.suggest_categorical('residual_scale', [.5, 1.]),
                     pool=trial.suggest_categorical('pool', ['avg', 'avgmax']),
                     dropout=trial.suggest_categorical('dropout', [0., .1]),
                     drop_path=trial.suggest_float('drop_path', 0., .15))
        if base['smoke']:
            model['widths'], model['depths'], model['expansion'] = [16, 32, 64], [1, 1, 1], 2
    config['model'] = model
    kind = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'adamw'])
    config['optimizer'] = kind
    config['lr'] = trial.suggest_float('lr_' + kind, .03 if kind == 'sgd' else .0002,
                                      .2 if kind == 'sgd' else .003, log=True)
    config['weight_decay'] = trial.suggest_float('weight_decay', 5e-5, 1e-3, log=True)
    config['batch_size'] = base['batch_size'] if base['smoke'] else trial.suggest_categorical('batch_size', [64, 128])
    config['label_smoothing'] = trial.suggest_categorical('label_smoothing', [0., .05, .1])
    config['augmentation'] = trial.suggest_categorical('augmentation', ['basic', 'autoaugment', 'randaugment', 'trivialaugment'])
    mixing = trial.suggest_categorical('mixing', ['none', 'mixup', 'cutmix'])
    config['mixup'], config['cutmix'] = (.2 if mixing == 'mixup' else 0.), (1. if mixing == 'cutmix' else 0.)
    config['scheduler'] = trial.suggest_categorical('scheduler', ['cosine', 'onecycle'])
    config['warmup'] = trial.suggest_categorical('warmup', [0, 5])
    config['ema_decay'] = trial.suggest_categorical('ema_decay', [.99, .999])
    return config


def search(base, trials=20, study_name=None, multi_objective=False,
           stop_after_epochs=None, storage_path=None):
    if trials < 1:
        raise ValueError('trials must be positive')
    name = study_name or f"{base['dataset']}_{'smoke_' if base['smoke'] else ''}architecture_v2_{'pareto' if multi_objective else 'accuracy'}"
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('Study name must be safe as a local directory name')
    database = Path(storage_path or ROOT / 'experiments' / 'optuna.db').resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    storage = 'sqlite:///' + database.as_posix()
    root = Path(base['run_dir']).parent / ('search_' + name)
    contract = {'base': base, 'multi_objective': multi_objective,
                'space_version': 2, 'source_sha256': engine.source_fingerprint()}
    with engine.run_lock(root):
        sampler = optuna.samplers.NSGAIISampler(seed=base['seed']) if multi_objective else optuna.samplers.TPESampler(seed=base['seed'])
        study = optuna.create_study(storage=storage, study_name=name, load_if_exists=True,
                                   directions=['maximize', 'minimize'] if multi_objective else ['maximize'],
                                   sampler=sampler, pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5))
        if study.user_attrs.get('contract') not in (None, contract):
            raise ValueError('Study configuration/source changed; use a new study-name')
        study.set_user_attr('contract', contract)
        terminal = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
        while sum(t.state in terminal for t in study.trials) < trials:
            pending = [t for t in study.trials if t.state == TrialState.RUNNING]
            if pending:
                # Optuna exposes Trial(study, trial_id); FrozenTrial retains the ID.
                trial = optuna.trial.Trial(study, pending[0]._trial_id)
                print(f'OPTUNA RESUME trial={trial.number} study={name}', flush=True)
            else:
                seed = base['seed'] + len(study.trials)
                study.sampler = optuna.samplers.NSGAIISampler(seed=seed) if multi_objective else optuna.samplers.TPESampler(seed=seed)
                trial = study.ask()
            started = time.monotonic()
            try:
                config = trial.user_attrs.get('config')
                if config is None:
                    config = sample_config(trial, base)
                    config['seed'] = base['seed']  # Comparable initialization across candidates.
                    config['run_dir'] = str(root / f'trial_{trial.number:05d}')
                    trial.set_user_attr('config', config)
                profile = profile_model(build_model(config['model'], 10 if config['dataset'] == 'cifar10' else 100), enforce=False)
                trial.set_user_attr('architecture', config['model'])
                trial.set_user_attr('hyperparameters', config)
                trial.set_user_attr('parameters', profile['parameters'])
                trial.set_user_attr('flops', profile['flops'])
                trial.set_user_attr('seed', config['seed'])
                trial.set_user_attr('checkpoint_path', str(Path(config['run_dir']) / 'best.pt'))
                trial.set_user_attr('test_accuracy', None)
                if profile['parameters'] > 6_000_000 or profile['flops'] >= 1e9:
                    raise optuna.TrialPruned('Candidate exceeds exam budget')
                def callback(epoch, accuracy):
                    trial.set_user_attr('best_validation_accuracy', accuracy)
                    if not multi_objective:
                        trial.report(accuracy, epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned(f'Pruned on validation at epoch {epoch}')
                result = engine.run_experiment(config, stop_after_epochs, callback)
                if result['status'] == 'interrupted':
                    print('OPTUNA paused at durable epoch; repeat the same command to resume this trial.', flush=True)
                    return study
                trial.set_user_attr('training_time', result['training_time'])
                trial.set_user_attr('best_validation_accuracy', result['best_val_accuracy'])
                values = [result['best_val_accuracy'], result['parameters']] if multi_objective else result['best_val_accuracy']
                study.tell(trial, values)
            except optuna.TrialPruned as error:
                trial.set_user_attr('failure_reason', str(error))
                trial.set_user_attr('training_time', time.monotonic()-started)
                study.tell(trial, state=TrialState.PRUNED)
            except KeyboardInterrupt:
                print('Trial left RUNNING for checkpoint recovery.', flush=True)
                raise
            except Exception as error:
                trial.set_user_attr('failure_reason', f'{type(error).__name__}: {error}')
                trial.set_user_attr('training_time', time.monotonic()-started)
                study.tell(trial, state=TrialState.FAIL)
                raise
            if engine.REQUEST_STOP:
                break
        print(f'OPTUNA {name}: {len(study.trials)} trials; database={database}', flush=True)
        return study
