import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def default_config(dataset='cifar10'):
    if dataset not in ('cifar10', 'cifar100'):
        raise ValueError('Dataset must be cifar10 or cifar100')
    return {'dataset': dataset, 'model': {'family': 'compactmix'},
            'epochs': 600, 'batch_size': 128, 'workers': 2, 'seed': 42,
            'split_seed': 2026, 'optimizer': 'sgd', 'lr': .1,
            'weight_decay': .0005, 'momentum': .9, 'warmup': 5,
            'scheduler': 'cosine', 'label_smoothing': .05,
            'mixup': .2, 'cutmix': 1., 'mix_probability': .75,
            'augmentation': 'autoaugment', 'random_erasing': 0.,
            'ema_decay': .999, 'grad_clip': 5., 'amp': True,
            'deterministic': True, 'tensorboard': True, 'device': 'auto',
            'data_root': str(ROOT / ('datasets' if (ROOT / 'datasets').exists() else 'data')),
            'run_dir': str(ROOT / 'checkpoints' / dataset), 'download': True,
            'smoke': False, 'limit_train': 0, 'limit_val': 0,
            'phase': 'selection', 'log_every': 100, 'final_weights': 'ema',
            'search_families': ['compactmix', 'wrn', 'preactresnet', 'convnext']}


def merge_config(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if key not in result:
            raise ValueError(f'Unknown configuration field: {key}')
        if key == 'model':
            result[key].update(value)
        else:
            result[key] = value
    return result
