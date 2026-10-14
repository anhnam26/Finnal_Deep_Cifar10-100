"""Reproducible splits and sample-seeded augmentations for epoch replay."""
from contextlib import contextmanager
import hashlib
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

STATS = {
    'cifar10': ((.4914, .4822, .4465), (.2470, .2435, .2616)),
    'cifar100': ((.5071, .4867, .4408), (.2675, .2565, .2761))}


def stratified_split(targets, classes, seed):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train, val = [], []
    for label in range(classes):
        indices = np.flatnonzero(targets == label)
        if len(indices) < 2:
            raise ValueError('Each class needs at least two samples')
        rng.shuffle(indices)
        n = max(1, len(indices) // 10)
        val.extend(indices[:n].tolist())
        train.extend(indices[n:].tolist())
    return sorted(train), sorted(val)


def prepare_data(config):
    classes = 10 if config['dataset'] == 'cifar10' else 100
    if config['smoke']:
        source = datasets.FakeData(size=48, image_size=(3, 32, 32), num_classes=classes)
        training, validation = list(range(32)), list(range(32, 48))
    else:
        constructor = datasets.CIFAR10 if classes == 10 else datasets.CIFAR100
        source = constructor(root=config['data_root'], train=True, download=config['download'])
        training, validation = stratified_split(source.targets, classes, config['split_seed'])
    if config['phase'] == 'final':
        training = sorted(training + validation)
        validation = []
    if config['limit_train']:
        training = training[:config['limit_train']]
    if config['limit_val']:
        validation = validation[:config['limit_val']]
    payload = {'training': training, 'validation': validation}
    return source, {**payload, 'classes': classes,
                    'sha256': hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
                    'synthetic': config['smoke'],
                    'subset': bool(config['limit_train'] or config['limit_val'])}


@contextmanager
def sample_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def image_transforms(config, training):
    mean, std = STATS[config['dataset']]
    ops = []
    if training:
        ops = [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
               transforms.RandomHorizontalFlip()]
        kind = config['augmentation']
        if kind == 'autoaugment':
            ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        elif kind == 'randaugment':
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif kind == 'trivialaugment':
            ops.append(transforms.TrivialAugmentWide())
        elif kind != 'basic':
            raise ValueError(f'Unknown augmentation: {kind}')
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if training and config['random_erasing']:
        ops.append(transforms.RandomErasing(p=config['random_erasing']))
    return transforms.Compose(ops)


class SeededImages(Dataset):
    def __init__(self, source, indices, transform, seed, epoch):
        self.source, self.indices, self.transform = source, indices, transform
        self.seed, self.epoch = seed, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        seed = (self.seed * 1_000_003 + self.epoch * 97_409 + index) % (2**63-1)
        with sample_rng(seed):
            image, label = self.source[index]
            return self.transform(image), label


def make_loader(source, indices, config, epoch, training=False):
    dataset = SeededImages(source, indices, image_transforms(config, training), config['seed'], epoch)
    generator = torch.Generator().manual_seed(config['seed'] + epoch * 1_000_003)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=training,
                      num_workers=config['workers'], generator=generator,
                      pin_memory=config['device'] == 'cuda', persistent_workers=False,
                      **({'prefetch_factor': 2} if config['workers'] else {}))
