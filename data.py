"""Official CIFAR data; stratified validation is selected ONLY from training data."""

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATASETS = {"cifar10": (datasets.CIFAR10, 10), "cifar100": (datasets.CIFAR100, 100)}
# Published, fixed preprocessing constants; not computed from test images.
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def stratified_split(targets, num_classes, val_per_class, seed):
    labels = np.asarray(targets)
    if set(np.unique(labels)) != set(range(num_classes)):
        raise ValueError("Dataset labels do not match expected classes")
    if val_per_class <= 0:
        raise ValueError("val_per_class must be positive")
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(num_classes):
        indices = np.flatnonzero(labels == label)
        if len(indices) <= val_per_class:
            raise ValueError("Validation leaves no training samples for a class")
        rng.shuffle(indices)
        val.extend(indices[:val_per_class].tolist())
        train.extend(indices[val_per_class:].tolist())
    return sorted(train), sorted(val)


def split_fingerprint(train, val):
    digest = hashlib.sha256()
    digest.update(np.asarray(train, dtype="<i8").tobytes())
    digest.update(b"validation")
    digest.update(np.asarray(val, dtype="<i8").tobytes())
    return digest.hexdigest()


def build_transforms(dataset, augment=True):
    mean, std = STATS[dataset]
    basic = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    train = transforms.Compose(
        ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
         if augment else []) + basic
    )
    return train, transforms.Compose(basic)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size, workers, shuffle=False, seed=42, use_cuda=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=use_cuda, worker_init_fn=seed_worker, generator=generator,
        persistent_workers=False, drop_last=False,
    )


def build_train_loaders(config, use_cuda=False):
    name = config["dataset"]
    cls, num_classes = DATASETS[name]
    root = Path(config["data_root"])
    train_tf, eval_tf = build_transforms(name, config["augment"])
    source = cls(root=str(root), train=True, download=config["download"], transform=train_tf)
    if config["full_train"]:
        train_indices, val_indices = list(range(len(source))), []
    else:
        train_indices, val_indices = stratified_split(
            source.targets, num_classes, 500 if num_classes == 10 else 50, config["split_seed"],
        )
    train_loader = make_loader(
        Subset(source, train_indices), config["batch_size"], config["workers"],
        True, config["seed"], use_cuda,
    )
    val_loader = None
    if val_indices:
        # Separate dataset objects prevent train augmentations leaking into validation.
        validation = cls(root=str(root), train=True, download=False, transform=eval_tf)
        val_loader = make_loader(
            Subset(validation, val_indices), config["batch_size"], config["workers"],
            False, config["seed"], use_cuda,
        )
    metadata = {
        "dataset": name, "num_classes": num_classes, "split_seed": config["split_seed"],
        "train_count": len(train_indices), "validation_count": len(val_indices),
        "fingerprint": split_fingerprint(train_indices, val_indices),
        "train_indices": train_indices, "validation_indices": val_indices,
    }
    return train_loader, val_loader, metadata


def build_test_loader(config, use_cuda=False):
    cls, _ = DATASETS[config["dataset"]]
    _, transform = build_transforms(config["dataset"], augment=False)
    dataset = cls(root=config["data_root"], train=False, download=config["download"], transform=transform)
    return make_loader(dataset, config["batch_size"], config["workers"], use_cuda=use_cuda)