'''Single-entry CIFAR experiments with epoch summaries and epoch checkpoints.'''

import argparse
import copy
import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
VERSION = 1
STOP_REQUESTED = False
STATS = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


class TrainingStopped(Exception):
    pass


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, inputs):
        if not self.training or self.probability == 0:
            return inputs
        keep = 1 - self.probability
        mask = torch.rand((inputs.shape[0], 1, 1, 1), device=inputs.device) < keep
        return inputs * mask.to(inputs.dtype) / keep


class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, max(channels // 16, 8), 1)
        self.activate = nn.SiLU()
        self.expand = nn.Conv2d(max(channels // 16, 8), channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        gate = self.sigmoid(self.expand(self.activate(self.reduce(self.pool(inputs)))))
        return inputs * gate


class MixBlock(nn.Module):
    def __init__(self, channels, variant, drop_path):
        super().__init__()
        expanded = channels * 2
        self.pre = nn.Sequential(
            nn.BatchNorm2d(channels), nn.SiLU(),
            nn.Conv2d(channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded), nn.SiLU(),
        )
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        kernel = 5 if variant == 'dualpath' else 3
        self.context = nn.Conv2d(channels, channels, kernel, padding=kernel // 2,
                                 groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.BatchNorm2d(expanded), nn.SiLU(),
            nn.Conv2d(expanded, expanded, 3, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(expanded), nn.SiLU(), ChannelGate(expanded),
            nn.Conv2d(expanded, channels, 1, bias=False),
        )
        self.drop = DropPath(drop_path)

    def forward(self, inputs):
        local, context = self.pre(inputs).chunk(2, dim=1)
        residual = self.mix(torch.cat((self.local(local), self.context(context)), dim=1))
        return inputs + self.drop(residual)


class CompactMixNet(nn.Module):
    '''Custom residual composition; dualpath/localpath form a controlled ablation.'''

    def __init__(self, num_classes=10, variant='dualpath', drop_path=0.1):
        super().__init__()
        if variant not in ('dualpath', 'localpath') or num_classes not in (10, 100):
            raise ValueError('Unsupported architecture or class count')
        widths, depths = (40, 112, 256), (2, 3, 8)
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        layers, incoming, block_number = [], widths[0], 0
        for stage, (channels, depth) in enumerate(zip(widths, depths)):
            if stage:
                layers.append(nn.Sequential(
                    nn.BatchNorm2d(incoming), nn.SiLU(), nn.AvgPool2d(2),
                    nn.Conv2d(incoming, channels, 1, bias=False),
                ))
            for _ in range(depth):
                layers.append(MixBlock(channels, variant,
                                       drop_path * block_number / (sum(depths) - 1)))
                block_number += 1
            incoming = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.BatchNorm2d(incoming), nn.SiLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(incoming, num_classes))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        return self.head(self.body(self.stem(inputs)))


def profile_model(model):
    counts = {'flops': 0}
    handles = []

    def count(module, inputs, output):
        elements = output.numel()
        if isinstance(module, nn.Conv2d):
            operations = elements * (module.in_channels // module.groups) * math.prod(module.kernel_size) * 2
            operations += elements if module.bias is not None else 0
        elif isinstance(module, nn.Linear):
            operations = elements * module.in_features * 2 + elements
        elif isinstance(module, nn.BatchNorm2d):
            operations = elements * 4
        elif isinstance(module, (nn.SiLU, nn.Sigmoid)):
            operations = elements * (6 if isinstance(module, nn.SiLU) else 4)
        elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            operations = inputs[0].numel() + elements
        elif isinstance(module, (MixBlock, ChannelGate)):
            operations = elements
        else:
            operations = 0
        counts['flops'] += operations

    was_training = model.training
    for module in model.modules():
        handles.append(module.register_forward_hook(count))
    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    result = {'parameters': sum(parameter.numel() for parameter in model.parameters()),
              'flops': counts['flops'], 'input': [1, 3, 32, 32],
              'convention': '2 FLOPs/MAC; BN=4, SiLU=6, sigmoid=4; pools/add/gates included'}
    if result['parameters'] > 6_000_000 or result['flops'] >= 1_000_000_000:
        raise ValueError('Architecture exceeds exam budget: {}'.format(result))
    return result


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(states):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'].cpu())
    if states['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in states['cuda']])


@contextmanager
def sample_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def stratified_split(targets, classes, seed):
    labels = np.asarray(targets)
    generator = np.random.default_rng(seed)
    training, validation = [], []
    for label in range(classes):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError('Each class needs at least two training images')
        generator.shuffle(indices)
        held_out = max(1, len(indices) // 10)
        validation.extend(indices[:held_out].tolist())
        training.extend(indices[held_out:].tolist())
    return sorted(training), sorted(validation)


class SeededImages(Dataset):
    def __init__(self, source, indices, transform, seed, epoch=0):
        self.source, self.indices, self.transform = source, indices, transform
        self.seed, self.epoch = seed, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        seed = (self.seed * 1_000_003 + self.epoch * 97_409 + index) % (2 ** 63 - 1)
        with sample_rng(seed):
            image, label = self.source[index]
            return self.transform(image), label


class RemainingBatches(Sampler):
    def __init__(self, length, batch_size, seed, epoch, next_batch=0):
        self.length, self.batch_size, self.next_batch = length, batch_size, next_batch
        generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)
        self.order = torch.randperm(length, generator=generator).tolist()

    def __iter__(self):
        for start in range(self.next_batch * self.batch_size, self.length, self.batch_size):
            yield self.order[start:start + self.batch_size]

    def __len__(self):
        return max(0, math.ceil(self.length / self.batch_size) - self.next_batch)


def image_transforms(name):
    mean, std = STATS[name]
    basic = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    training = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        *basic,
    ])
    return training, transforms.Compose(basic)


def prepare_data(config):
    classes = 10 if config['dataset'] == 'cifar10' else 100
    if config['smoke']:
        source = datasets.FakeData(size=48, image_size=(3, 32, 32), num_classes=classes,
                                   random_offset=0)
        training, validation = list(range(32)), list(range(32, 48))
    else:
        dataset_class = datasets.CIFAR10 if classes == 10 else datasets.CIFAR100
        source = dataset_class(root=config['data_root'], train=True, download=config['download'])
        training, validation = stratified_split(source.targets, classes, config['split_seed'])
    serialized = json.dumps({'train': training, 'validation': validation}, separators=(',', ':'))
    metadata = {'train_indices': training, 'validation_indices': validation,
                'split_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
                'classes': classes, 'synthetic': config['smoke']}
    return source, metadata


def make_loader(dataset, config, epoch=0, next_batch=None):
    generator = torch.Generator().manual_seed(config['seed'] + epoch)
    options = {'num_workers': config['workers'], 'pin_memory': config['device'] == 'cuda',
               'generator': generator, 'persistent_workers': config['workers'] > 0}
    if config['workers']:
        options['prefetch_factor'] = 2
    if next_batch is not None:
        sampler = RemainingBatches(len(dataset), config['batch_size'], config['seed'], epoch, next_batch)
        return DataLoader(dataset, batch_sampler=sampler, **options)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, **options)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path, payload, rotate=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if rotate and path.exists():
        os.replace(path, path.with_name('previous.pt'))
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def recover_checkpoint(directory):
    errors = []
    for name in ('last.pt', 'previous.pt'):
        path = Path(directory) / name
        if not path.exists():
            continue
        try:
            checkpoint = load_checkpoint(path)
            if checkpoint['version'] != VERSION:
                raise ValueError('Unsupported checkpoint version')
            if name == 'previous.pt':
                logging.warning('Recovering previous.pt; latest checkpoint missing or unreadable')
            return checkpoint
        except Exception as error:
            errors.append('{}: {}'.format(path, error))
    if errors:
        raise RuntimeError('No readable checkpoint. ' + '; '.join(errors))
    return None


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def make_optimizer(model, config):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    groups = [{'params': decay, 'weight_decay': config['weight_decay']},
              {'params': no_decay, 'weight_decay': 0.0}]
    if config['optimizer'] == 'sgd':
        return torch.optim.SGD(groups, lr=config['lr'], momentum=0.9, nesterov=True)
    return torch.optim.Adam(groups, lr=config['lr'], betas=(0.9, 0.999), eps=1e-8)


def learning_rate(config, epoch, batch, batches):
    progress = epoch + batch / batches
    warmup = min(5, config['epochs'] // 10)
    if warmup and progress < warmup:
        return config['lr'] * (0.1 + 0.9 * progress / warmup)
    fraction = (progress - warmup) / max(1, config['epochs'] - warmup)
    return config['lr'] * (0.001 + 0.999 * (1 + math.cos(math.pi * fraction)) / 2)


def mixed_batch(images, labels, classes, config, epoch):
    targets = functional.one_hot(labels, classes).float()
    shuffled_labels, coefficient = labels, 1.0
    mode = 'none'
    mixing_until = config['epochs'] - min(30, config['epochs'] // 10)
    choice = random.random() if epoch < mixing_until else 1.0
    if choice < 0.75 and len(labels) > 1:
        permutation = torch.randperm(len(labels), device=images.device)
        shuffled_labels = labels[permutation]
        if choice < 0.5:
            coefficient = float(np.random.beta(1.0, 1.0))
            height, width = images.shape[-2:]
            ratio = math.sqrt(1 - coefficient)
            cut_width, cut_height = int(width * ratio), int(height * ratio)
            center_x, center_y = random.randrange(width), random.randrange(height)
            left, right = max(0, center_x - cut_width // 2), min(width, center_x + cut_width // 2)
            top, bottom = max(0, center_y - cut_height // 2), min(height, center_y + cut_height // 2)
            images = images.clone()
            images[:, :, top:bottom, left:right] = images[permutation, :, top:bottom, left:right]
            coefficient = 1 - (right - left) * (bottom - top) / (height * width)
            mode = 'cutmix'
        else:
            coefficient = float(np.random.beta(0.2, 0.2))
            images = images * coefficient + images[permutation] * (1 - coefficient)
            mode = 'mixup'
        targets = coefficient * targets + (1 - coefficient) * targets[permutation]
    targets = targets * 0.95 + 0.05 / classes
    return images, targets, shuffled_labels, coefficient, mode


@torch.no_grad()
def update_ema(ema, model, updates):
    decay = min(0.999, (1 + updates) / (10 + updates))
    current = model.state_dict()
    for name, value in ema.state_dict().items():
        if value.is_floating_point():
            value.lerp_(current[name], 1 - decay)
        else:
            value.copy_(current[name])


def initial_state():
    return {'epoch': 0, 'next_batch': 0, 'phase': 'train', 'global_step': 0,
            'best_accuracy': -1.0, 'best_epoch': None, 'best_weights': None,
            'train_loss_sum': 0.0, 'train_correct': 0.0, 'train_seen': 0,
            'epoch_seconds': 0.0, 'amp_skips': 0, 'history': []}


def save_progress(directory, model, ema, optimizer, scaler, state, config, metadata):
    if (state['epoch'] < 1 or state['next_batch'] != 0
            or state['phase'] not in ('train', 'test')
            or len(state['history']) != state['epoch']):
        raise ValueError('Checkpoints require a completed training and validation epoch')
    payload = {'version': VERSION, 'source_sha256': source_hash(), 'config': config,
               'split_sha256': metadata['split_sha256'], 'state': copy.deepcopy(state),
               'model': model.state_dict(), 'ema': ema.state_dict(),
               'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
               'rng': capture_rng()}
    atomic_checkpoint(Path(directory) / 'last.pt', payload, rotate=True)


def write_history(directory, history):
    if not history:
        return
    path = Path(directory) / 'metrics.csv'
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def check_stop():
    if STOP_REQUESTED:
        raise TrainingStopped()


@torch.inference_mode()
def evaluate(model, loader, classes, device):
    model.eval()
    loss_sum, correct, top5, seen = 0.0, 0, 0, 0
    confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    for images, labels in loader:
        check_stop()
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device, enabled=device == 'cuda'):
            logits = model(images)
            loss = functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite evaluation loss')
        predicted = logits.argmax(dim=1)
        count = len(labels)
        loss_sum += loss.item() * count
        correct += (predicted == labels).sum().item()
        top5 += (logits.topk(min(5, classes), dim=1).indices == labels[:, None]).any(dim=1).sum().item()
        confusion += torch.bincount(labels * classes + predicted,
                                     minlength=classes * classes).reshape(classes, classes)
        seen += count
    check_stop()
    matrix = confusion.cpu().numpy()
    diagonal = matrix.diagonal().astype(float)
    support, predicted_count = matrix.sum(axis=1), matrix.sum(axis=0)
    recall = diagonal / np.maximum(support, 1)
    precision = diagonal / np.maximum(predicted_count, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {'loss': loss_sum / seen, 'accuracy': 100 * correct / seen,
            'top5_accuracy': 100 * top5 / seen, 'macro_f1': float(100 * f1.mean()),
            'samples': seen, 'confusion_matrix': matrix.tolist(),
            'per_class_recall': (100 * recall).tolist()}


def train_epoch(model, ema, optimizer, scaler, source, metadata, state, config, runtime):
    training_transform, _ = image_transforms(config['dataset'])
    epoch = state['epoch']
    training = SeededImages(source, metadata['train_indices'], training_transform, config['seed'], epoch)
    loader = make_loader(training, config, epoch, state['next_batch'])
    batches = math.ceil(len(training) / config['batch_size'])
    starting_batch = state['next_batch']
    model.train()
    last_batch_time = time.perf_counter()
    check_stop()
    for offset, (images, labels) in enumerate(loader):
        batch = starting_batch + offset
        rate = learning_rate(config, epoch, batch, batches)
        for group in optimizer.param_groups:
            group['lr'] = rate
        images = images.to(config['device'], non_blocking=True)
        labels = labels.to(config['device'], non_blocking=True)
        images, targets, shuffled, coefficient, _ = mixed_batch(
            images, labels, metadata['classes'], config, epoch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=config['device'], enabled=config['device'] == 'cuda'):
            logits = model(images)
            loss = -(targets * functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss; resume from the last durable checkpoint')
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if not scaler.is_enabled() and not torch.isfinite(gradient_norm):
            raise RuntimeError('Non-finite gradients; last durable checkpoint is retained')
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        if skipped:
            state['amp_skips'] += 1
        else:
            state['global_step'] += 1
            update_ema(ema, model, state['global_step'])
        predictions = logits.detach().argmax(dim=1)
        correct = coefficient * (predictions == labels).float().sum()
        correct += (1 - coefficient) * (predictions == shuffled).float().sum()
        state['train_correct'] += correct.item()
        state['train_loss_sum'] += loss.item() * len(labels)
        state['train_seen'] += len(labels)
        state['next_batch'] = batch + 1
        now = time.perf_counter()
        state['epoch_seconds'] += now - last_batch_time
        last_batch_time = now
        runtime['processed_batches'] += 1
        if runtime['stop_after_steps'] and runtime['processed_batches'] >= runtime['stop_after_steps']:
            request_stop(None, None)
        check_stop()
    state['phase'] = 'validate'


def validate_epoch(model, ema, optimizer, scaler, source, metadata, state, config, directory):
    started = time.perf_counter()
    _, validation_transform = image_transforms(config['dataset'])
    validation = SeededImages(source, metadata['validation_indices'], validation_transform, config['seed'])
    loader = make_loader(validation, config)
    raw = evaluate(model, loader, metadata['classes'], config['device'])
    averaged = evaluate(ema, loader, metadata['classes'], config['device'])
    chosen_model, chosen, weights = (ema, averaged, 'ema') if averaged['accuracy'] > raw['accuracy'] else (model, raw, 'raw')
    epoch = state['epoch'] + 1
    if chosen['accuracy'] > state['best_accuracy']:
        state.update(best_accuracy=chosen['accuracy'], best_epoch=epoch, best_weights=weights)
        atomic_checkpoint(Path(directory) / 'best.pt', {
            'version': VERSION, 'source_sha256': source_hash(), 'config': config,
            'model': chosen_model.state_dict(), 'epoch': epoch, 'weights': weights,
            'validation_accuracy': chosen['accuracy'], 'split_sha256': metadata['split_sha256'],
        })
    row = {'epoch': epoch, 'train_loss': state['train_loss_sum'] / state['train_seen'],
           'train_mix_accuracy': 100 * state['train_correct'] / state['train_seen'],
           'validation_loss': raw['loss'], 'validation_accuracy': raw['accuracy'],
           'ema_validation_loss': averaged['loss'], 'ema_validation_accuracy': averaged['accuracy'],
           'best_validation_accuracy': state['best_accuracy'],
           'lr': optimizer.param_groups[0]['lr'], 'train_seconds': state['epoch_seconds'],
           'amp_skips': state['amp_skips'], 'global_step': state['global_step'],
           'epoch_seconds': state['epoch_seconds'] + time.perf_counter() - started}
    state['history'].append(row)
    state.update(epoch=epoch, next_batch=0, train_loss_sum=0.0, train_correct=0.0,
                 train_seen=0, epoch_seconds=0.0, amp_skips=0,
                 phase='test' if epoch == config['epochs'] else 'train')
    save_progress(directory, model, ema, optimizer, scaler, state, config, metadata)
    write_history(directory, state['history'])
    memory = torch.cuda.max_memory_allocated() / 2 ** 30 if config['device'] == 'cuda' else 0
    logging.info('%s/%s/%s EPOCH %d/%d | loss=%.4f train=%.2f%% (mixed) '
                 'val=%.2f%% ema=%.2f%% best=%.2f%%@%d | lr=%.6g '
                 'time=%.1fs GPU_peak=%.2fGiB AMP_skips=%d | checkpoint=last.pt',
                 config['dataset'], config['model'], config['optimizer'], epoch, config['epochs'],
                 row['train_loss'], row['train_mix_accuracy'], raw['accuracy'],
                 averaged['accuracy'], state['best_accuracy'], state['best_epoch'], row['lr'],
                 row['epoch_seconds'], memory, row['amp_skips'])


def final_test(directory, config, runtime):
    best = load_checkpoint(Path(directory) / 'best.pt')
    classes = 10 if config['dataset'] == 'cifar10' else 100
    model = CompactMixNet(classes, config['model']).to(config['device'])
    model.load_state_dict(best['model'])
    _, transform = image_transforms(config['dataset'])
    if config['smoke']:
        source = datasets.FakeData(size=16, image_size=(3, 32, 32), num_classes=classes, random_offset=10_000)
    else:
        dataset_class = datasets.CIFAR10 if classes == 10 else datasets.CIFAR100
        source = dataset_class(root=config['data_root'], train=False,
                               download=config['download'] or runtime['download'])
    loader = make_loader(SeededImages(source, list(range(len(source))), transform, config['seed']), config)
    metrics = evaluate(model, loader, classes, config['device'])
    result = {'dataset': config['dataset'], 'model': config['model'], 'optimizer': config['optimizer'],
              'initial_lr': config['lr'], 'epochs': config['epochs'],
              'best_epoch': best['epoch'], 'selected_weights': best['weights'],
              'best_validation_accuracy': best['validation_accuracy'], 'test': metrics,
              'target_accuracy': config['target'], 'target_met': not config['smoke'] and metrics['accuracy'] >= config['target'],
              'synthetic': config['smoke'], 'profile': profile_model(model),
              'source_sha256': source_hash(), 'completed_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(Path(directory) / 'result.json', result)
    atomic_json(Path(directory) / 'confusion_matrix.json', metrics['confusion_matrix'])
    logging.info('RESULT %s test=%.2f%% target=%.1f%% met=%s macro_F1=%.2f%% | %s',
                 config['dataset'], metrics['accuracy'], config['target'], result['target_met'],
                 metrics['macro_f1'], Path(directory) / 'result.json')
    return result


def run_job(directory, config, runtime):
    directory.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(directory / 'terminal.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(file_handler)
    try:
        checkpoint = recover_checkpoint(directory)
        if checkpoint:
            if checkpoint['source_sha256'] != source_hash() or checkpoint['config'] != config:
                raise ValueError('Resume blocked: code/recipe changed. Use --new for a fresh experiment')
            if (checkpoint['state']['phase'] in ('test', 'complete')
                    and (directory / 'result.json').exists()):
                return json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        seed_everything(config['seed'], not config['fast'])
        device = config['device']
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('This session requires CUDA, but CUDA is unavailable')
        source, metadata = prepare_data({**config, 'download': config['download'] or runtime['download']})
        if checkpoint and checkpoint['split_sha256'] != metadata['split_sha256']:
            raise ValueError('Dataset split changed; cannot safely resume')
        atomic_json(directory / 'split.json', metadata)
        model = CompactMixNet(metadata['classes'], config['model']).to(device)
        profile = profile_model(model)
        ema = copy.deepcopy(model).eval()
        ema.requires_grad_(False)
        optimizer = make_optimizer(model, config)
        scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda', init_scale=4096)
        state = initial_state()
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
            ema.load_state_dict(checkpoint['ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            state = checkpoint['state']
            restore_rng(checkpoint['rng'])
            del checkpoint
        atomic_json(directory / 'config.json', {**config, 'profile': profile})
        write_history(directory, state['history'])
        while state['phase'] != 'complete':
            check_stop()
            if state['phase'] == 'train':
                train_epoch(model, ema, optimizer, scaler, source, metadata, state, config, runtime)
            elif state['phase'] == 'validate':
                validate_epoch(model, ema, optimizer, scaler, source, metadata, state, config, directory)
            elif state['phase'] == 'test':
                result = final_test(directory, config, runtime)
                return result
            else:
                raise ValueError('Unknown checkpoint phase')
        raise RuntimeError('Completed run has no result')
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


@contextmanager
def training_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    acquired = False
    try:
        stream.seek(0, io.SEEK_END)
        if stream.tell() == 0:
            stream.write(b' ')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as error:
            raise RuntimeError('Another trainer owns {}. Do not run two sessions together'.format(path)) from error
        stream.seek(0)
        stream.write(str(os.getpid()).encode().ljust(32))
        stream.flush()
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=('both', 'cifar10', 'cifar100'), default='both')
    parser.add_argument('--model', choices=('dualpath', 'localpath'), default='dualpath')
    parser.add_argument('--optimizer', choices=('sgd', 'adam'), default='sgd')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--resume', default='auto', help='auto or an existing session directory')
    parser.add_argument('--new', action='store_true', help='start a fresh session; do not resume')
    parser.add_argument('--download', action='store_true', help='download missing official CIFAR files')
    parser.add_argument('--matrix', action='store_true', help='SGD 0.05/0.1/0.15 and Adam 0.0003/0.001/0.003')
    parser.add_argument('--fast', action='store_true', help='allow nondeterministic cuDNN for speed')
    parser.add_argument('--profile', action='store_true', help='print all model budgets without training')
    parser.add_argument('--smoke', action='store_true', help='2 epochs of FakeData; NOT an accuracy benchmark')
    parser.add_argument('--stop-after-steps', type=int, default=0, help='controlled stop after N batches in this invocation')
    args = parser.parse_args(argv)
    for key in ('epochs', 'batch_size'):
        if getattr(args, key) <= 0:
            parser.error('--{} must be positive'.format(key.replace('_', '-')))
    if args.workers is not None and args.workers < 0:
        parser.error('--workers must be nonnegative')
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error('--lr must be finite and positive')
    if args.seed < 0 or args.split_seed < 0 or args.seed >= 2 ** 32 or args.split_seed >= 2 ** 32:
        parser.error('seeds must be in [0, 2**32)')
    if args.stop_after_steps < 0:
        parser.error('--stop-after-steps must be nonnegative')
    if args.new and args.resume != 'auto':
        parser.error('--new and explicit --resume cannot be combined')
    if args.matrix and args.lr is not None:
        parser.error('--matrix defines its own learning rates')
    return args


def create_session(args, output_root):
    identity = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    directory = output_root / identity
    directory.mkdir(parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device != 'auto':
        device = args.device
    selected = ('cifar10', 'cifar100') if args.dataset == 'both' else (args.dataset,)
    settings = [(args.optimizer, args.lr or (0.1 if args.optimizer == 'sgd' else 0.001))]
    if args.matrix:
        settings = [('sgd', rate) for rate in (0.05, 0.1, 0.15)]
        settings += [('adam', rate) for rate in (0.0003, 0.001, 0.003)]
    jobs = []
    for name in selected:
        for optimizer, rate in settings:
            config = {'dataset': name, 'model': args.model, 'optimizer': optimizer, 'lr': rate,
                      'weight_decay': 5e-4 if optimizer == 'sgd' else 1e-4,
                      'epochs': 2 if args.smoke else args.epochs,
                      'batch_size': min(args.batch_size, 8) if args.smoke else args.batch_size,
                      'workers': args.workers if args.workers is not None else (0 if args.smoke else 2),
                      'seed': args.seed, 'split_seed': args.split_seed,
                      'device': device, 'fast': args.fast, 'download': args.download,
                      'data_root': str(args.data_root.resolve()), 'smoke': args.smoke,
                      'target': 98.0 if name == 'cifar10' else 95.0}
            label = '{}_{}_{}_lr{}'.format(name, args.model, optimizer, format(rate, '.6g'))
            jobs.append({'name': label, 'config': config})
    session = {'version': VERSION, 'source_sha256': source_hash(), 'status': 'running',
               'created_utc': datetime.now(timezone.utc).isoformat(), 'jobs': jobs,
               'environment': {'python': sys.version, 'torch': torch.__version__,
                               'cuda': torch.version.cuda, 'device': device,
                               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    atomic_json(directory / 'session.json', session)
    atomic_json(output_root / 'latest_session.json', {'directory': str(directory.resolve())})
    return directory, session


def choose_session(args, output_root):
    if not args.new:
        path = None
        if args.resume != 'auto':
            path = Path(args.resume).resolve()
        elif (output_root / 'latest_session.json').exists():
            pointer = json.loads((output_root / 'latest_session.json').read_text(encoding='utf-8'))
            path = Path(pointer['directory'])
        if path is not None:
            try:
                session = json.loads((path / 'session.json').read_text(encoding='utf-8'))
            except FileNotFoundError:
                if args.resume != 'auto':
                    raise
                return create_session(args, output_root)
            if session['status'] != 'complete' or args.resume != 'auto':
                if session['source_sha256'] != source_hash() or session['version'] != VERSION:
                    if args.resume != 'auto':
                        raise ValueError('Session code changed; use --new instead of resuming')
                    return create_session(args, output_root)
                return path, session
    return create_session(args, output_root)


def main(argv=None):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    args = parse_args(argv)
    if args.profile:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        profiles = {}
        for variant in ('dualpath', 'localpath'):
            for classes in (10, 100):
                profiles['{}_{}'.format(variant, classes)] = profile_model(CompactMixNet(classes, variant))
        print(json.dumps(profiles, indent=2))
        return 0
    output_root = (args.output_root or ROOT / 'outputs' / ('rebuilt_smoke' if args.smoke else 'rebuilt')).resolve()
    runtime = {'stop_after_steps': args.stop_after_steps, 'processed_batches': 0, 'download': args.download}
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, request_stop)
    if os.name == 'nt' and hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, request_stop)
    try:
        with training_lock(output_root / 'training.lock'):
            directory, session = choose_session(args, output_root)
            handler = logging.FileHandler(directory / 'session.log', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            logging.getLogger().addHandler(handler)
            try:
                results = []
                for job in session['jobs']:
                    check_stop()
                    config = job['config']
                    if config['device'] == 'cpu':
                        torch.set_num_threads(min(4, os.cpu_count() or 1))
                    result = run_job(directory / job['name'], config, runtime)
                    results.append({'run': job['name'], 'test_accuracy': result['test']['accuracy'],
                                    'best_validation_accuracy': result['best_validation_accuracy'],
                                    'target_met': result['target_met'], 'synthetic': result['synthetic']})
                    atomic_json(directory / 'results.json', results)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                session['status'] = 'complete'
                session['completed_utc'] = datetime.now(timezone.utc).isoformat()
                atomic_json(directory / 'session.json', session)
                return 0
            except TrainingStopped:
                session['status'] = 'paused'
                atomic_json(directory / 'session.json', session)
                logging.info('STOPPED | Resume from the last completed epoch: %s', directory)
                return 0
            except Exception:
                session['status'] = 'failed'
                atomic_json(directory / 'session.json', session)
                logging.exception('FAILED. The last durable checkpoint is retained; no automatic restart loop')
                return 1
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()
    except Exception:
        logging.exception('Cannot start session')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
