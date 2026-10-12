"""Epoch-atomic training: two checkpoints, strict automatic resume, readable logs."""
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .config import ROOT
from .datasets import prepare_data, make_loader
from .models import build_model, profile_model
from .recipes import learning_rate, make_optimizer, mix_batch, update_ema

REQUEST_STOP = False
CHECKPOINT_VERSION = 2


def source_fingerprint():
    digest = hashlib.sha256()
    for name in ('config.py', 'datasets.py', 'models.py', 'timm_models.py', 'recipes.py', 'engine.py'):
        digest.update((ROOT / 'src' / name).read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def environment(config):
    return {'python': platform.python_version(), 'pytorch': torch.__version__,
            'numpy': np.__version__, 'cuda': torch.version.cuda,
            'device': config['device'], 'platform': platform.system(),
            'gpu': torch.cuda.get_device_name(0) if config['device'] == 'cuda' else None}


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(states):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'].cpu())
    if states['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in states['cuda']])


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.train.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another trainer owns {directory}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb' if binary else 'w',
                            **({} if binary else {'encoding': 'utf-8', 'newline': ''})) as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, payload):
    atomic_write(path, lambda stream: json.dump(payload, stream, indent=2, allow_nan=False))


def atomic_torch(path, payload):
    atomic_write(path, lambda stream: torch.save(payload, stream), binary=True)


def load_checkpoint(path):
    # Only deserialize checkpoints written by this trusted repository.
    return torch.load(path, map_location='cpu', weights_only=False)


def cpu_weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_logger(directory):
    logger = logging.getLogger('cifar.' + str(directory))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(directory / 'training.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def duration(seconds):
    seconds = int(seconds)
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def train_epoch(model, ema, optimizer, scaler, source, split, config, epoch, global_step):
    loader = make_loader(source, split['training'], config, epoch, training=True)
    model.train()
    loss_sum = correct = seen = skips = 0
    for batch, (images, labels) in enumerate(loader):
        images, labels = images.to(config['device'], non_blocking=True), labels.to(config['device'], non_blocking=True)
        images, targets, paired, coefficient = mix_batch(images, labels, split['classes'], config, epoch)
        lr = learning_rate(config, epoch, batch, len(loader))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(config['device'], enabled=scaler.is_enabled()):
            logits = model(images)
            loss = -(targets * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss; last completed checkpoint preserved')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < previous_scale
        skips += int(skipped)
        if not skipped:
            global_step += 1
            update_ema(ema, model, global_step, config['ema_decay'])
        n = labels.numel()
        loss_sum += loss.item() * n
        prediction = logits.argmax(1)
        correct += (coefficient * prediction.eq(labels).sum().item()
                    + (1-coefficient) * prediction.eq(paired).sum().item())
        seen += n
    return {'train_loss': loss_sum/seen, 'train_mixed_accuracy': 100*correct/seen,
            'lr': lr, 'amp_skips': skips}, global_step


@torch.no_grad()
def evaluate(model, loader, device, extended=False):
    model.eval()
    loss_sum = correct = seen = 0
    confusion = None
    top5 = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
        predictions = logits.argmax(1)
        correct += predictions.eq(labels).sum().item()
        seen += labels.numel()
        if extended:
            classes = logits.shape[1]
            if confusion is None:
                confusion = torch.zeros(classes, classes, dtype=torch.long)
            confusion += torch.bincount((labels*classes+predictions).cpu(), minlength=classes**2).reshape(classes, classes)
            top5 += logits.topk(min(5, classes), 1).indices.eq(labels[:, None]).any(1).sum().item()
    if seen == 0:
        raise ValueError('Evaluation split is empty')
    result = {'loss': loss_sum/seen, 'accuracy': 100*correct/seen}
    if extended:
        tp = confusion.diag().float()
        recall = tp / confusion.sum(1).clamp_min(1)
        precision = tp / confusion.sum(0).clamp_min(1)
        f1 = 2*precision*recall / (precision+recall).clamp_min(1e-12)
        result.update(top5_accuracy=100*top5/seen, macro_f1=f1.mean().item(),
                      per_class_recall=recall.tolist(), confusion_matrix=confusion.tolist())
    return result


def write_history(directory, history):
    if not history:
        return
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    atomic_write(directory / 'metrics.csv', write)


def validate_config(config):
    if config['phase'] not in ('selection', 'final'):
        raise ValueError('phase must be selection or final')
    if min(config['epochs'], config['batch_size']) < 1 or config['workers'] < 0:
        raise ValueError('Invalid epoch/batch/worker settings')
    if not 0 <= config['label_smoothing'] < 1 or not 0 <= config['mix_probability'] <= 1:
        raise ValueError('Invalid label smoothing or mixing probability')
    if config['lr'] <= 0 or config['weight_decay'] < 0 or config['grad_clip'] <= 0:
        raise ValueError('Invalid learning rate, weight decay or gradient clipping')
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the selected Python; use --device cpu for debugging')


def run_experiment(config, stop_after_epochs=None, epoch_callback=None):
    config = copy.deepcopy(config)
    validate_config(config)
    directory = Path(config['run_dir']).resolve()
    config['run_dir'] = str(directory)
    with run_lock(directory):
        logger = make_logger(directory)
        writer = None
        try:
            identity, env = source_fingerprint(), environment(config)
            last = directory / 'last.pt'
            previous = load_checkpoint(last) if last.exists() else None
            if previous:
                if previous['version'] != CHECKPOINT_VERSION or previous['source_sha256'] != identity:
                    raise ValueError('Changed training source/version; use a new --run-name, existing checkpoints preserved')
                if previous['config'] != config:
                    raise ValueError('Changed configuration; resume rejected. Restore settings or use --run-name')
                if previous['environment'] != env:
                    raise ValueError('Changed runtime environment; exact resume requires the recorded environment')
                if (directory / 'result.json').exists() and previous['epoch'] == config['epochs']:
                    return json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            seed_everything(config['seed'], config['deterministic'])
            source, split = prepare_data(config)
            if previous and previous['split_sha256'] != split['sha256']:
                raise ValueError('Changed dataset split; resume rejected')
            model = build_model(config['model'], split['classes']).to(config['device'])
            profile = profile_model(model)
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            optimizer = make_optimizer(model, config)
            scaler = torch.amp.GradScaler('cuda', enabled=config['device'] == 'cuda' and config['amp'], init_scale=4096.)
            history, best_weights = [], None
            best_needs_repair = False
            epoch, global_step, best_accuracy, best_epoch, best_kind, elapsed_seconds = 0, 0, -1., 0, 'raw', 0.
            if previous:
                model.load_state_dict(previous['model'])
                ema.load_state_dict(previous['ema'])
                optimizer.load_state_dict(previous['optimizer'])
                scaler.load_state_dict(previous['scaler'])
                epoch, global_step = previous['epoch'], previous['global_step']
                history = previous['history']
                best_accuracy, best_epoch, best_kind = previous['best_accuracy'], previous['best_epoch'], previous['best_kind']
                best_weights, elapsed_seconds = previous['best_weights'], previous['elapsed_seconds']
                restore_rng(previous['rng'])
                best_path = directory / 'best.pt'
                try:
                    best_document = load_checkpoint(best_path) if best_path.exists() else None
                except Exception:
                    best_document = None
                if (best_document is None or best_document['epoch'] != best_epoch
                        or best_document['accuracy'] != best_accuracy):
                    best_needs_repair = True
            try:
                git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                git_commit = 'unavailable'
            atomic_json(directory / 'config.json', config)
            atomic_json(directory / 'split.json', split)
            atomic_json(directory / 'metadata.json', {'environment': env, 'python_executable': sys.executable,
                        'git_commit': git_commit, 'source_sha256': identity, 'profile': profile,
                        'started_at_utc': previous['started_at_utc'] if previous else datetime.now(timezone.utc).isoformat()})
            metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
            completed_this_call = 0
            while epoch < config['epochs']:
                started = time.monotonic()
                if config['device'] == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                metrics, global_step = train_epoch(model, ema, optimizer, scaler, source, split, config, epoch, global_step)
                if config['phase'] == 'selection':
                    loader = make_loader(source, split['validation'], config, 0)
                    raw = evaluate(model, loader, config['device'])
                    averaged = evaluate(ema, loader, config['device'])
                    score, kind, selected = raw['accuracy'], 'raw', model
                    if averaged['accuracy'] > score:
                        score, kind, selected = averaged['accuracy'], 'ema', ema
                    metrics.update(val_loss=raw['loss'], val_accuracy=raw['accuracy'],
                                   ema_val_loss=averaged['loss'], ema_val_accuracy=averaged['accuracy'])
                    improved = score > best_accuracy
                else:
                    selected = ema if config['final_weights'] == 'ema' else model
                    score, kind, improved = -1., config['final_weights'] + '_fixed_epoch', True
                epoch += 1
                if improved:
                    best_accuracy, best_epoch, best_kind, best_weights = score, epoch, kind, cpu_weights(selected)
                record = {'epoch': epoch, **metrics, 'best_val_accuracy': best_accuracy, 'best_epoch': best_epoch}
                history.append(record)
                seconds = time.monotonic() - started
                elapsed_seconds += seconds
                if config['tensorboard'] and writer is None:
                    from torch.utils.tensorboard import SummaryWriter
                    writer = SummaryWriter(str(directory / 'tensorboard'), purge_step=epoch)
                payload = {'version': CHECKPOINT_VERSION, 'source_sha256': identity, 'environment': env,
                           'config': config, 'split_sha256': split['sha256'], 'epoch': epoch,
                           'global_step': global_step, 'model': cpu_weights(model), 'ema': cpu_weights(ema),
                           'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                           'rng': capture_rng(), 'history': history, 'best_accuracy': best_accuracy,
                           'best_epoch': best_epoch, 'best_kind': best_kind, 'best_weights': best_weights,
                           'elapsed_seconds': elapsed_seconds, 'started_at_utc': metadata['started_at_utc']}
                best_saved = improved or best_needs_repair
                if best_saved:
                    atomic_torch(directory / 'best.pt', {'model': best_weights, 'epoch': best_epoch,
                                 'accuracy': best_accuracy, 'kind': best_kind, 'config': config})
                    best_needs_repair = False
                atomic_torch(last, payload)
                write_history(directory, history)
                if writer:
                    for key, value in record.items():
                        writer.add_scalar(key, value, epoch)
                    writer.flush()
                vram = torch.cuda.max_memory_allocated()/2**30 if config['device'] == 'cuda' else 0.
                logger.info('%s %s %s/%s(base_lr=%g) | EPOCH %d/%d | loss=%.4f mixed_acc=%.2f%% val_raw=%s val_ema=%s best=%s @%d | lr=%.6g time=%s total_ETA=%s VRAM=%.2fGB AMP_skips=%d | saved last.pt%s',
                            config['dataset'], config['phase'], config['model']['family'], config['optimizer'], config['lr'],
                            epoch, config['epochs'], metrics['train_loss'], metrics['train_mixed_accuracy'],
                            f"{metrics['val_accuracy']:.2f}%" if 'val_accuracy' in metrics else 'sealed',
                            f"{metrics['ema_val_accuracy']:.2f}%" if 'ema_val_accuracy' in metrics else 'sealed',
                            f'{best_accuracy:.2f}%' if best_accuracy >= 0 else 'fixed final epoch', best_epoch,
                            metrics['lr'], duration(seconds), duration(elapsed_seconds/epoch*(config['epochs']-epoch)),
                            vram, metrics['amp_skips'], ' + best.pt' if best_saved else '')
                completed_this_call += 1
                if epoch_callback:
                    epoch_callback(epoch, best_accuracy)
                if REQUEST_STOP or (stop_after_epochs and completed_this_call >= stop_after_epochs):
                    if epoch < config['epochs']:
                        return {'status': 'interrupted', 'epoch': epoch, 'history': history}
                    break
            result = {'status': 'complete', 'experiment_id': directory.name, 'dataset': config['dataset'],
                      'model_name': config['model']['family'], 'architecture_version': 'compactmix-L-v2',
                      'seed': config['seed'], 'parameters': profile['parameters'], 'flops': profile['flops'],
                      'optimizer': config['optimizer'], 'learning_rate': config['lr'], 'weight_decay': config['weight_decay'],
                      'scheduler': config['scheduler'], 'augmentation': config['augmentation'], 'epochs': epoch,
                      'best_val_accuracy': best_accuracy if config['phase'] == 'selection' else None,
                      'best_epoch': best_epoch, 'best_kind': best_kind,
                      'final_val_accuracy': max(history[-1]['val_accuracy'], history[-1]['ema_val_accuracy']) if config['phase'] == 'selection' else None,
                      'test_accuracy': None, 'training_time': elapsed_seconds, 'checkpoint': str(directory / 'best.pt'),
                      'synthetic': split['synthetic'], 'subset': split['subset'], 'phase': config['phase'],
                      'history': history, 'config': config, 'profile': profile, 'metadata': metadata,
                      'notes': 'Synthetic/subset checks are not benchmark evidence' if split['synthetic'] or split['subset'] else ''}
            atomic_json(directory / 'result.json', result)
            return result
        finally:
            if writer:
                writer.close()
            close_logger(logger)
