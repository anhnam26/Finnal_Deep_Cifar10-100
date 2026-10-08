"""Training utilities with epoch-boundary reproducible resume."""

import json
import math
import os
import random
import hashlib
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def source_fingerprint():
    """Fingerprint executable project source, not datasets or internal logs."""
    files = sorted(list(ROOT.glob("*.py")) + list((ROOT / "models").glob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def learning_rate(epoch, step, steps_per_epoch, epochs, warmup_epochs, base_lr, min_lr):
    """Step-based warmup followed by cosine, reaching min_lr on the last step."""
    if not (epochs > 0 and steps_per_epoch > 0 and 0 <= warmup_epochs < epochs):
        raise ValueError("Invalid schedule lengths")
    if not (0 <= min_lr <= base_lr and base_lr > 0):
        raise ValueError("Invalid learning rates")
    current = epoch * steps_per_epoch + step
    total = epochs * steps_per_epoch
    warmup = warmup_epochs * steps_per_epoch
    if current < warmup:
        return base_lr * (current + 1) / warmup
    progress = (current - warmup) / max(1, total - warmup - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def cutmix(images, targets, alpha):
    if alpha <= 0 or images.ndim != 4 or images.shape[0] != targets.shape[0]:
        raise ValueError("CutMix needs alpha > 0 and matching NCHW images/labels")
    permutation = torch.randperm(images.shape[0], device=images.device)
    lam = float(np.random.beta(alpha, alpha))
    height, width = images.shape[-2:]
    ratio = math.sqrt(1 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    center_y, center_x = np.random.randint(height), np.random.randint(width)
    y1, y2 = max(0, center_y - cut_h // 2), min(height, center_y + cut_h // 2)
    x1, x2 = max(0, center_x - cut_w // 2), min(width, center_x + cut_w // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    actual_lambda = 1 - ((y2 - y1) * (x2 - x1) / (height * width))
    return mixed, targets[permutation], actual_lambda


DEFAULT_CONFIG = {
    "dataset": "cifar10", "variant": "fusion", "optimizer": "sgd", "lr": 0.1,
    "epochs": 150, "warmup_epochs": 5, "min_lr": 0.00001,
    "batch_size": 128, "workers": 2, "seed": 42, "split_seed": 2026,
    "momentum": 0.9, "nesterov": True, "weight_decay": 0.0005,
    "betas": [0.9, 0.999], "eps": 1e-8, "label_smoothing": 0.0,
    "cutmix_alpha": 0.0, "cutmix_probability": 0.0,
    "augment": True, "amp": True, "deterministic": False,
    "download": False, "full_train": False,
    "data_root": "datasets", "output_dir": "outputs/cifar10_sgd_0.1",
    "device": "auto", "max_train_batches": None, "max_val_batches": None,
}


def load_config(path, overrides=None):
    supplied = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **supplied, **(overrides or {})}
    for key in ("data_root", "output_dir"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else ROOT / value)
    validate_config(config)
    return config


def validate_config(config):
    if config["dataset"] not in ("cifar10", "cifar100"):
        raise ValueError("Unknown dataset")
    if config["variant"] not in ("fusion", "no_fusion"):
        raise ValueError("Unknown variant")
    if config["optimizer"] not in ("sgd", "adam"):
        raise ValueError("Only SGD/Adam supported")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid device")
    for key in ("epochs", "batch_size"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["workers"], int) or config["workers"] < 0 or config["weight_decay"] < 0:
        raise ValueError("workers and weight_decay must be nonnegative")
    for key in ("seed", "split_seed"):
        if not isinstance(config[key], int) or not 0 <= config[key] < 2 ** 32:
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    if not isinstance(config["warmup_epochs"], int):
        raise ValueError("warmup_epochs must be an integer")
    if not 0 <= config["warmup_epochs"] < config["epochs"]:
        raise ValueError("warmup_epochs must be >= 0 and less than epochs")
    if not 0 <= config["min_lr"] <= config["lr"] or config["lr"] <= 0:
        raise ValueError("Invalid learning rate")
    if not 0 <= config["momentum"] < 1:
        raise ValueError("Invalid momentum")
    if config["optimizer"] == "sgd" and config["nesterov"] and config["momentum"] <= 0:
        raise ValueError("Nesterov requires positive momentum")
    if len(config["betas"]) != 2 or any(not 0 <= b < 1 for b in config["betas"]) or config["eps"] <= 0:
        raise ValueError("Invalid Adam settings")
    if not 0 <= config["label_smoothing"] < 1:
        raise ValueError("Invalid label smoothing")
    if not 0 <= config["cutmix_probability"] <= 1 or config["cutmix_alpha"] < 0:
        raise ValueError("Invalid CutMix settings")
    if config["cutmix_probability"] > 0 and config["cutmix_alpha"] <= 0:
        raise ValueError("CutMix probability > 0 requires positive alpha")
    for key in ("max_train_batches", "max_val_batches"):
        if config[key] is not None and (not isinstance(config[key], int) or config[key] <= 0):
            raise ValueError(f"{key} must be null or a positive integer")
    if config["full_train"] and any(config[k] is not None for k in ("max_train_batches", "max_val_batches")):
        raise ValueError("Full-data final training cannot use debug batch limits")


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)