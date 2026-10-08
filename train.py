"""Train on official training data only. This entry point never reads test data."""

import argparse
import csv
import json
import platform
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn

from data import build_train_loaders
from models import LightNet
from profile_model import profile_operations
from utils import (capture_rng, cutmix, learning_rate, load_config, resolve_device,
                   restore_rng, save_checkpoint, seed_everything, write_json, source_fingerprint)


def build_optimizer(model, config):
    # Weight decay applies to all learnable parameters, consistently in each run.
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["lr"],
                               momentum=config["momentum"], nesterov=config["nesterov"],
                               weight_decay=config["weight_decay"])
    return torch.optim.Adam(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                            eps=config["eps"], weight_decay=config["weight_decay"])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_loader(model, loader, device, max_batches=None, collect=False):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    predictions, labels = [], []
    for batch, (images, targets) in enumerate(loader):
        if max_batches is not None and batch >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        pred = logits.argmax(dim=1)
        loss_sum += loss.item()
        correct += (pred == targets).sum().item()
        count += len(targets)
        if collect:
            predictions.extend(pred.cpu().tolist())
            labels.extend(targets.cpu().tolist())
    if count == 0:
        raise ValueError("Empty evaluation loader")
    result = {"loss": loss_sum / count, "accuracy": 100 * correct / count, "samples": count}
    return (result, predictions, labels) if collect else result


def train_epoch(model, loader, optimizer, scaler, config, device, epoch):
    model.train()
    steps = min(len(loader), config["max_train_batches"] or len(loader))
    loss_sum, weighted_correct, count = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    first_lr, last_lr = None, None
    skipped_amp_steps = 0
    initial_amp_scale = scaler.get_scale()
    synchronize(device)
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step, (images, targets) in enumerate(loader):
        if step >= steps:
            break
        lr = learning_rate(epoch, step, steps, config["epochs"], config["warmup_epochs"],
                           config["lr"], config["min_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        first_lr = lr if first_lr is None else first_lr
        last_lr = lr
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        mixed = config["cutmix_probability"] > 0 and np.random.random() < config["cutmix_probability"]
        other_targets, lam = targets, 1.0
        if mixed:
            images, other_targets, lam = cutmix(images, targets, config["cutmix_alpha"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            loss = lam * criterion(logits, targets) + (1 - lam) * criterion(logits, other_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, batch {step}")
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        skipped_amp_steps += int(scaler.get_scale() < old_scale)
        pred = logits.detach().argmax(dim=1)
        weighted_correct += (lam * (pred == targets).sum().item()
                             + (1 - lam) * (pred == other_targets).sum().item())
        loss_sum += loss.item() * len(targets)
        count += len(targets)
    if not count:
        raise ValueError("Empty training loader")
    synchronize(device)
    elapsed = time.perf_counter() - start
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise FloatingPointError("Non-finite model parameters after training epoch")
    if skipped_amp_steps / steps > .01:
        raise FloatingPointError(f"AMP skipped {skipped_amp_steps}/{steps} updates (>1%); investigate instability")
    return {
        "train_loss": loss_sum / count, "train_weighted_accuracy": 100 * weighted_correct / count,
        "train_samples": count, "train_seconds": elapsed, "images_per_second": count / elapsed,
        "lr_first": first_lr, "lr_last": last_lr, "skipped_amp_steps": skipped_amp_steps,
        "amp_scale_initial": initial_amp_scale, "amp_scale_final": scaler.get_scale(),
        "amp_skipped_fraction": skipped_amp_steps/steps,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }


def run(config, resume=None):
    torch.set_num_threads(4)
    seed_everything(config["seed"], config["deterministic"])
    device = resolve_device(config["device"])
    output = Path(config["output_dir"])
    checkpoint = None
    if resume:
        # Only load trusted, locally produced checkpoints: they contain Python RNG state.
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        ignored = {"output_dir", "data_root", "download", "device", "workers"}
        differences = [key for key in config if key not in ignored and config[key] != checkpoint["config"][key]]
        if differences:
            raise ValueError(f"Resume changes training settings: {differences}")
        if Path(checkpoint["config"]["output_dir"]).resolve() != output.resolve():
            raise ValueError("Resume must use the original output directory")
        if not (output / "metrics.csv").exists():
            raise ValueError("Resume requires original metrics.csv")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use --resume or a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    if checkpoint and checkpoint.get("source_fingerprint", fingerprint) != fingerprint:
        raise ValueError("Source code changed since checkpoint; do not resume a different implementation")
    model = LightNet(10 if config["dataset"] == "cifar10" else 100, config["variant"]).to(device)
    profile = profile_operations(model)
    if not profile["within_limits"]:
        raise ValueError("Model exceeds final-exam limits")
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", init_scale=4096., enabled=config["amp"] and device.type == "cuda")
    train_loader, val_loader, split = build_train_loaders(config, device.type == "cuda")
    start_epoch, best_accuracy = 0, -1.0
    if checkpoint:
        if split["fingerprint"] != checkpoint["split_fingerprint"]:
            raise ValueError("Resume split mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_accuracy = checkpoint["epoch"] + 1, checkpoint["best_validation_accuracy"]
        train_loader.generator.set_state(checkpoint["train_loader_rng"])
        if val_loader is not None:
            val_loader.generator.set_state(checkpoint["val_loader_rng"])
        restore_rng(checkpoint["rng"])
        # Discard any orphan CSV rows after the checkpoint's completed epoch.
        with (output / "metrics.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row["epoch"]) <= start_epoch]
        if len(rows) != start_epoch:
            raise ValueError("Checkpoint and metrics history do not agree")
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        timing_path = output / "timing.csv"
        if timing_path.exists():
            with timing_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                timing_fields = reader.fieldnames
                timing_rows = [row for row in reader if int(row["epoch"]) <= start_epoch]
            with timing_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=timing_fields)
                writer.writeheader()
                writer.writerows(timing_rows)
    else:
        write_json(output / "config.json", config)
        write_json(output / "split.json", split)
        write_json(output / "profile.json", profile)
        write_json(output / "environment.json", {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "protocol": "full_train" if config["full_train"] else "train_validation",
            "debug_run": config["max_train_batches"] is not None or config["max_val_batches"] is not None,
            "scheduler": "per-step linear warmup then cosine; epoch-boundary resume",
            "accuracy_note": "Train accuracy is lambda-weighted when CutMix is used; validation is hard-label top-1.",
            "weight_decay_scope": "all learnable parameters",
            "amp_initial_scale": 4096.,
            "source_fingerprint": fingerprint,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        })
    if start_epoch >= config["epochs"]:
        raise ValueError("Checkpoint already completed all configured epochs")
    for epoch in range(start_epoch, config["epochs"]):
        write_json(output / "progress.json", {"phase": "training", "pid": os.getpid(),
                                              "completed_epochs": epoch, "total_epochs": config["epochs"]})
        epoch_start = time.perf_counter()
        metrics = {"epoch": epoch + 1, **train_epoch(model, train_loader, optimizer, scaler, config, device, epoch)}
        validation_start = time.perf_counter()
        validation = evaluate_loader(model, val_loader, device, config["max_val_batches"]) if val_loader else None
        synchronize(device)
        metrics["validation_seconds"] = time.perf_counter() - validation_start
        metrics["validation_loss"] = validation["loss"] if validation else None
        metrics["validation_accuracy"] = validation["accuracy"] if validation else None
        metrics["validation_samples"] = validation["samples"] if validation else 0
        is_best = validation is not None and validation["accuracy"] > best_accuracy
        if is_best:
            best_accuracy = validation["accuracy"]
        history = output / "metrics.csv"
        new_file = not history.exists()
        with history.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(metrics))
            if new_file:
                writer.writeheader()
            writer.writerow(metrics)
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "config": config, "epoch": epoch, "best_validation_accuracy": best_accuracy,
            "split_fingerprint": split["fingerprint"], "rng": capture_rng(),
            "train_loader_rng": train_loader.generator.get_state(),
            "val_loader_rng": val_loader.generator.get_state() if val_loader else None,
            "source_fingerprint": fingerprint,
        }
        checkpoint_start = time.perf_counter()
        save_checkpoint(output / "last.pt", state)
        if is_best:
            save_checkpoint(output / "best.pt", state)
        # Timing includes checkpoint writes; CSV above remains epoch-atomic on resume.
        timing = {"epoch": epoch+1, "checkpoint_seconds": time.perf_counter()-checkpoint_start,
                  "epoch_wall_seconds": time.perf_counter()-epoch_start}
        timing_path = output / "timing.csv"
        new_timing = not timing_path.exists()
        with timing_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(timing))
            if new_timing:
                writer.writeheader()
            writer.writerow(timing)
        write_json(output / "summary.json", {
            "completed_epochs": epoch + 1, "best_validation_accuracy": best_accuracy if val_loader else None,
            "final_metrics": metrics, "test_evaluated": False,
        })
        print(json.dumps(metrics), flush=True)
    write_json(output / "progress.json", {"phase":"completed", "completed_epochs": config["epochs"],
                                          "total_epochs": config["epochs"], "test_evaluated":False})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config, {"download": True} if args.download else None)
    run(config, args.resume)


if __name__ == "__main__":
    main()