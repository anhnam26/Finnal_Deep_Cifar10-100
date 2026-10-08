"""Export full validation metrics; official test is explicitly gated."""

import argparse
import json
import time
import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from data import DATASETS, build_transforms, make_loader, split_fingerprint, stratified_split
from metrics import classification_metrics
from models import LightNet
from profile_model import benchmark, profile_operations
from reporting import export_evaluation
from utils import resolve_device, write_json


@torch.inference_mode()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for images, targets in loader:
        result = model(images.to(device, non_blocking=True)).float()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite evaluation logits")
        logits.append(result.cpu().numpy())
        labels.append(targets.numpy())
    if not logits:
        raise ValueError("Empty loader")
    return np.concatenate(logits), np.concatenate(labels)


def evaluate_checkpoint(checkpoint, output_dir, split="validation", device_name="auto",
                        confirm_final_test=False, measure_cost=True):
    if split not in ("validation", "test", "train"):
        raise ValueError("Invalid evaluation split")
    if split == "test" and not confirm_final_test:
        raise ValueError("Official test requires --confirm-final-test; freeze decisions first")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation exists: {output}")
    torch.set_num_threads(4)
    checkpoint_path = Path(checkpoint).resolve()
    # Trusted local checkpoints only: contain Python RNG objects.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    if config["max_train_batches"] is not None or config["max_val_batches"] is not None:
        raise ValueError("Partial-batch debug checkpoints are not eligible for full evaluation")
    if split == "validation" and config["full_train"]:
        raise ValueError("Full-data model has trained on validation indices; no held-out validation available")
    if split == "test" and config["full_train"] and state["epoch"] + 1 != config["epochs"]:
        raise ValueError("Full-data training must finish its frozen epoch budget")
    cls, classes = DATASETS[config["dataset"]]
    _, transform = build_transforms(config["dataset"], False)
    source = cls(root=config["data_root"], train=split != "test", download=False, transform=transform)
    if split == "test":
        indices = list(range(len(source)))
    else:
        if config["full_train"]:
            training, validation = list(range(len(source))), []
        else:
            training, validation = stratified_split(source.targets, classes, 500 if classes == 10 else 50,
                                                   config["split_seed"])
        if split_fingerprint(training, validation) != state["split_fingerprint"]:
            raise ValueError("Checkpoint split fingerprint does not match dataset")
        indices = validation if split == "validation" else training
    device = resolve_device(device_name)
    loader = make_loader(Subset(source, indices), config["batch_size"], config["workers"], use_cuda=device.type == "cuda")
    model = LightNet(classes, config["variant"]).to(device)
    model.load_state_dict(state["model"])
    del state["optimizer"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    logits, labels = collect_logits(model, loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - start
    names = source.classes
    metrics, probabilities = classification_metrics(logits, labels, names)
    profile = profile_operations(model)
    if measure_cost:
        profile["latency_batch1"] = benchmark(model, 1, steps=50, warmup=10)
        profile["throughput_batch128"] = benchmark(model, 128, steps=30, warmup=10)
    metadata = {
        "checkpoint": str(checkpoint_path), "dataset": config["dataset"], "variant": config["variant"],
        "split": split, "split_fingerprint": state["split_fingerprint"],
        "epoch": state["epoch"] + 1, "configured_epochs": config["epochs"],
        "run_completed": (checkpoint_path.parent / "summary.json").exists() and
                         json.loads((checkpoint_path.parent / "summary.json").read_text())["completed_epochs"] == config["epochs"],
        "optimizer": config["optimizer"], "lr": config["lr"], "seed": config["seed"],
        "full_train": config["full_train"], "profile": profile,
        "evaluation_seconds": evaluation_seconds, "evaluation_images_per_second": len(labels)/evaluation_seconds,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "source_fingerprint": state.get("source_fingerprint"),
        "config": config, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "protocol": "float32, single model/view, no TTA, hard-label metrics; latency excludes preprocessing/I/O",
    }
    history_path = checkpoint_path.parent / "metrics.csv"
    timing_path = checkpoint_path.parent / "timing.csv"
    if history_path.exists():
        import pandas as pd
        history = pd.read_csv(history_path)
        metadata["training_efficiency"] = {
            "logged_epochs": len(history), "train_seconds_total": float(history.train_seconds.sum()),
            "validation_seconds_total": float(history.validation_seconds.sum()) if "validation_seconds" in history else None,
            "training_images_per_second_mean": float(history.images_per_second.mean()),
            "training_images_per_second_std": float(history.images_per_second.std(ddof=1)) if len(history)>1 else None,
            "train_seconds_epoch_mean": float(history.train_seconds.mean()),
            "train_seconds_epoch_std": float(history.train_seconds.std(ddof=1)) if len(history)>1 else None,
            "skipped_amp_steps_total": int(history.skipped_amp_steps.sum()),
            "peak_training_allocated_bytes": int(history.peak_allocated_bytes.max()) if history.peak_allocated_bytes.notna().any() else None,
            "peak_training_reserved_bytes": int(history.peak_reserved_bytes.max()) if "peak_reserved_bytes" in history and history.peak_reserved_bytes.notna().any() else None,
        }
        if timing_path.exists():
            timing = pd.read_csv(timing_path)
            metadata["training_efficiency"].update({"epoch_wall_seconds_total":float(timing.epoch_wall_seconds.sum()),
                                                   "epoch_wall_seconds_mean":float(timing.epoch_wall_seconds.mean()),
                                                   "checkpoint_seconds_total":float(timing.checkpoint_seconds.sum())})
    weights = io.BytesIO()
    torch.save({key:value.cpu() for key,value in model.state_dict().items()}, weights)
    metadata["model_weights_bytes"] = weights.getbuffer().nbytes
    export_evaluation(output, metrics, logits, labels, np.asarray(indices), probabilities, names, metadata,
                      history_path, source.data)
    (output / "model_weights.pt").write_bytes(weights.getvalue())
    # File size is model-only weights, not optimizer state. Publish completion last.
    write_json(output / "complete.json", {"complete": True, "split": split, "samples": len(labels)})
    print(f"Exported {split}: {metrics['summary']}", flush=True)
    return metrics["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["validation","test","train"], default="validation")
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--no-cost-benchmark", action="store_true")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.output_dir, args.split, args.device,
                        args.confirm_final_test, not args.no_cost_benchmark)


if __name__ == "__main__":
    main()