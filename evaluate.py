"""Explicitly gated official test evaluation. Never use these results for tuning."""

import argparse
from pathlib import Path

import torch

from data import DATASETS, build_test_loader
from models import LightNet
from train import evaluate_loader
from utils import resolve_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted local checkpoint only")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--confirm-final-test", action="store_true", help="All selection decisions are already frozen")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.confirm_final_test:
        parser.error("Official test requires --confirm-final-test; use validation during development")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Refusing to overwrite existing test results")
    torch.set_num_threads(4)
    checkpoint_path = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    if config["max_train_batches"] is not None or config["max_val_batches"] is not None:
        parser.error("Debug/partial-batch checkpoints are not eligible for official test evaluation")
    if config["full_train"] and state["epoch"] + 1 != config["epochs"]:
        parser.error("Full-data training must finish its frozen epoch budget before test evaluation")
    device = resolve_device(args.device)
    _, classes = DATASETS[config["dataset"]]
    model = LightNet(classes, config["variant"]).to(device)
    model.load_state_dict(state["model"])
    loader = build_test_loader(config, device.type == "cuda")
    metrics, predictions, labels = evaluate_loader(model, loader, device, collect=True)
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    for true, pred in zip(labels, predictions):
        matrix[true, pred] += 1
    result = {
        "checkpoint": str(checkpoint_path), "dataset": config["dataset"],
        "epoch": state["epoch"] + 1, "metrics": metrics,
        "protocol": "single model, float32, single view, no TTA",
        "selection_protocol": "full_train_final_epoch" if config["full_train"] else "frozen_train_validation_checkpoint",
        "confusion_matrix": matrix.tolist(),
        "per_class_accuracy": (matrix.diag().float() / matrix.sum(dim=1).clamp_min(1) * 100).tolist(),
    }
    write_json(output, result)
    print(metrics)


if __name__ == "__main__":
    main()