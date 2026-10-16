"""Load best weights and evaluate the complete CIFAR-10/CIFAR-100 test sets.

    python report/evaluate_best.py
    python report/evaluate_best.py --dataset cifar10 --checkpoint path/to/best.pt

Defaults to the two previously evaluated, archived best checkpoints.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from src.python_policy import ensure_selected_python
ensure_selected_python()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torchvision import datasets
from src import engine
from src.datasets import make_loader
from src.models import build_model


def load_model(checkpoint, dataset):
    """Reject a wrong dataset or incompatible architecture before opening test."""
    config = dict(checkpoint["config"])
    if config["dataset"] != dataset:
        raise ValueError(f"Checkpoint dataset={config['dataset']}, requested {dataset}")
    if config.get("smoke") or config.get("limit_train") or config.get("limit_val"):
        raise ValueError("Smoke/subset checkpoints are not final test benchmarks")
    classes = 10 if dataset == "cifar10" else 100
    if config["model"]["family"] == "compactmix_manual":
        spec = importlib.util.spec_from_file_location("test_manual_model", HERE / "compactmix_manual_train.py")
        manual = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manual)
        model = manual.make_model(classes)
    else:
        model = build_model(config["model"], classes)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, config


def evaluate_checkpoint(dataset, path, data_root, device, batch_size):
    path = path.resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = engine.load_checkpoint(path)
    model, config = load_model(checkpoint, dataset)
    config.update(data_root=str(data_root.resolve()), device=device, batch_size=batch_size, workers=0)
    engine.seed_everything(config["seed"], config["deterministic"])
    model = model.to(device)
    constructor = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    source = constructor(root=config["data_root"], train=False, download=False)
    if len(source) != 10_000:
        raise ValueError(f"Expected the complete 10,000-image test set, got {len(source)}")
    loader = make_loader(source, list(range(len(source))), config, 0, training=False)
    # evaluate() sets model.eval() and disables gradients. No crop/flip/AutoAugment.
    metrics = engine.evaluate(model, loader, device, extended=True)
    confusion = metrics["confusion_matrix"]
    correct = sum(confusion[i][i] for i in range(len(confusion)))
    return {"dataset": dataset, "checkpoint": str(path), "checkpoint_sha256": digest,
            "epoch": checkpoint["epoch"], "weights": checkpoint.get("kind", "model"),
            "phase": config.get("phase", "unknown"), "samples": len(source),
            "device": device, "batch_size": batch_size,
            "python": sys.version.split()[0], "pytorch": torch.__version__,
            "correct": correct, "accuracy_pct": metrics["accuracy"],
            "top5_pct": metrics["top5_accuracy"], "loss": metrics["loss"],
            "macro_f1": metrics["macro_f1"], "metrics": metrics,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["both", "cifar10", "cifar100"], default="both")
    parser.add_argument("--checkpoint", type=Path, help="Use with a single --dataset")
    parser.add_argument("--checkpoint-cifar10", type=Path, default=HERE / "checkpoints/cifar10/best.pt")
    parser.add_argument("--checkpoint-cifar100", type=Path, default=HERE / "checkpoints/cifar100/best.pt")
    parser.add_argument("--data-root", type=Path, default=HERE.parent / "datasets")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "experiments/test_best")
    args = parser.parse_args(argv)
    if args.checkpoint and args.dataset == "both":
        parser.error("--checkpoint requires --dataset cifar10 or cifar100")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    torch.set_num_threads(2)
    names = ("cifar10", "cifar100") if args.dataset == "both" else (args.dataset,)
    rows = []
    for name in names:
        path = args.checkpoint or getattr(args, "checkpoint_" + name)
        result = evaluate_checkpoint(name, path, args.data_root, device, args.batch_size)
        engine.atomic_json(args.output_dir / (name + "_test.json"), result)
        rows.append({k: result[k] for k in ("dataset", "epoch", "weights", "samples", "correct",
                                           "accuracy_pct", "top5_pct", "loss", "macro_f1", "checkpoint")})
        def write(stream):
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        engine.atomic_write(args.output_dir / "test_summary.csv", write)
        print(f"{name.upper()} | epoch={result['epoch']} weights={result['weights']} | "
              f"TEST accuracy={result['accuracy_pct']:.2f}% ({result['correct']}/{result['samples']}) | "
              f"top5={result['top5_pct']:.2f}% loss={result['loss']:.6f} macro-F1={result['macro_f1']:.6f}\n"
              f"  checkpoint: {result['checkpoint']}", flush=True)
    print(f"Results: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
