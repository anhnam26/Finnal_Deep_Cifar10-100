"""GPU/CPU training smoke test with SYNTHETIC data; no CIFAR accuracy claims."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from data import make_loader
from models import LightNet
from train import build_optimizer, train_epoch
from utils import DEFAULT_CONFIG, ROOT, resolve_device, seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--classes", type=int, choices=[10, 100], default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "synthetic_benchmark.json"))
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batches <= 0:
        parser.error("Batch size and batches must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("Refusing to overwrite benchmark output")
    torch.set_num_threads(4)
    seed_everything(42)
    device = resolve_device(args.device)
    count = args.batch_size * args.batches
    source = TensorDataset(torch.randn(count, 3, 32, 32), torch.randint(args.classes, (count,)))
    loader = make_loader(source, args.batch_size, 0, use_cuda=device.type == "cuda")
    model = LightNet(args.classes).to(device)
    config = {
        **DEFAULT_CONFIG, "optimizer": args.optimizer,
        "lr": 0.1 if args.optimizer == "sgd" else 0.001,
        "weight_decay": 0.0005 if args.optimizer == "sgd" else 0.0001,
        "epochs": 2, "warmup_epochs": 0,
    }
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    results = [train_epoch(model, loader, optimizer, scaler, config, device, epoch) for epoch in range(2)]
    write_json(output, {
        "synthetic_only": True, "device": str(device), "optimizer": args.optimizer,
        "classes": args.classes, "batch_size": args.batch_size, "epochs": results,
        "warning": "Synthetic throughput only; no CIFAR data, augmentation, validation or checkpoint overhead. Not a reliable epoch-time prediction.",
    })
    print(output)
    for metrics in results:
        print(metrics)


if __name__ == "__main__":
    main()