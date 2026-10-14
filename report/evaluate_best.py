"""Re-evaluate an archived best using the user's selected Python and local CIFAR data."""
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-root", type=Path, default=HERE.parent / "datasets")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    checkpoint = engine.load_checkpoint(HERE / "checkpoints" / args.dataset / "best.pt")
    config = dict(checkpoint["config"])
    config.update(data_root=str(args.data_root.resolve()), device=args.device, batch_size=64, workers=0)
    engine.seed_everything(config["seed"], config["deterministic"])
    torch.set_num_threads(2)
    classes = 10 if args.dataset == "cifar10" else 100
    model = build_model(config["model"], classes)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(args.device)
    constructor = datasets.CIFAR10 if classes == 10 else datasets.CIFAR100
    source = constructor(root=config["data_root"], train=False, download=False)
    loader = make_loader(source, list(range(len(source))), config, 0, training=False)
    metrics = engine.evaluate(model, loader, args.device, extended=True)
    print(json.dumps({"dataset": args.dataset, "epoch": checkpoint["epoch"],
                      "samples": len(source), "metrics": metrics}, indent=2))

if __name__ == "__main__":
    main()
