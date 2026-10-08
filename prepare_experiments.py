"""Write the 12 optimizer/LR configs. Does NOT start any training jobs."""

import argparse
from pathlib import Path

from utils import ROOT, DEFAULT_CONFIG, validate_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--destination", default=str(ROOT / "configs" / "grid"))
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    planned = []
    for dataset in ("cifar10", "cifar100"):
        for optimizer, lrs in (("sgd", (0.05, 0.1, 0.15)), ("adam", (0.0003, 0.001, 0.003))):
            for lr in lrs:
                name = f"{dataset}_{optimizer}_{lr:g}"
                config = {
                    **DEFAULT_CONFIG, "dataset": dataset, "optimizer": optimizer, "lr": lr,
                    "weight_decay": 5e-4 if optimizer == "sgd" else 1e-4,
                    "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
                    "output_dir": f"outputs/grid/{name}",
                }
                validate_config(config)
                path = destination / f"{name}.json"
                if path.exists():
                    raise FileExistsError(f"Refusing to overwrite: {path}")
                planned.append((path, config))
    for path, config in planned:
        write_json(path, config)
        print(path)


if __name__ == "__main__":
    main()