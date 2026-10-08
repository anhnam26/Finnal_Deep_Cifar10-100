"""Create variant-aware configs in a new location. Does not train."""

import argparse
from pathlib import Path

from utils import DEFAULT_CONFIG, ROOT, validate_config, write_json


def prepare(destination, output_root):
    destination, output_root = Path(destination).resolve(), Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {destination}")
    planned, grid, smoke = [], [], []
    for dataset in ("cifar10", "cifar100"):
        for variant in ("baseline", "multiscale"):
            name = f"smoke_{dataset}_{variant}_w0"
            config = {**DEFAULT_CONFIG, "dataset":dataset, "variant":variant, "epochs":2,
                      "warmup_epochs":0, "workers":0, "device":"cuda",
                      "optimizer":"sgd" if dataset == "cifar10" else "adam",
                      "lr":.1 if dataset == "cifar10" else .001,
                      "weight_decay":5e-4 if dataset == "cifar10" else 1e-4,
                      "output_dir":str(output_root / "smoke" / name)}
            path = destination / "smoke" / f"{name}.json"
            smoke.append(str(path))
            planned.append((path,config))
    for variant in ("baseline","multiscale"):
        name = f"workers_cifar100_{variant}_w2"
        config = {**DEFAULT_CONFIG, "dataset":"cifar100", "variant":variant, "epochs":2,
                  "warmup_epochs":0, "workers":2, "device":"cuda",
                  "optimizer":"adam", "lr":.001, "weight_decay":1e-4,
                  "output_dir":str(output_root / "smoke" / name)}
        path = destination / "smoke" / f"{name}.json"
        smoke.append(str(path))
        planned.append((path,config))
    # Interleave variants within each optimizer/LR to reduce temporal order bias.
    for dataset in ("cifar10", "cifar100"):
        for optimizer, rates in (("sgd",(.05,.1,.15)),("adam",(.0003,.001,.003))):
            for lr in rates:
                for variant in ("baseline","multiscale"):
                    name = f"{dataset}_{variant}_{optimizer}_{lr:g}_s42"
                    config = {**DEFAULT_CONFIG, "dataset":dataset, "variant":variant,
                              "optimizer":optimizer, "lr":lr,
                              "weight_decay":5e-4 if optimizer == "sgd" else 1e-4,
                              "device":"cuda", "output_dir":str(output_root / "runs" / name)}
                    path = destination / "grid" / f"{name}.json"
                    grid.append(str(path))
                    planned.append((path,config))
    for path,config in planned:
        validate_config(config)
    for path,config in planned:
        write_json(path,config)
    manifest = {"output_root":str(output_root), "smoke":smoke, "grid":grid,
                "grid_epochs":3600, "test_evaluation":False,
                "worker_selection":"median full-epoch wall time w0/w2 on CIFAR100, same workers for all grid runs",
                "grid_protocol":"150epochs seed42, fixed stratified45k/5k, no test tuning",
                "max_grid_hours":168, "budget_reserve_factor":1.25}
    write_json(destination / "manifest.json",manifest)
    return destination / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",default=str(ROOT / "configs" / "study"))
    parser.add_argument("--output-root",default=str(ROOT / "outputs" / "study_two_models"))
    args = parser.parse_args()
    print(prepare(args.destination,args.output_root))


if __name__ == "__main__":
    main()