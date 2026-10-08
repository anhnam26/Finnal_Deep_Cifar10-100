"""Prepare C100-first architecture screening or the exam LR matrix; never train."""

import argparse
from pathlib import Path

from utils import DEFAULT_CONFIG, ROOT, source_fingerprint, validate_config, write_json


def prepare(destination, output_root, mode="screen", matrix_variant="fusion"):
    destination, output_root = Path(destination).resolve(), Path(output_root).resolve()
    if mode not in ("screen", "matrix") or matrix_variant not in ("fusion", "no_fusion"):
        raise ValueError("Invalid study mode/variant")
    for location in (destination, output_root):
        if location.exists() and any(location.iterdir()):
            raise FileExistsError(f"Refusing to overwrite/reuse {location}")
    planned, smoke, grid = [], [], []
    base = {**DEFAULT_CONFIG, "workers": 0, "device": "cuda",
            "data_root": str(ROOT / "datasets")}

    def add(kind, dataset, variant, optimizer, lr, epochs):
        name = f"v4_{dataset}_{variant}_{optimizer}_{lr:g}_s42_e{epochs}"
        config = {**base, "dataset": dataset, "variant": variant, "optimizer": optimizer,
                  "lr": lr, "epochs": epochs, "warmup_epochs": 0 if kind == "smoke" else 5,
                  "weight_decay": 5e-4 if optimizer == "sgd" else 1e-4,
                  "output_dir": str(output_root / ("smoke" if kind == "smoke" else "runs") / name)}
        path = destination / kind / f"{name}.json"
        validate_config(config)
        planned.append((path, config))
        (smoke if kind == "smoke" else grid).append(str(path))

    variants = ("no_fusion", "fusion") if mode == "screen" else (matrix_variant,)
    for dataset in ("cifar100", "cifar10"):
        for variant in variants:
            add("smoke", dataset, variant, "sgd", .1, 2)
    if mode == "screen":
        for variant in variants:
            add("grid", "cifar100", variant, "sgd", .1, 150)
    else:
        for dataset in ("cifar100", "cifar10"):
            for optimizer, rates in (("sgd", (.05, .1, .15)), ("adam", (.0003, .001, .003))):
                for lr in rates:
                    add("grid", dataset, matrix_variant, optimizer, lr, 150)
    for path, config in planned:
        write_json(path, config)
    manifest = {
        "output_root": str(output_root), "source_root": str(ROOT),
        "required_source_fingerprint": source_fingerprint(), "smoke": smoke, "grid": grid,
        "grid_epochs": sum(config["epochs"] for path, config in planned if str(path) in grid),
        "test_evaluation": False, "max_grid_hours": 24 if mode == "screen" else 72,
        "budget_reserve_factor": 1.25, "mode": mode,
        "worker_selection": "fixed workers0; full45k/5k smoke on both datasets",
        "grid_protocol": "150epochs seed42 split2026 crop/flip; identical screening recipe, only fusion differs",
        "selection_rule": "Validation-only screening; regenerate matrix for selected architecture/recipe before launch",
        "target": "C10~98%/C100~95% validation are aspirational, not measured or guaranteed",
    }
    write_json(destination / "manifest.json", manifest)
    return destination / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["screen", "matrix"], default="screen")
    parser.add_argument("--variant", choices=["fusion", "no_fusion"], default="fusion")
    parser.add_argument("--destination")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    directory = "candidate" if args.mode == "screen" else "candidate_matrix"
    print(prepare(args.destination or ROOT / "configs" / directory,
                  args.output_root or ROOT / "outputs" / directory, args.mode, args.variant))


if __name__ == "__main__":
    main()