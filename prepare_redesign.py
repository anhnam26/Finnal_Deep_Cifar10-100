"""Prepare a matched 150-epoch backbone comparison; never train or read test data."""

import argparse
from pathlib import Path

from utils import DEFAULT_CONFIG, ROOT, source_fingerprint, validate_config, write_json


def prepare(destination, output_root):
    destination, output_root = Path(destination).resolve(), Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse outputs {output_root}")
    planned, smoke, grid = [], [], []
    for variant in ("baseline", "preact"):
        base = {**DEFAULT_CONFIG, "dataset": "cifar10", "variant": variant,
                "optimizer": "sgd", "lr": .1, "weight_decay": 5e-4,
                "workers": 0, "device": "cuda", "label_smoothing": 0.,
                "cutmix_alpha": 0., "cutmix_probability": 0.,
                "data_root": str(ROOT / "datasets")}
        for kind, epochs, warmup, collection in (
                ("smoke", 2, 0, smoke), ("grid", 150, 5, grid)):
            prefix = "smoke_v3" if kind == "smoke" else "v3"
            name = f"{prefix}_cifar10_{variant}_plain_sgd_0.1_s42_e{epochs}"
            config = {**base, "epochs": epochs, "warmup_epochs": warmup,
                      "output_dir": str(output_root / ("smoke" if kind == "smoke" else "runs") / name)}
            path = destination / kind / f"{name}.json"
            validate_config(config)
            planned.append((path, config))
            collection.append(str(path))
    for path, config in planned:
        write_json(path, config)
    manifest = {"output_root": str(output_root), "smoke": smoke, "grid": grid,
                "source_root": str(ROOT), "required_source_fingerprint": source_fingerprint(),
                "grid_epochs": 300, "test_evaluation": False, "max_grid_hours": 24,
                "budget_reserve_factor": 1.25,
                "grid_protocol": "Matched 150epochs SGD0.1 plain crop/flip, seed42 split2026 45k/5k; only backbone differs; compare time as well as accuracy",
                "worker_selection": "workers0; measured full-epoch smoke for both backbones",
                "selection_rule": "Screen on validation only; no automatic winner, extra epochs, augmentations or official-test evaluation",
                "target": "official-test top1 >97% is aspirational, not a measured or promised result"}
    write_json(destination / "manifest.json", manifest)
    return destination / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default=str(ROOT / "configs/redesign"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs/study_redesign"))
    args = parser.parse_args()
    print(prepare(args.destination, args.output_root))


if __name__ == "__main__":
    main()