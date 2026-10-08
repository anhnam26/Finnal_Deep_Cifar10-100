"""Create validation-only 300-epoch architecture/recipe ablations. Never train/test."""

import argparse
from pathlib import Path

from utils import DEFAULT_CONFIG, ROOT, validate_config, write_json


def prepare(destination, output_root):
    destination, output_root = Path(destination).resolve(), Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {destination}")
    planned, smoke, grid = [], [], []
    for variant in ("baseline", "hybrid"):
        base = {**DEFAULT_CONFIG, "dataset":"cifar10", "variant":variant,
                "optimizer":"sgd", "lr":.1, "weight_decay":5e-4,
                "workers":0, "device":"cuda", "label_smoothing":0.0}
        name = f"smoke_v2_cifar10_{variant}_cutmix_s42"
        config = {**base, "epochs":2, "warmup_epochs":0,
                  "cutmix_alpha":1.0, "cutmix_probability":.5,
                  "output_dir":str(output_root/"smoke"/name)}
        path = destination/"smoke"/f"{name}.json"
        smoke.append(str(path))
        planned.append((path, config))
        for recipe in ("plain", "cutmix"):
            name = f"v2_cifar10_{variant}_{recipe}_sgd_0.1_s42_e300"
            config = {**base, "epochs":300, "warmup_epochs":5,
                      "cutmix_alpha":1.0 if recipe == "cutmix" else 0.0,
                      "cutmix_probability":.5 if recipe == "cutmix" else 0.0,
                      "output_dir":str(output_root/"runs"/name)}
            path = destination/"grid"/f"{name}.json"
            grid.append(str(path))
            planned.append((path, config))
    for path, config in planned:
        validate_config(config)
    for path, config in planned:
        write_json(path, config)
    manifest = {"output_root":str(output_root), "smoke":smoke, "grid":grid,
                "grid_epochs":1200, "test_evaluation":False,
                "max_grid_hours":72, "budget_reserve_factor":1.25,
                "grid_protocol":"300epochs seed42 stratified45k/5k; validation only; architecture x CutMix ablation",
                "worker_selection":"workers0; measured on new full-epoch CIFAR10 smoke",
                "target":"official-test top1 >97%, aspirational, not guaranteed; freeze before final50k/test"}
    write_json(destination/"manifest.json", manifest)
    return destination/"manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default=str(ROOT/"configs/improvement"))
    parser.add_argument("--output-root", default=str(ROOT/"outputs/study_improvement"))
    args = parser.parse_args()
    print(prepare(args.destination, args.output_root))


if __name__ == "__main__":
    main()