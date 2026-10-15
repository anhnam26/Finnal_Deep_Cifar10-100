"""Small, resumable CIFAR-10 validation comparisons; never opens official test."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import time


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "report/compactmix_manual_train.py"
spec = importlib.util.spec_from_file_location("quick_comparison_engine", ENGINE_PATH)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)  # Enforces the selected Store Python installation.
np, torch = engine.np, engine.torch

VARIANTS = (
    ("baseline", {}),
    ("momentum_080", {"momentum": 0.8}),
    ("lr_005", {"lr": 0.05}),
    ("scheduler_constant", {"scheduler": "constant"}),
    ("momentum_095", {"momentum": 0.95}),
    ("lr_015", {"lr": 0.15}),
    ("scheduler_step", {"scheduler": "step"}),
)
ORIGINAL_LR = engine.learning_rate


def learning_rate(config, epoch, batch, batches):
    if config["scheduler"] == "cosine":
        return ORIGINAL_LR(config, epoch, batch, batches)
    progress = epoch + batch / batches
    warmup = min(config["warmup"], config["epochs"] / 10)
    if warmup and progress < warmup:
        return config["lr"] * (0.1 + 0.9 * progress / warmup)
    if config["scheduler"] == "constant":
        return config["lr"]
    if config["scheduler"] == "step":
        midpoint = warmup + (config["epochs"] - warmup) / 2
        return config["lr"] * (1 if progress < midpoint else 0.1)
    raise ValueError("Unsupported quick-comparison scheduler")


def balanced_subset(candidates, targets, total, seed):
    if total < 10 or total % 10:
        raise ValueError("Subset size must be a positive multiple of 10")
    rng = np.random.default_rng(seed)
    candidates = np.asarray(candidates)
    targets = np.asarray(targets)
    chosen = []
    for label in range(10):
        pool = candidates[targets[candidates] == label]
        if len(pool) < total // 10:
            raise ValueError("Not enough images in the requested split")
        chosen.extend(rng.choice(pool, total // 10, replace=False).tolist())
    return sorted(chosen)


def install_comparison_hooks():
    engine.learning_rate = learning_rate
    fingerprint = hashlib.sha256(
        Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        + ENGINE_PATH.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    engine.source_fingerprint = lambda: fingerprint
    cache = {}

    def prepare_data(config):
        key = (config["data_root"], config["split_seed"],
               config["limit_train"], config["limit_val"])
        if key not in cache:
            source = engine.datasets.CIFAR10(config["data_root"], train=True, download=False)
            train, val = engine.stratified_split(source.targets, 10, config["split_seed"])
            train = balanced_subset(train, source.targets, config["limit_train"], config["split_seed"] + 1)
            val = balanced_subset(val, source.targets, config["limit_val"], config["split_seed"] + 2)
            assert not set(train) & set(val)
            payload = {"training": train, "validation": val}
            split = {**payload, "classes": 10, "synthetic": False, "subset": True,
                     "sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
            cache[key] = source, split
        return cache[key]

    engine.prepare_data = prepare_data
    return fingerprint


def config_for(args, name, changes):
    config = copy.deepcopy(engine.TRAIN_SETTINGS)
    config.update(dataset="cifar10", model={"family": "compactmix_manual"},
                  phase="selection", final_weights="ema", device="cuda",
                  data_root=str(ROOT / "datasets"), download=False, smoke=False,
                  limit_train=args.train_images, limit_val=args.val_images,
                  epochs=args.epochs, workers=0, seed=42, split_seed=2026,
                  run_dir=str(args.output.resolve() / name))
    config.update(changes)
    return config


def export_summary(directory):
    results = []
    for name, _ in VARIANTS:
        path = directory / name / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    records = []
    for result in results:
        c = result["config"]
        history = result["history"]
        records.append({
            "variant": result["experiment_id"], "momentum": c["momentum"],
            "base_lr": c["lr"], "scheduler": c["scheduler"], "epochs": c["epochs"],
            "train_images": c["limit_train"], "validation_images": c["limit_val"],
            "best_val_pct": result["best_val_accuracy"], "best_epoch": result["best_epoch"],
            "best_weights": result["best_kind"],
            "final_raw_val_pct": history[-1]["val_accuracy"],
            "final_ema_val_pct": history[-1]["ema_val_accuracy"],
            "final_raw_val_loss": history[-1]["val_loss"],
            "amp_skips": sum(row["amp_skips"] for row in history),
            "engine_seconds": result["training_time"], "test_accuracy": None,
            "subset": True, "synthetic": False,
        })
    engine.atomic_json(directory / "comparison.json", records)
    if records:
        def write(stream):
            writer = engine.csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        engine.atomic_write(directory / "comparison.csv", write)
    return records


def check(args):
    c = config_for(args, "baseline", {})
    for step in range(args.epochs * 10):
        assert learning_rate(c, step // 10, step % 10, 10) == ORIGINAL_LR(c, step // 10, step % 10, 10)
    for scheduler in ("cosine", "constant", "step"):
        candidate = {**c, "scheduler": scheduler}
        rates = [learning_rate(candidate, step // 10, step % 10, 10)
                 for step in range(args.epochs * 10)]
        assert all(math.isfinite(rate) and 0 < rate <= c["lr"] for rate in rates)
    for name, changes in VARIANTS:
        candidate = config_for(args, name, changes)
        different = {k for k in c if c[k] != candidate[k]} - {"run_dir"}
        assert different == set(changes)
    assert balanced_subset(range(100), np.repeat(np.arange(10), 10), 20, 9) == balanced_subset(
        range(100), np.repeat(np.arange(10), 10), 20, 9)
    print("OK: one-variable comparisons, deterministic balanced subsets, scheduler formulas")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/quick_recipe_20261016")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-images", type=int, default=5000)
    parser.add_argument("--val-images", type=int, default=1000)
    parser.add_argument("--budget-seconds", type=float, default=780)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This timed comparison requires CUDA in the selected Python")
    started = time.monotonic()
    deadline = started + args.budget_seconds
    fingerprint = install_comparison_hooks()
    args.output.mkdir(parents=True, exist_ok=True)
    c = config_for(args, "baseline", {})
    source, split = engine.prepare_data(c)
    engine.seed_everything(c["seed"], c["deterministic"])
    model = engine.make_model(10)
    initial = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial.update(name.encode())
        initial.update(value.cpu().numpy().tobytes())
    profile = engine.inspect_model(model, 10, backward=True)
    del model
    receipt = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": fingerprint, "initial_weights_sha256": initial.hexdigest(),
               "environment": engine.environment(c), "python_executable": engine.sys.executable,
               "torch_location": torch.__file__, "numpy_location": np.__file__,
               "base_environment": engine.sys.prefix == engine.sys.base_prefix,
               "profile": {k: profile[k] for k in ("parameters", "macs", "flops")},
               "split_sha256": split["sha256"], "variants": VARIANTS,
               "effective_warmup_epochs": min(c["warmup"], c["epochs"] / 10),
               "test_opened": False, "subset": True, "seed": c["seed"]}
    receipt_path = args.output / "suite.json"
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        for key in ("source_sha256", "split_sha256", "initial_weights_sha256", "environment"):
            if old[key] != receipt[key]:
                raise ValueError("Changed suite identity; select a new --output")
    else:
        engine.atomic_json(receipt_path, receipt)

    def stop(signum=None, frame=None):
        engine.REQUEST_STOP = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def boundary(epoch, score):
        if time.monotonic() >= deadline:
            stop()

    for name, changes in VARIANTS:
        if engine.REQUEST_STOP or time.monotonic() >= deadline:
            break
        engine.run_experiment(config_for(args, name, changes), epoch_callback=boundary)
        export_summary(args.output)
        if engine.REQUEST_STOP:
            break
    records = export_summary(args.output)
    engine.atomic_json(args.output / "timing.json", {
        "wall_seconds": time.monotonic() - started, "budget_seconds": args.budget_seconds,
        "complete_runs": len(records), "planned_runs": len(VARIANTS),
        "stopped_at_epoch_boundary": engine.REQUEST_STOP,
    })


if __name__ == "__main__":
    main()
