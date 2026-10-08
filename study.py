"""Sequential GPU queue: smoke -> measured budget -> development grid. Never test."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    # Keep the supervising process stdlib-only: importing utils loads PyTorch.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def source_fingerprint(root=ROOT):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(list(root.glob("*.py")) + list((root / "models").glob("*.py"))):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def queue_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    config["output_dir"] = str(output if output.is_absolute() else ROOT / output)
    return config


def now():
    return datetime.now(timezone.utc).isoformat()


def completed(config):
    path = Path(config["output_dir"])/"summary.json"
    return path.exists() and json.loads(path.read_text())["completed_epochs"] == config["epochs"]


def measure_budget(manifest):
    import numpy as np
    from utils import load_config

    values = {0:[], 2:[]}
    detailed = []
    for path in manifest["smoke"]:
        config = load_config(path)
        if not completed(config):
            raise ValueError("All smoke runs must finish before estimating budget")
        output = Path(config["output_dir"])
        with (output/"metrics.csv").open(newline="") as file:
            metrics = list(csv.DictReader(file))
        with (output/"timing.csv").open(newline="") as file:
            timing = list(csv.DictReader(file))
        if len(metrics) != config["epochs"] or len(timing) != config["epochs"]:
            raise ValueError("Incomplete smoke epoch metrics/timing")
        for row in metrics:
            if (int(row["train_samples"]) != 45000 or int(row["validation_samples"]) != 5000
                    or not np.isfinite(float(row["train_loss"])) or not np.isfinite(float(row["validation_loss"]))):
                raise ValueError("Smoke sample-count/finite-loss validation failed")
            steps = int(np.ceil(int(row["train_samples"])/config["batch_size"]))
            # Dynamic loss scaling may skip initial overflow steps while calibrating.
            # Fail on persistent (>1%) skipped updates, not one bounded scale adjustment.
            if int(row["skipped_amp_steps"])/steps > .01:
                raise ValueError("Smoke AMP skipped-step rate exceeds1%; investigate before long grid")
        # Use both full epochs including first-epoch setup: conservative estimate.
        seconds = [float(row["epoch_wall_seconds"]) for row in timing]
        detailed.append({"dataset":config["dataset"], "variant":config["variant"],
                         "workers":config["workers"], "epoch_wall_seconds":seconds,
                         "train_images_per_second":[float(row["images_per_second"]) for row in metrics],
                         "skipped_amp_steps":[int(row["skipped_amp_steps"]) for row in metrics],
                         "max_skipped_amp_fraction":max(int(row["skipped_amp_steps"])/np.ceil(45000/config["batch_size"]) for row in metrics),
                         "source_fingerprint":json.loads((output/"environment.json").read_text()).get("source_fingerprint")
                                              if (output/"environment.json").exists() else None,
                         "peak_allocated_bytes":max(int(row["peak_allocated_bytes"]) for row in metrics),
                         "peak_reserved_bytes":max(int(row["peak_reserved_bytes"]) for row in metrics)})
        if config["dataset"] == "cifar100":
            values[config["workers"]].extend(seconds)
    medians = {key:float(np.median(val)) for key,val in values.items() if val}
    if not medians:
        # New CIFAR10-only trials use fixed workers0, measured on their own smoke.
        for row in detailed:
            values.setdefault(row["workers"], []).extend(row["epoch_wall_seconds"])
        medians = {key:float(np.median(val)) for key,val in values.items() if val}
    if not medians:
        raise ValueError("No remaining smoke measurements for budget")
    selected = min(medians,key=medians.get)
    # CIFAR10 and CIFAR100 have the same input sizes; use slower matched measurement.
    chosen = [max(row["epoch_wall_seconds"]) for row in detailed if row["workers"] == selected]
    seconds = max(chosen)
    reserve = manifest["budget_reserve_factor"]
    return {"measured_utc":now(), "measurements":detailed, "median_seconds_by_workers":medians,
            "selected_workers":selected, "conservative_epoch_seconds":seconds,
            "grid_hours_with_reserve":seconds*manifest["grid_epochs"]/3600*reserve,
            "maximum_9600_epoch_hours_with_reserve":seconds*9600/3600*reserve,
            "maximum_9600_epoch_hours_adjusted_full50k":seconds*(9600-3600+3600*50000/45000)/3600*reserve,
            "reserve_factor":reserve, "max_grid_hours":manifest["max_grid_hours"],
            "amp_acceptance":"Finite train/validation losses, full sample counts, <=1% skipped steps each epoch; adaptive scaling skips recorded, not hidden",
            "note":"Laptop thermals and external GPU usage can change throughput. 9600-epoch estimate assumes12final runs x300epochs and adds50k/45k training adjustment separately (conservative includes val time); not a promise. Grid only; recipe/final stages require later validation decisions."}


def execute(config_path, root, status, evaluate=True, source_root=ROOT):
    config = queue_config(config_path)
    output = Path(config["output_dir"])
    logs = root/"logs"
    logs.mkdir(parents=True,exist_ok=True)
    name = output.name
    status.update({"phase":"training", "current_run":name, "config":str(config_path), "updated_utc":now()})
    write_json(root/"status.json",status)
    if not completed(config):
        command = [sys.executable,"-B","-u",str(source_root/"train.py"),"--config",str(config_path)]
        if (output/"last.pt").exists():
            command += ["--resume",str(output/"last.pt")]
        elif output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Incomplete run has no resumable checkpoint: {output}")
        with (logs/f"{name}.stdout.log").open("a",encoding="utf-8") as out, (logs/f"{name}.stderr.log").open("a",encoding="utf-8") as err:
            process = subprocess.Popen(command,cwd=source_root,stdout=out,stderr=err)
            status["child_pid"] = process.pid
            write_json(root/"status.json",status)
            code = process.wait()
            status.pop("child_pid",None)
            if code:
                raise RuntimeError(f"Training failed ({code}): {name}. See {logs}")
    if not completed(config):
        raise RuntimeError(f"Run exited without finishing: {name}")
    if evaluate:
        destination = output/"evaluation_validation"
        if not (destination/"complete.json").exists():
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Incomplete evaluation export; preserve and inspect before retry: {destination}")
            status.update({"phase":"exporting_validation", "updated_utc":now()})
            write_json(root/"status.json",status)
            with (logs/f"{name}.evaluation.log").open("a",encoding="utf-8") as log:
                command = [sys.executable,"-B","-u",str(source_root/"evaluate_full.py"),
                           "--checkpoint",str(output/"best.pt"), "--output-dir",str(destination), "--device","cuda"]
                process = subprocess.Popen(command,cwd=source_root,stdout=log,stderr=log)
                status["child_pid"] = process.pid
                write_json(root/"status.json",status)
                code = process.wait()
                status.pop("child_pid",None)
                if code:
                    raise RuntimeError(f"Validation export failed: {name}; see evaluation.log")
    status["updated_utc"] = now()
    write_json(root/"status.json",status)


def run_queue(manifest_path, stage="all"):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["output_root"])
    source_root = Path(manifest.get("source_root", ROOT)).resolve()
    fingerprint = source_fingerprint(source_root)
    if manifest.get("required_source_fingerprint", fingerprint) != fingerprint:
        raise RuntimeError("Manifest source snapshot fingerprint mismatch")
    root.mkdir(parents=True,exist_ok=True)
    lock = root/"queue.lock"
    # Exclusive creation: stale locks must be examined, never blindly removed.
    with lock.open("x",encoding="utf-8") as file:
        json.dump({"pid":os.getpid(),"created_utc":now(),"manifest":str(manifest_path)},file)
    status = {"pid":os.getpid(),"started_utc":now(),"phase":"starting", "stage":stage,
              "source_fingerprint":fingerprint, "source_root":str(source_root), "test_evaluated":False,
              "completed_grid_runs":0,"total_grid_runs":len(manifest["grid"])}
    gpu_lock = ROOT / "outputs" / "training.lock"
    owns_gpu_lock = False
    try:
        gpu_lock.parent.mkdir(parents=True, exist_ok=True)
        with gpu_lock.open("x", encoding="utf-8") as file:
            json.dump({"pid":os.getpid(), "manifest":str(manifest_path), "created_utc":now()}, file)
        owns_gpu_lock = True
        for config_path in manifest["smoke"]:
            if (root/"STOP_AFTER_RUN").exists():
                status["phase"] = "stopped_between_runs"
                return
            if source_fingerprint(source_root) != status["source_fingerprint"]:
                raise RuntimeError("Source changed while smoke queue running; fail-stop")
            execute(config_path,root,status,evaluate=True,source_root=source_root)
        subprocess.run([sys.executable, "-B", str(ROOT/"study.py"),
                        "--manifest", str(manifest_path), "--budget-only"], cwd=ROOT, check=True)
        budget = json.loads((root/"budget.json").read_text(encoding="utf-8"))
        if stage == "smoke":
            status["phase"] = "smoke_completed"
            return
        if budget["grid_hours_with_reserve"] > manifest["max_grid_hours"]:
            status.update({"phase":"budget_blocked", "reason":"Measured grid exceeds configured time allowance"})
            return
        required_bytes = 12 * 1024**3
        if shutil.disk_usage(root).free < required_bytes:
            raise RuntimeError("Insufficient free disk for grid checkpoints (need at least12GiB)")
        # Freeze workers in all configs before the first grid run. Never modify a started run.
        for path in manifest["grid"]:
            supplied = json.loads(Path(path).read_text())
            if Path(supplied["output_dir"]).exists():
                if supplied["workers"] != budget["selected_workers"]:
                    raise RuntimeError("Cannot change workers after grid has started")
            else:
                supplied["workers"] = budget["selected_workers"]
                write_json(path,supplied)
        write_json(root/"frozen_grid_protocol.json", {"manifest":manifest, "budget":budget,
                   "source_fingerprint":fingerprint, "test_evaluation":False})
        for path in manifest["grid"]:
            if (root/"STOP_AFTER_RUN").exists():
                status["phase"] = "stopped_between_runs"
                return
            if source_fingerprint(source_root) != status["source_fingerprint"]:
                raise RuntimeError("Source changed while queue running; fail-stop to preserve comparable experiments")
            execute(path,root,status,source_root=source_root)
            status["completed_grid_runs"] += 1
            subprocess.run([sys.executable, "-B", str(ROOT/"study.py"),
                            "--manifest", str(manifest_path), "--report-only"], cwd=ROOT, check=True)
            write_json(root/"status.json",status)
        status["phase"] = "grid_completed_awaiting_recipe_selection"
    except BaseException as error:
        status.update({"phase":"failed", "error":str(error), "error_type":type(error).__name__})
        raise
    finally:
        status["updated_utc"] = now()
        write_json(root/"status.json",status)
        lock.unlink(missing_ok=True)
        if owns_gpu_lock:
            gpu_lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",default=str(ROOT/"configs"/"study"/"manifest.json"))
    parser.add_argument("--stage",choices=["all","smoke"],default="all")
    helper = parser.add_mutually_exclusive_group()
    helper.add_argument("--budget-only", action="store_true")
    helper.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.budget_only or args.report_only:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if args.budget_only:
            write_json(Path(manifest["output_root"])/"budget.json", measure_budget(manifest))
        else:
            from reporting import aggregate_study
            aggregate_study(Path(manifest["output_root"]))
    else:
        run_queue(args.manifest,args.stage)


if __name__ == "__main__":
    main()