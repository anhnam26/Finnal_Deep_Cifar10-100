"""Install only into PROJECT_ROOT/.venv, with a project-local pip cache."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "generated"


def inventory():
    home = Path.home()
    files = {}
    for folder in (home / ".agents", home / ".codex" / "plugins",
                   home / ".codex" / "skills"):
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    files[str(path)] = [stat.st_size, stat.st_mtime_ns]
    config = home / ".codex" / "config.toml"
    return {"files": files,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()
            if config.exists() else None,
            "packages": sorted((d.metadata["Name"], d.version)
                               for d in importlib.metadata.distributions())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-after", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    before_path = REPORTS / "isolation-before.json"
    if args.audit_after:
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = inventory()
        diff = {"added_files": sorted(set(after["files"]) - set(before["files"])),
                "modified_files": [p for p in before["files"]
                                   if p in after["files"] and before["files"][p] != after["files"][p]],
                "removed_files": sorted(set(before["files"]) - set(after["files"])),
                "global_config_unchanged": before["config_sha256"] == after["config_sha256"],
                "global_packages_unchanged": before["packages"] == after["packages"]}
        (REPORTS / "isolation-audit.json").write_text(
            json.dumps(diff, indent=2), encoding="utf-8")
        print(json.dumps(diff, indent=2))
        return
    if not before_path.exists():
        before_path.write_text(json.dumps(inventory()), encoding="utf-8")
    directory = ROOT / ".venv"
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(directory)
    env = dict(os.environ, PIP_CACHE_DIR=str(ROOT / ".cache" / "pip"),
               PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    def pip(*arguments):
        subprocess.run([str(python), "-m", "pip", *arguments],
                       cwd=ROOT, env=env, check=True)
    pip("install", "--upgrade", "pip")
    index = "cpu" if args.cpu else "cu118"
    pip("install", "torch==2.7.1", "torchvision==0.22.1",
        "--index-url", f"https://download.pytorch.org/whl/{index}")
    pip("install", "-r", str(ROOT / "requirements.txt"))
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True)
    (ROOT / "requirements-lock.txt").write_text(frozen, encoding="utf-8")
    subprocess.run([str(python), "-c",
                    "import torch,timm,optuna; print(torch.__version__,timm.__version__,optuna.__version__); "
                    "print('CUDA',torch.cuda.is_available()); "
                    "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"],
                   cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
