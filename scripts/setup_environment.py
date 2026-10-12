"""Reuse existing Python; install missing distributions only into .deps."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

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
            "packages": sorted([d.metadata["Name"], d.version]
                               for d in importlib.metadata.distributions())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-after", action="store_true")
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
    from packaging.requirements import Requirement
    directory = ROOT / ".deps"
    directory.mkdir(exist_ok=True)
    paths = [str(directory), *sys.path]
    env = dict(os.environ, PIP_CACHE_DIR=str(ROOT / ".cache" / "pip"),
               PIP_DISABLE_PIP_VERSION_CHECK="1")
    queue = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    checked = set()
    resolved = {}
    while queue:
        raw = queue.pop(0)
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        if raw in checked:
            continue
        checked.add(raw)
        name = requirement.name.lower().replace("_", "-")
        candidates = [d for d in importlib.metadata.distributions(path=paths)
                      if d.metadata["Name"].lower().replace("_", "-") == name]
        matching = [d for d in candidates if d.version in requirement.specifier]
        dist = next((d for d in matching if d.version == resolved.get(name)),
                    matching[0] if matching else None)
        if dist is None:
            if name in ("torch", "torchvision", "numpy"):
                raise RuntimeError(f"Existing {name} is incompatible with {raw}; do not replace the selected runtime")
            subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(directory),
                            "--upgrade", "--no-deps", raw], cwd=ROOT, env=env, check=True)
            importlib.invalidate_caches()
            dist = next(d for d in importlib.metadata.distributions(path=[str(directory)])
                        if d.metadata["Name"].lower().replace("_", "-") == name)
        queue.extend(dist.requires or [])
        resolved[name] = dist.version
    versions = sorted(f'{name}=={version}' for name, version in resolved.items())
    (ROOT / "requirements-lock.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-c",
                    "import sys; sys.path.insert(0,'.deps'); import torch,timm,optuna; "
                    "print(sys.executable); print(torch.__version__,timm.__version__,optuna.__version__); "
                    "print('CUDA',torch.cuda.is_available())"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
