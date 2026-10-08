"""Regression checks for lightweight supervision and frozen legacy execution."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import study
from prepare_study import prepare


def test_supervisor_import_is_lightweight():
    code = "import study,sys; assert 'torch' not in sys.modules; assert 'reporting' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=study.ROOT, check=True)


def test_snapshot_mismatch_before_lock(tmp_path):
    path = prepare(tmp_path/"configs", tmp_path/"outputs")
    manifest = json.loads(path.read_text())
    manifest["source_root"] = str(tmp_path/"legacy")
    manifest["required_source_fingerprint"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        study.run_queue(path)
    assert not (tmp_path/"outputs/queue.lock").exists()


def test_execute_uses_legacy_source(tmp_path, monkeypatch):
    source = tmp_path/"legacy"
    source.mkdir()
    output = tmp_path/"run"
    output.mkdir()
    (output/"last.pt").touch()
    path = tmp_path/"config.json"
    path.write_text(json.dumps({"output_dir":str(output), "epochs":150}))
    calls = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs["cwd"]))

        def wait(self):
            (output/"summary.json").write_text('{"completed_epochs":150}')
            return 0

    monkeypatch.setattr(study.subprocess, "Popen", Process)
    study.execute(path, tmp_path/"queue", {}, evaluate=False, source_root=source)
    assert calls[0][0][3] == str(source/"train.py")
    assert calls[0][0][-2:] == ["--resume", str(output/"last.pt")]
    assert calls[0][1] == source


def test_remaining_manifest_never_requeues_deleted():
    manifest = json.loads((study.ROOT/"configs/candidate/manifest.json").read_text())
    excluded = set(manifest.get("excluded_completed_configs", []))
    assert not excluded.intersection(manifest["grid"] + manifest["smoke"])
    assert all(Path(p).exists() for p in excluded)


def test_excluded_run_cannot_be_reintroduced(tmp_path):
    path = prepare(tmp_path/"configs", tmp_path/"outputs")
    manifest = json.loads(path.read_text())
    manifest["excluded_completed_configs"] = [manifest["grid"][0]]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Deleted completed configs"):
        study.run_queue(path)
    assert not (tmp_path/"outputs/queue.lock").exists()


def test_candidate_paths_are_absolute():
    manifest = json.loads((study.ROOT/"configs/candidate/manifest.json").read_text())
    for path in manifest["grid"] + manifest["smoke"]:
        config = json.loads(Path(path).read_text())
        assert Path(config["data_root"]).is_absolute()
        assert Path(config["output_dir"]).is_absolute()