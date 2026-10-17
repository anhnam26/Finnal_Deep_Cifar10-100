"""Tests for the literal, standalone file delivered for manual architecture edits."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def single():
    path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("manual_cifar_file", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    return module


@pytest.mark.parametrize("classes,parameters,flops", [(10,5248172,842406944),(100,5271302,842453114)])
def test_default_forward_backward_and_budget(single, classes, parameters, flops):
    model = single.make_model(classes)
    profile = single.inspect_model(model, classes, backward=True)
    assert profile["parameters"] == parameters
    assert profile["flops"] == flops
    assert len(profile["layers"]) == 261


@pytest.mark.parametrize("change,message", [
    ("expand", "block01.bn_expand"),
    ("kernel", "Block01.concat"),
    ("projection", "Block01.ADD"),
    ("classifier", "classifier"),
])
def test_inconsistent_manual_edits_fail_at_the_named_connection(single, change, message):
    model = single.make_model(10)
    if change == "expand":
        model.block01.expand = nn.Conv2d(40, 96, 1, bias=False)
    elif change == "kernel":
        model.block01.context = nn.Conv2d(40, 40, 7, padding=2, groups=40, bias=False)
    elif change == "projection":
        model.block01.project = nn.Conv2d(80, 48, 1, bias=False)
    else:
        model.classifier = nn.Linear(128, 10)
    with pytest.raises(single.ShapeError, match=message):
        single.inspect_model(model, 10)


def test_manually_repaired_40_to_48_channels_runs_forward_and_backward(single):
    model = single.make_model(100)
    model.stem = nn.Conv2d(3, 48, 3, padding=1, bias=False)
    for block in (model.block01, model.block02):
        block.bn_pre = nn.BatchNorm2d(48)
        block.expand = nn.Conv2d(48, 96, 1, bias=False)
        block.bn_expand = nn.BatchNorm2d(96)
        block.local = nn.Conv2d(48, 48, 3, padding=1, groups=48, bias=False)
        block.context = nn.Conv2d(48, 48, 5, padding=2, groups=48, bias=False)
        block.bn_mix = nn.BatchNorm2d(96)
        block.grouped = nn.Conv2d(96, 96, 3, padding=1, groups=8, bias=False)
        block.bn_gate = nn.BatchNorm2d(96)
        block.se_reduce = nn.Conv2d(96, 8, 1)
        block.se_expand = nn.Conv2d(8, 96, 1)
        block.project = nn.Conv2d(96, 48, 1, bias=False)
    model.t1_bn = nn.BatchNorm2d(48)
    model.t1_conv = nn.Conv2d(48, 112, 1, bias=False)
    result = single.inspect_model(model, 100, backward=True)
    assert result["parameters"] < 6_000_000
    assert result["flops"] < 1_000_000_000
    assert result["layers"][-1]["output"] == [1, 100]


def tiny_training(single, monkeypatch, directory, optimizer="sgd"):
    def tiny(classes):
        return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8),
                             nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                             nn.Dropout(.2), nn.Linear(8, classes))
    monkeypatch.setattr(single, "make_model", tiny)
    args = single.parser().parse_args([
        "--smoke", "--dataset", "cifar10", "--device", "cpu", "--epochs", "2",
        "--batch-size", "8", "--output-root", str(directory), "--optimizer", optimizer])
    return single.configs_from_args(args)[0]


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_resume_replays_optimizer_ema_rng_and_emits_only_epoch_lines(single, monkeypatch, tmp_path, capsys, optimizer):
    import copy
    config = tiny_training(single, monkeypatch, tmp_path/"resume", optimizer)
    partial = single.run_experiment(config, stop_after_epochs=1)
    assert partial["status"] == "interrupted"
    assert len(capsys.readouterr().out.splitlines()) == 1
    resumed = single.run_experiment(config)
    assert len(capsys.readouterr().out.splitlines()) == 1
    full_config = copy.deepcopy(config)
    full_config["run_dir"] = str(tmp_path/"whole")
    full = single.run_experiment(full_config)
    assert len(capsys.readouterr().out.splitlines()) == 2
    assert resumed["history"] == full["history"]
    a = single.load_checkpoint(Path(config["run_dir"])/"last.pt")
    b = single.load_checkpoint(Path(full_config["run_dir"])/"last.pt")
    for key in ("model", "ema", "best_weights"):
        for name in a[key]:
            assert torch.equal(a[key][name], b[key][name]), (key, name)
    assert a["global_step"] == b["global_step"] == 8
    for parameter_id, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            other = b["optimizer"]["state"][parameter_id][name]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    assert {p.name for p in Path(config["run_dir"]).glob("*.pt")} == {"best.pt", "last.pt"}
    single.run_experiment(config)  # Already complete: quiet, no training.
    assert capsys.readouterr().out == ""


def test_best_write_failure_cannot_advance_last_or_emit_epoch(single, monkeypatch, tmp_path, capsys):
    config = tiny_training(single, monkeypatch, tmp_path)
    single.run_experiment(config, stop_after_epochs=1)
    capsys.readouterr()
    directory = Path(config["run_dir"])
    before = (directory/"last.pt").read_bytes()
    monkeypatch.setattr(single, "evaluate", lambda *a, **k: {"loss": .1, "accuracy": 99.9})
    original = single.atomic_torch
    def fail_best(path, payload):
        if Path(path).name == "best.pt":
            raise OSError("simulated best write failure")
        original(path, payload)
    monkeypatch.setattr(single, "atomic_torch", fail_best)
    with pytest.raises(OSError, match="best write failure"):
        single.run_experiment(config)
    assert (directory/"last.pt").read_bytes() == before
    assert capsys.readouterr().out == ""


def test_interrupted_validation_does_not_save_partial_epoch(single, monkeypatch, tmp_path, capsys):
    config = tiny_training(single, monkeypatch, tmp_path)
    single.run_experiment(config, stop_after_epochs=1)
    capsys.readouterr()
    directory = Path(config["run_dir"])
    before = (directory/"last.pt").read_bytes()
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(single, "evaluate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        single.run_experiment(config)
    assert (directory/"last.pt").read_bytes() == before
    assert capsys.readouterr().out == ""


def test_changed_source_rejects_resume_without_touching_checkpoints(single, monkeypatch, tmp_path):
    config = tiny_training(single, monkeypatch, tmp_path)
    single.run_experiment(config, stop_after_epochs=1)
    directory = Path(config["run_dir"])
    before = (directory/"last.pt").read_bytes()
    monkeypatch.setattr(single, "source_fingerprint", lambda: "modified-source")
    with pytest.raises(ValueError, match="source/version"):
        single.run_experiment(config)
    assert (directory/"last.pt").read_bytes() == before


def test_missing_last_does_not_overwrite_orphan_best(single, monkeypatch, tmp_path):
    config = tiny_training(single, monkeypatch, tmp_path)
    single.run_experiment(config, stop_after_epochs=1)
    directory = Path(config["run_dir"])
    before = (directory/"best.pt").read_bytes()
    (directory/"last.pt").unlink()
    with pytest.raises(ValueError, match="last.pt is missing"):
        single.run_experiment(config)
    assert (directory/"best.pt").read_bytes() == before
    assert not (directory/"last.pt").exists()


def test_audited_legacy_resume_preserves_optimizer_rng_and_checks_bytes(single, monkeypatch, tmp_path, capsys):
    import copy
    import hashlib
    import json
    config = tiny_training(single, monkeypatch, tmp_path/"checkpoints/run")
    monkeypatch.setattr(single, "ROOT", tmp_path)
    monkeypatch.setattr(single, "ALLOW_LEGACY_RESUME", True)
    single.run_experiment(config, stop_after_epochs=1)
    directory = Path(config["run_dir"])
    path = directory/"last.pt"
    checkpoint = single.load_checkpoint(path)
    checkpoint.update(version=2, source_sha256="audited-legacy-source")
    single.atomic_torch(path, checkpoint)
    original_bytes = path.read_bytes()
    receipt = {"trainer_sha256": single.source_fingerprint(), "runs": {
        directory.relative_to(tmp_path).as_posix(): {
            "version": 2, "source_sha256": "audited-legacy-source",
            "last_sha256": hashlib.sha256(original_bytes).hexdigest()}}}
    single.atomic_json(tmp_path/"checkpoints/legacy_compatibility.json", receipt)
    assert single.verified_legacy_resume(checkpoint, directory)
    path.write_bytes(original_bytes+b"changed")
    assert not single.verified_legacy_resume(checkpoint, directory)
    path.write_bytes(original_bytes)
    resumed = single.run_experiment(config)
    whole_config = copy.deepcopy(config)
    whole_config["run_dir"] = str(tmp_path/"whole")
    whole = single.run_experiment(whole_config)
    assert resumed["history"] == whole["history"]
    a = single.load_checkpoint(path)
    b = single.load_checkpoint(Path(whole_config["run_dir"])/"last.pt")
    for kind in ("model", "ema", "best_weights"):
        for name, value in a[kind].items():
            assert torch.equal(value, b[kind][name])
    assert a["version"] == single.CHECKPOINT_VERSION
    assert a["source_sha256"] == single.source_fingerprint()


def test_read_only_resume_flag_cannot_accidentally_start_training(single):
    with pytest.raises(ValueError, match="requires --resume-run"):
        single.main(["--verify-resume"])
    with pytest.raises(ValueError, match="saved settings"):
        single.main(["--resume-run", "checkpoints/any", "--epochs", "1"])


def test_selection_never_opens_test_and_smoke_cannot_finalize(single, monkeypatch, tmp_path):
    config = tiny_training(single, monkeypatch, tmp_path/"smoke")
    config["epochs"] = 1
    single.run_experiment(config)
    with pytest.raises(ValueError, match="smoke/subset"):
        single.finalize(config["run_dir"])
    class GuardedCifar(single.datasets.FakeData):
        def __init__(self, root, train, download):
            assert train is True
            super().__init__(size=100, image_size=(3, 32, 32), num_classes=10)
            self.targets = [i % 10 for i in range(100)]
        def __getitem__(self, index):
            image, _ = super().__getitem__(index)
            return image, self.targets[index]
    monkeypatch.setattr(single.datasets, "CIFAR10", GuardedCifar)
    config.update(smoke=False, limit_train=32, limit_val=10, run_dir=str(tmp_path/"guard"))
    result = single.run_experiment(config)
    assert result["test_accuracy"] is None
    assert result["subset"] is True


def test_freeze_retrain_resume_and_cached_test_with_fixture_data(single, monkeypatch, tmp_path, capsys):
    """Fixture data exercises the final protocol; never opens official test data."""
    import json
    config = tiny_training(single, monkeypatch, tmp_path)
    config["smoke"] = False
    opened_test = []
    final_directory = Path(config["run_dir"] + "_final")

    class FixtureCifar(single.datasets.FakeData):
        def __init__(self, root, train, download):
            if not train:
                assert single.load_checkpoint(final_directory/"last.pt")["epoch"] == 2
                opened_test.append(True)
            super().__init__(size=20, image_size=(3, 32, 32), num_classes=10)
            self.targets = [i % 10 for i in range(20)]

        def __getitem__(self, index):
            image, _ = super().__getitem__(index)
            return image, self.targets[index]

    monkeypatch.setattr(single.datasets, "CIFAR10", FixtureCifar)
    evaluate = single.evaluate
    scores = iter([10., 20., 30., 40.])
    monkeypatch.setattr(single, "evaluate", lambda *a, **k: {"loss": .1, "accuracy": next(scores)})
    selection = single.run_experiment(config)
    assert selection["best_epoch"] == 2 and selection["best_kind"] == "ema"
    assert opened_test == []
    monkeypatch.setattr(single, "evaluate", evaluate)
    with monkeypatch.context() as changed:
        changed.setattr(single, "source_fingerprint", lambda: "modified")
        with pytest.raises(ValueError, match="Selection source"):
            single.finalize(config["run_dir"])
        assert not final_directory.exists()
    capsys.readouterr()
    assert single.finalize(config["run_dir"], stop_after_epochs=1)["status"] == "interrupted"
    assert len(capsys.readouterr().out.splitlines()) == 1
    assert opened_test == []
    frozen_bytes = (final_directory/"frozen.json").read_bytes()
    final = single.finalize(config["run_dir"])
    assert len(capsys.readouterr().out.splitlines()) == 1
    assert opened_test == [True]
    assert final["test_accuracy"] is not None and final["test_checkpoint_sha256"]
    split = json.loads((final_directory/"split.json").read_text())
    assert len(split["training"]) == 20 and split["validation"] == []
    assert (final_directory/"frozen.json").read_bytes() == frozen_bytes
    assert single.finalize(config["run_dir"])["test_accuracy"] == final["test_accuracy"]
    assert capsys.readouterr().out == "" and opened_test == [True]
    assert {p.name for p in final_directory.glob("*.pt")} == {"best.pt", "last.pt"}


def test_file_is_self_contained_and_checks_work_from_another_directory(single, tmp_path):
    import ast
    import os
    import shutil
    import subprocess
    import sys
    path = Path(single.__file__)
    allowed = set(sys.stdlib_module_names) | {"torch", "torchvision", "numpy"}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in allowed for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module.split(".")[0] in allowed
    isolated = tmp_path/"only_this_file.py"
    shutil.copyfile(path, isolated)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, "-B", "-X", "utf8", str(isolated),
                                "--check", "--dataset", "both"],
                               cwd=tmp_path, env=env, capture_output=True, text=True,
                               encoding="utf-8", timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("FORWARD + BACKWARD OK") == 2
    assert "5,248,172" in completed.stdout and "5,271,302" in completed.stdout
