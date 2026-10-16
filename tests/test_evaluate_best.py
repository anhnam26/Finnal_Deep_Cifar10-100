"""Checkpoint identity and two-dataset test export, without opening CIFAR test."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def evaluator():
    path = Path(__file__).resolve().parents[1] / "report/evaluate_best.py"
    spec = importlib.util.spec_from_file_location("report_test_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_runs_both_and_exports_per_dataset_and_summary(evaluator, monkeypatch, tmp_path, capsys):
    calls = []
    def measured(dataset, path, data_root, device, batch_size):
        calls.append((dataset, path))
        return {"dataset": dataset, "epoch": 5, "weights": "ema", "samples": 10000,
                "correct": 5000, "accuracy_pct": 50., "top5_pct": 80., "loss": 1.,
                "macro_f1": .5, "checkpoint": str(path)}
    monkeypatch.setattr(evaluator, "evaluate_checkpoint", measured)
    evaluator.main(["--device", "cpu", "--output-dir", str(tmp_path)])
    assert [c[0] for c in calls] == ["cifar10", "cifar100"]
    assert all(path == evaluator.HERE / "checkpoints" / name / "best.pt" for name, path in calls)
    for name, _ in calls:
        assert json.loads((tmp_path / (name + "_test.json")).read_text())["dataset"] == name
    with (tmp_path / "test_summary.csv").open() as stream:
        assert [r["dataset"] for r in csv.DictReader(stream)] == ["cifar10", "cifar100"]
    assert capsys.readouterr().out.count("TEST accuracy=50.00%") == 2


def test_wrong_dataset_is_rejected_before_model_construction(evaluator, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Must reject checkpoint identity first")
    monkeypatch.setattr(evaluator, "build_model", forbidden)
    with pytest.raises(ValueError, match="Checkpoint dataset"):
        evaluator.load_model({"config": {"dataset": "cifar100"}}, "cifar10")


def test_subset_checkpoint_is_not_silently_used_for_official_test(evaluator):
    with pytest.raises(ValueError, match="Smoke/subset"):
        evaluator.load_model({"config": {"dataset": "cifar10", "limit_train": 5000}}, "cifar10")


def test_manual_weights_load_strictly_and_reject_wrong_classifier(evaluator):
    path = evaluator.HERE / "compactmix_manual_train.py"
    spec = importlib.util.spec_from_file_location("manual_test_weights", path)
    manual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manual)
    original = manual.make_model(10)
    checkpoint = {"config": {"dataset": "cifar10", "model": {"family": "compactmix_manual"}},
                  "model": original.state_dict()}
    loaded, _ = evaluator.load_model(checkpoint, "cifar10")
    assert torch.equal(loaded.classifier.weight, original.classifier.weight)
    checkpoint["model"]["classifier.weight"] = torch.zeros(100, 256)
    with pytest.raises(RuntimeError, match="size mismatch"):
        evaluator.load_model(checkpoint, "cifar10")


def test_ambiguous_custom_checkpoint_requires_one_dataset(evaluator):
    with pytest.raises(SystemExit) as error:
        evaluator.main(["--checkpoint", "any_best.pt"])
    assert error.value.code == 2
