"""Tests for the literal, standalone file delivered for manual architecture edits."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def single():
    path = Path(__file__).resolve().parents[1] / "report/compactmix_manual_train.py"
    spec = importlib.util.spec_from_file_location("manual_cifar_file", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    return module


@pytest.mark.parametrize("classes", [10, 100])
def test_literal_default_matches_original_weights_outputs_and_budget(single, classes):
    from src.models import build_model, profile_model
    reference = build_model({"family": "compactmix"}, classes).eval()
    actual = single.make_model(classes).eval()
    old_leaves = [m for m in reference.modules() if not list(m.children())]
    new_leaves = [m for m in actual.modules() if not list(m.children())]
    assert len(old_leaves) == len(new_leaves) == 261
    for before, after in zip(old_leaves, new_leaves):
        assert type(before).__name__ == type(after).__name__
        after.load_state_dict(before.state_dict(), strict=True)
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(actual(images), reference(images), rtol=0, atol=0)
    expected = profile_model(reference)
    stats = single.inspect_model(actual, classes, backward=False)
    assert {k: stats[k] for k in ("parameters", "macs", "flops")} == {
        k: expected[k] for k in ("parameters", "macs", "flops")}


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
