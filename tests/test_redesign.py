"""Feasibility and fair-budget checks, not evidence of CIFAR superiority."""

import json
import os

# Process-local test setting, before any test initializes CUDA/CuBLAS.
# Required by PyTorch for deterministic CUDA matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

import train
from data import make_loader
from models import LightNet
from models.lightnet import PreActivationBlock
from prepare_redesign import prepare
from profile_model import profile_operations
from utils import DEFAULT_CONFIG, learning_rate, load_config


@pytest.mark.parametrize("classes", [10, 100])
def test_preact_profile_shapes_gradients(classes):
    torch.set_num_threads(4)
    model = LightNet(classes, "preact")
    report = profile_operations(model)
    assert report["within_limits"] and report["flops"] < 950_000_000
    assert report["learnable_parameters"] < 5_000_000
    x = torch.randn(2, 3, 32, 32)
    h = model.stem(x)
    for stage, channels, size in zip(model.stages, (40, 104, 248), (32, 16, 8)):
        h = stage(h)
        assert h.shape == (2, channels, size, size)
    assert [len(stage) for stage in model.stages] == [2, 3, 4]
    assert all(m.groups == 1 for m in model.modules() if isinstance(m, nn.Conv2d))
    logits = model(x)
    assert logits.shape == (2, classes)
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_preact_identity_keeps_negative_signal():
    block = PreActivationBlock(4, 4).eval()
    nn.init.zeros_(block.conv2.weight)
    x = -torch.ones(2, 4, 8, 8)
    assert torch.equal(block(x), x)  # No post-add ReLU on identity path.


@pytest.mark.parametrize("widths,depths", [((40, 104), (2, 3)),
                                          ((40, 104, 248), (2, 0, 4)),
                                          ((40, 104., 248), (2, 3, 4))])
def test_preact_rejects_invalid_layout(widths, depths):
    with pytest.raises(ValueError, match="three|3 positive integers"):
        LightNet(10, "preact", widths, depths)


def test_matched_redesign_configs(tmp_path):
    path = prepare(tmp_path / "configs", tmp_path / "outputs")
    manifest = json.loads(path.read_text())
    assert manifest["grid_epochs"] == 300 and not manifest["test_evaluation"]
    assert manifest["required_source_fingerprint"]
    configs = [load_config(p) for p in manifest["grid"]]
    assert {c["variant"] for c in configs} == {"baseline", "preact"}
    left, right = configs
    assert {k for k in left if left[k] != right[k]} == {"variant", "output_dir"}
    assert left["epochs"] == 150 and left["warmup_epochs"] == 5
    assert left["seed"] == 42 and left["split_seed"] == 2026
    assert left["batch_size"] == 128 and left["workers"] == 0
    assert left["cutmix_probability"] == left["label_smoothing"] == 0
    assert not left["full_train"] and not left["download"] and left["augment"]
    assert left["max_train_batches"] is left["max_val_batches"] is None
    assert learning_rate(149, 351, 352, 150, 5, .1, 1e-5) == pytest.approx(1e-5)
    assert learning_rate(149, 351, 352, 300, 5, .1, 1e-5) > .05
    smoke = [load_config(p) for p in manifest["smoke"]]
    assert len(smoke) == 2 and all(c["epochs"] == 2 and c["warmup_epochs"] == 0 for c in smoke)
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "configs", tmp_path / "outputs")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="outputs"):
        prepare(tmp_path / "other", tmp_path / "outputs")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_preact_plain_resume_identity(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(11)
    x, y = torch.randn(8, 3, 32, 32, generator=generator), torch.arange(8)

    def loaders(config, use_cuda):
        source = TensorDataset(x, y)
        return (make_loader(source, 4, 0, True, config["seed"]),
                make_loader(source, 4, 0, False, config["seed"]),
                {"fingerprint": "synthetic-fixed", "train_indices": list(range(8)), "validation_indices": []})

    monkeypatch.setattr(train, "build_train_loaders", loaders)
    monkeypatch.setattr(train, "LightNet", lambda classes, variant: LightNet(
        classes, variant, widths=(8, 12, 24), depths=(1, 1, 1)))
    config = {**DEFAULT_CONFIG, "variant": "preact", "epochs": 2, "warmup_epochs": 0,
              "device": device, "amp": True, "deterministic": True, "workers": 0,
              "output_dir": str(tmp_path / "continuous")}
    train.run(config)
    interrupted = {**config, "output_dir": str(tmp_path / "resumed")}
    original = train.train_epoch

    def interrupt(*args):
        if args[-1] == 1:
            raise RuntimeError("simulated interruption")
        return original(*args)

    monkeypatch.setattr(train, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        train.run(interrupted)
    monkeypatch.setattr(train, "train_epoch", original)
    train.run(interrupted, tmp_path / "resumed/last.pt")
    a = torch.load(tmp_path / "continuous/last.pt", weights_only=False, map_location="cpu")
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=False, map_location="cpu")
    assert a["epoch"] == b["epoch"] == 1
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["scaler"] == b["scaler"]
    assert a["best_validation_accuracy"] == b["best_validation_accuracy"]