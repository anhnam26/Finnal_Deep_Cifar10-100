"""Feasibility, controlled ablation and exact resume; no accuracy claims."""

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

import train
from data import make_loader
from models import StageFusionNet
from models.stagefusion import ResidualUnit
from prepare_study import prepare
from profile_model import profile_operations
from utils import DEFAULT_CONFIG, load_config


def test_stage_shapes_and_ablation_path():
    torch.set_num_threads(2)
    model = StageFusionNet().eval()
    plain = StageFusionNet(variant="no_fusion").eval()
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if k in plain.state_dict()})
    x = torch.randn(2, 3, 32, 32)
    h = model.stem(x)
    for stage, width, size in zip(model.stages, (40, 104, 240), (32, 16, 8)):
        h = stage(h)
        assert h.shape == (2, width, size, size)
    for gate in model.gates:
        nn.init.zeros_(gate[0].weight)
        nn.init.constant_(gate[0].bias, -1000.)
    with torch.inference_mode():
        assert torch.equal(model(x), plain(x))
    block = ResidualUnit(4, 4).eval()
    nn.init.zeros_(block.conv2.weight)
    negative = -torch.ones(2, 4, 8, 8)
    assert torch.equal(block(negative), negative)


def test_layernorm_and_broadcast_profiler():
    from models.stagefusion import Multiply
    assert profile_operations(nn.LayerNorm(8), (2, 8))["flops"] == 8 * 16 + 3 * 2

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)
            self.sigmoid = nn.Sigmoid()
            self.multiply = Multiply()

        def forward(self, x):
            return self.multiply(x, self.sigmoid(self.linear(x)))

    report = profile_operations(Gate(), (1, 4))
    assert report["macs"] == 4
    assert report["flops"] == 8 + 1 + 4 + 4


@pytest.mark.parametrize("variant", ["fusion", "no_fusion"])
def test_reject_invalid_layout_and_old_variants(variant):
    with pytest.raises(ValueError, match="3 positive"):
        StageFusionNet(10, variant, (40, 104), (2, 3))
    with pytest.raises(ValueError, match="variant"):
        StageFusionNet(10, "baseline")


def test_screen_and_matrix_configs(tmp_path):
    screen = json.loads(prepare(tmp_path / "screen", tmp_path / "screen_out").read_text())
    left, right = [load_config(p) for p in screen["grid"]]
    assert {k for k in left if left[k] != right[k]} == {"variant", "output_dir"}
    assert left["dataset"] == "cifar100" and screen["grid_epochs"] == 300
    assert not screen["test_evaluation"]
    matrix = json.loads(prepare(tmp_path / "matrix", tmp_path / "matrix_out", "matrix").read_text())
    configs = [load_config(p) for p in matrix["grid"]]
    assert len(configs) == 12 and matrix["grid_epochs"] == 1800
    assert {c["variant"] for c in configs} == {"fusion"}
    assert len({(c["dataset"], c["optimizer"], c["lr"]) for c in configs}) == 12
    assert all(c["max_train_batches"] is c["max_val_batches"] is None for c in configs)
    assert all(c["seed"] == 42 and c["split_seed"] == 2026 for c in configs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
@pytest.mark.parametrize("variant", ["fusion", "no_fusion"])
def test_resume_identity(tmp_path, monkeypatch, device, optimizer, variant):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    x = torch.randn(8, 3, 32, 32, generator=torch.Generator().manual_seed(11))
    y = torch.arange(8)

    def loaders(config, use_cuda):
        source = TensorDataset(x, y)
        return (make_loader(source, 4, 0, True, config["seed"]),
                make_loader(source, 4, 0, False, config["seed"]),
                {"fingerprint": "synthetic-fixed", "train_indices": list(range(8)), "validation_indices": []})

    monkeypatch.setattr(train, "build_train_loaders", loaders)
    monkeypatch.setattr(train, "LightNet", lambda classes, choice: StageFusionNet(
        classes, choice, widths=(8, 12, 24), depths=(1, 1, 1)))
    config = {**DEFAULT_CONFIG, "variant": variant, "optimizer": optimizer,
              "lr": .01 if optimizer == "sgd" else .001, "epochs": 2, "warmup_epochs": 0,
              "device": device, "deterministic": True, "workers": 0,
              "cutmix_alpha": 1., "cutmix_probability": .7,
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