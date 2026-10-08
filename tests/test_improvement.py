"""Hybrid feasibility, recipe isolation, and epoch-boundary resume."""

import json

import pytest
import torch
from torch.utils.data import TensorDataset

import train
from data import make_loader
from models import LightNet
from prepare_improvement import prepare
from profile_model import profile_operations
from utils import DEFAULT_CONFIG, load_config


@pytest.mark.parametrize("classes", [10, 100])
def test_hybrid_profile_shape_gradient(classes):
    torch.set_num_threads(4)
    model = LightNet(classes, "hybrid")
    assert profile_operations(model)["within_limits"]
    x = torch.randn(2, 3, 32, 32)
    features = model.stages(model.stem(x))
    assert features.shape == (2, 384, 8, 8)
    logits = model(x)
    assert logits.shape == (2, classes)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_improvement_configs(tmp_path):
    path = prepare(tmp_path/"configs", tmp_path/"outputs")
    manifest = json.loads(path.read_text())
    configs = [load_config(p) for p in manifest["grid"]]
    assert len(configs) == 4 and manifest["grid_epochs"] == 1200
    assert {(c["variant"], c["cutmix_probability"]) for c in configs} == {
        (variant, probability) for variant in ("baseline", "hybrid") for probability in (0., .5)}
    assert all(c["epochs"] == 300 and c["seed"] == 42 and c["split_seed"] == 2026
               and c["workers"] == 0 and c["label_smoothing"] == 0
               and not c["full_train"] and c["max_train_batches"] is None
               and c["max_val_batches"] is None for c in configs)
    with pytest.raises(FileExistsError):
        prepare(tmp_path/"configs", tmp_path/"outputs")


def test_hybrid_cutmix_resume_identity(tmp_path, monkeypatch):
    torch.set_num_threads(4)
    x, y = torch.randn(8, 3, 32, 32), torch.arange(8)

    def loaders(config, use_cuda):
        source = TensorDataset(x, y)
        return (make_loader(source, 4, 0, True, config["seed"]),
                make_loader(source, 4, 0, False, config["seed"]),
                {"fingerprint":"synthetic-fixed", "train_indices":list(range(8)), "validation_indices":[]})

    monkeypatch.setattr(train, "build_train_loaders", loaders)
    monkeypatch.setattr(train, "LightNet", lambda classes, variant: LightNet(
        classes, variant, widths=(8, 12, 16, 24), depths=(1, 1, 1, 1)))
    config = {**DEFAULT_CONFIG, "variant":"hybrid", "epochs":2, "warmup_epochs":0,
              "device":"cpu", "deterministic":True, "workers":0,
              "cutmix_alpha":1., "cutmix_probability":.5,
              "output_dir":str(tmp_path/"continuous")}
    train.run(config)
    interrupted = {**config, "output_dir":str(tmp_path/"resumed")}
    original = train.train_epoch

    def interrupt(*args):
        if args[-1] == 1:
            raise RuntimeError("simulated interruption")
        return original(*args)

    monkeypatch.setattr(train, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        train.run(interrupted)
    monkeypatch.setattr(train, "train_epoch", original)
    train.run(interrupted, tmp_path/"resumed/last.pt")
    a = torch.load(tmp_path/"continuous/last.pt", weights_only=False)
    b = torch.load(tmp_path/"resumed/last.pt", weights_only=False)
    assert a["epoch"] == b["epoch"] == 1
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["best_validation_accuracy"] == b["best_validation_accuracy"]