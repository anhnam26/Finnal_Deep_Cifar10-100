import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import TensorDataset

import train
from data import (build_transforms, make_loader, split_fingerprint, stratified_split)
from models import LightNet
from profile_model import profile_operations
from utils import (DEFAULT_CONFIG, cutmix, learning_rate, load_config,
                   seed_everything, validate_config)


@pytest.fixture(autouse=True)
def reproducible():
    torch.set_num_threads(2)
    seed_everything(7, deterministic=True)


@pytest.mark.parametrize("classes", [10, 100])
@pytest.mark.parametrize("variant", ["fusion", "no_fusion"])
def test_model_shapes_gradients_and_limits(classes, variant):
    model = LightNet(classes, variant)
    report = profile_operations(model)
    assert report["within_limits"]
    assert report["learnable_parameters"] <= 6_000_000
    assert report["flops"] < 1_000_000_000
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, classes)
    nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_profiler_known_conv_and_linear_counts():
    model = nn.Conv2d(3, 4, 3, padding=1, bias=True)
    result = profile_operations(model, (1, 3, 8, 8))
    assert result["macs"] == 4 * 8 * 8 * 3 * 3 * 3
    assert result["other_flops"] == 4 * 8 * 8
    assert result["learnable_parameters"] == 4 * 3 * 3 * 3 + 4
    linear = profile_operations(nn.Linear(8, 3), (1, 8))
    assert linear["macs"] == 24
    assert linear["flops"] == 51


def test_profiler_rejects_unknown_operations():
    with pytest.raises(ValueError, match="Unsupported"):
        profile_operations(nn.Tanh())


@pytest.mark.parametrize("classes,per_class,held_out", [(10, 5000, 500), (100, 500, 50)])
def test_official_size_stratified_split(classes, per_class, held_out):
    targets = np.repeat(np.arange(classes), per_class)
    training, validation = stratified_split(targets, classes, held_out, 2026)
    assert len(training) == 45000 and len(validation) == 5000
    assert not set(training) & set(validation)
    assert set(training) | set(validation) == set(range(50000))
    assert np.all(np.bincount(targets[validation], minlength=classes) == held_out)
    assert (training, validation) == stratified_split(targets, classes, held_out, 2026)
    second = stratified_split(targets, classes, held_out, 2027)
    assert split_fingerprint(training, validation) != split_fingerprint(*second)


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
def test_eval_transforms_are_deterministic(dataset):
    _, evaluation = build_transforms(dataset)
    image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    assert torch.equal(evaluation(image), evaluation(image))
    assert evaluation(image).shape == (3, 32, 32)


def test_learning_rate_endpoints():
    assert learning_rate(0, 0, 10, 4, 1, 0.1, 0.001) == pytest.approx(0.01)
    assert learning_rate(0, 9, 10, 4, 1, 0.1, 0.001) == pytest.approx(0.1)
    assert learning_rate(1, 0, 10, 4, 1, 0.1, 0.001) == pytest.approx(0.1)
    assert learning_rate(3, 9, 10, 4, 1, 0.1, 0.001) == pytest.approx(0.001)
    values = [learning_rate(e, s, 10, 4, 1, 0.1, 0.001) for e in range(1, 4) for s in range(10)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_cutmix_lambda_matches_actual_patch(monkeypatch):
    monkeypatch.setattr(torch, "randperm", lambda n, device: torch.tensor([1, 0], device=device))
    images = torch.stack([torch.zeros(3, 32, 32), torch.ones(3, 32, 32)])
    labels = torch.tensor([0, 1])
    for _ in range(20):
        mixed, other, lam = cutmix(images, labels, 1.0)
        assert mixed[0].mean().item() == pytest.approx(1 - lam)
        assert torch.equal(other, torch.tensor([1, 0]))
        assert 0 <= lam <= 1
        assert torch.equal(images[0], torch.zeros_like(images[0]))


def test_evaluation_sample_weighting():
    class Fixed(nn.Module):
        def forward(self, images):
            return images

    logits = torch.tensor([[3., 0.], [0., 2.], [1., 0.]])
    labels = torch.tensor([0, 1, 1])
    loader = make_loader(TensorDataset(logits, labels), 2, 0)
    result = train.evaluate_loader(Fixed(), loader, torch.device("cpu"))
    assert result["loss"] == pytest.approx(nn.functional.cross_entropy(logits, labels).item())
    assert result["accuracy"] == pytest.approx(100 * 2 / 3)
    assert result["samples"] == 3


def test_tiny_overfit_and_checkpoint_identity(tmp_path):
    model = LightNet(10, widths=(8, 12, 24), depths=(1, 1, 1))
    images, labels = torch.randn(8, 3, 32, 32), torch.arange(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    first_loss = None
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        first_loss = loss.item() if first_loss is None else first_loss
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        output = model(images)
    assert (output.argmax(1) == labels).float().mean().item() == 1.0
    assert loss.item() < first_loss * 0.1
    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = LightNet(10, widths=(8, 12, 24), depths=(1, 1, 1)).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.inference_mode():
        assert torch.equal(output, restored(images))


def test_config_validation_and_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"epochs": 2, "warmup_epochs": 0}), encoding="utf-8")
    assert load_config(path)["epochs"] == 2
    path.write_text(json.dumps({"typo": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    with pytest.raises(ValueError):
        validate_config({**DEFAULT_CONFIG, "full_train": True, "max_train_batches": 2})
    with pytest.raises(ValueError):
        validate_config({**DEFAULT_CONFIG, "cutmix_probability": 0.5, "cutmix_alpha": 0})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("optimizer_name", ["sgd", "adam"])
def test_cuda_amp_training(optimizer_name):
    seed_everything(42, deterministic=False)
    model = LightNet(100).cuda()
    config = {**DEFAULT_CONFIG, "optimizer": optimizer_name, "epochs": 2,
              "warmup_epochs": 0, "lr": 0.001 if optimizer_name == "adam" else 0.1}
    loader = make_loader(TensorDataset(torch.randn(16, 3, 32, 32), torch.arange(16)), 8, 0)
    optimizer = train.build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", init_scale=4096.)
    result = train.train_epoch(model, loader, optimizer, scaler, config, torch.device("cuda"), 0)
    assert np.isfinite(result["train_loss"])
    assert result["train_samples"] == 16
    assert result["peak_allocated_bytes"] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_train_validation_dataset_objects_are_separate(monkeypatch, tmp_path):
    import data
    instances = []

    class FakeCIFAR:
        def __init__(self, root, train, download, transform):
            assert train is True  # Training entry point may not instantiate test data.
            self.targets = np.repeat(np.arange(10), 501).tolist()
            self.transform = transform
            instances.append(self)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return self.transform(Image.new("RGB", (32, 32))), self.targets[index]

    monkeypatch.setitem(data.DATASETS, "cifar10", (FakeCIFAR, 10))
    config = {**DEFAULT_CONFIG, "workers": 0, "data_root": str(tmp_path)}
    training, validation, meta = data.build_train_loaders(config)
    assert training.dataset.dataset is not validation.dataset.dataset
    assert len(instances) == 2
    assert meta["validation_count"] == 5000 and meta["train_count"] == 10
    assert any(type(t).__name__ == "RandomCrop" for t in instances[0].transform.transforms)
    assert not any(type(t).__name__.startswith("Random") for t in instances[1].transform.transforms)
    instances.clear()
    training, validation, meta = data.build_train_loaders({**config, "full_train": True})
    assert validation is None and meta["validation_count"] == 0
    assert meta["train_count"] == 5010 and len(instances) == 1


def test_test_evaluation_requires_explicit_confirmation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "evaluate.py"), "--checkpoint", str(tmp_path / "missing.pt"),
         "--output", str(tmp_path / "test.json")], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--confirm-final-test" in result.stderr
    assert not (tmp_path / "test.json").exists()


def test_grid_covers_required_experiments(tmp_path):
    from prepare_study import prepare
    manifest = json.loads(prepare(tmp_path/"configs", tmp_path/"outputs", "matrix").read_text())
    paths = manifest["grid"]
    assert len(paths) == 12
    actual = set()
    for path in paths:
        config = load_config(path)
        assert config["epochs"] == 150 and config["warmup_epochs"] == 5
        assert config["max_train_batches"] is None
        assert config["max_val_batches"] is None
        actual.add((config["dataset"], config["optimizer"], config["lr"]))
    expected = {
        (dataset, optimizer, lr)
        for dataset in ("cifar10", "cifar100")
        for optimizer, rates in (("sgd", [0.05, 0.1, 0.15]), ("adam", [0.0003, 0.001, 0.003]))
        for lr in rates
    }
    assert actual == expected


@pytest.mark.parametrize("optimizer_name", ["sgd", "adam"])
def test_epoch_resume_is_identical(tmp_path, monkeypatch, optimizer_name):
    # Synthetic integration test is NOT a CIFAR accuracy experiment.
    x = torch.randn(8, 3, 32, 32)
    y = torch.arange(8) % 10

    def fake_loaders(config, use_cuda):
        source = TensorDataset(x, y)
        return (make_loader(source, 4, 0, True, config["seed"]),
                make_loader(source, 4, 0, False, config["seed"]),
                {"fingerprint": "synthetic-fixed", "train_indices": list(range(8)), "validation_indices": []})

    monkeypatch.setattr(train, "build_train_loaders", fake_loaders)
    monkeypatch.setattr(train, "LightNet", lambda classes, variant: LightNet(
        classes, variant, widths=(8, 12, 24), depths=(1, 1, 1)))
    config = {
        **DEFAULT_CONFIG, "optimizer": optimizer_name, "epochs": 2, "warmup_epochs": 0,
        "device": "cpu", "workers": 0, "deterministic": True,
        "cutmix_alpha": 1.0, "cutmix_probability": 0.7,
        "output_dir": str(tmp_path / "continuous"),
    }
    train.run(config)
    interrupted_config = {**config, "output_dir": str(tmp_path / "resumed")}
    original = train.train_epoch

    def interrupt(*args):
        if args[-1] == 1:
            raise RuntimeError("simulated interruption")
        return original(*args)

    monkeypatch.setattr(train, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        train.run(interrupted_config)
    monkeypatch.setattr(train, "train_epoch", original)
    train.run(interrupted_config, tmp_path / "resumed" / "last.pt")
    a = torch.load(tmp_path / "continuous" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    assert a["epoch"] == b["epoch"] == 1
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert a["best_validation_accuracy"] == b["best_validation_accuracy"]
    assert random.getstate() == b["rng"]["python"]