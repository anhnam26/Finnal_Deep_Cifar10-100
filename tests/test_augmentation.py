import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from data import build_transforms
from utils import DEFAULT_CONFIG, validate_config


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
def test_autoaugment_training_only(dataset):
    training, evaluation = build_transforms(dataset, autoaugment=True)
    _, plain_evaluation = build_transforms(dataset)
    assert any(isinstance(t, transforms.AutoAugment) for t in training.transforms)
    assert not any(isinstance(t, transforms.AutoAugment) for t in evaluation.transforms)
    image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    assert torch.equal(evaluation(image), plain_evaluation(image))
    assert torch.isfinite(training(image)).all()
    assert training(image).shape == (3, 32, 32)


def test_autoaugment_config_validation():
    with pytest.raises(ValueError, match="requires"):
        validate_config({**DEFAULT_CONFIG, "autoaugment": True, "augment": False})
    with pytest.raises(ValueError, match="boolean"):
        validate_config({**DEFAULT_CONFIG, "autoaugment": "yes"})
    with pytest.raises(ValueError, match="requires"):
        build_transforms("cifar10", augment=False, autoaugment=True)