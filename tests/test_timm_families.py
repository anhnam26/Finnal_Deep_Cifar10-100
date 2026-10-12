import pytest
import torch


@pytest.mark.parametrize('family', ['resnet', 'resnext', 'convnextv2', 'efficientnet', 'regnet', 'mobilenet', 'vit', 'deit', 'swin', 'hybrid'])
def test_reference_families_are_cifar_native_and_profiled(family):
    from src.models import build_model, profile_model
    torch.set_num_threads(2)
    model = build_model({'family': family}, 100)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 100)
    profile = profile_model(model, enforce=False)
    assert profile['parameters'] > 0 and profile['flops'] > 0
    assert max(profile['resolutions']) >= 8
