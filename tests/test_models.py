import pytest
import torch


@pytest.mark.parametrize('classes', [10, 100])
@pytest.mark.parametrize('family', ['compactmix', 'wrn', 'preactresnet', 'convnext'])
def test_models_have_valid_gradients_and_exam_budget(family, classes):
    from src.models import build_model, profile_model
    torch.set_num_threads(2)
    model = build_model({'family': family}, classes)
    stats = profile_model(model)
    assert 0 < stats['parameters'] <= 6_000_000
    assert 0 < stats['flops'] < 1_000_000_000
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, classes)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_profiler_rejects_over_budget_and_restores_mode():
    from src.models import build_model, profile_model
    model = build_model({'family': 'compactmix', 'widths': [64, 192, 512]}, 100)
    model.train()
    with pytest.raises(ValueError, match='budget'):
        profile_model(model)
    assert model.training


@pytest.mark.parametrize('options', [
    {'attention': 'none', 'norm': 'gn', 'activation': 'relu'},
    {'attention': 'eca', 'norm': 'ln', 'activation': 'gelu', 'pool': 'avgmax'},
    {'stem': 'stacked', 'context_kernel': 7, 'depths': [1, 1, 1]},
])
def test_searchable_blocks_work(options):
    from src.models import build_model, profile_model
    model = build_model({'family': 'compactmix', **options}, 10)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert profile_model(model)['parameters'] > 0
