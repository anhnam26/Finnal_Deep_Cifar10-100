import copy
from pathlib import Path
import pytest
import torch


def tiny_config(tmp_path, optimizer='sgd'):
    from src.config import default_config
    cfg = default_config('cifar10')
    cfg.update(smoke=True, epochs=3, workers=0, batch_size=8,
               run_dir=str(tmp_path), device='cpu', tensorboard=False,
               optimizer=optimizer, lr=.001 if optimizer == 'adam' else .1)
    cfg['model'].update(widths=[16, 32, 64], depths=[1, 1, 1])
    return cfg


@pytest.mark.parametrize('optimizer', ['sgd', 'adam'])
def test_epoch_resume_matches_uninterrupted_training(tmp_path, optimizer):
    from src.engine import run_experiment, load_checkpoint
    cfg = tiny_config(tmp_path / 'resume', optimizer)
    partial = run_experiment(cfg, stop_after_epochs=1)
    assert partial['status'] == 'interrupted'
    ckpt = load_checkpoint(Path(cfg['run_dir']) / 'last.pt')
    assert ckpt['epoch'] == 1
    assert len(list(Path(cfg['run_dir']).glob('*.pt'))) == 2
    resumed = run_experiment(cfg)
    other = copy.deepcopy(cfg)
    other['run_dir'] = str(tmp_path / 'whole')
    full = run_experiment(other)
    assert resumed['history'] == full['history']  # history excludes wall-clock timing.
    a = load_checkpoint(Path(cfg['run_dir']) / 'last.pt')
    b = load_checkpoint(Path(other['run_dir']) / 'last.pt')
    for key in ('model', 'ema'):
        for name in a[key]:
            assert torch.equal(a[key][name], b[key][name]), name
    assert a['global_step'] == b['global_step'] == 12
    for parameter_id, state in a['optimizer']['state'].items():
        for key, value in state.items():
            other_value = b['optimizer']['state'][parameter_id][key]
            assert torch.equal(value, other_value) if torch.is_tensor(value) else value == other_value


def test_best_is_not_replaced_by_worse_validation(tmp_path, monkeypatch):
    from src import engine
    scores = iter([80., 70., 60., 50., 40., 30.])
    monkeypatch.setattr(engine, 'evaluate', lambda *a, **kw: {'loss': 1., 'accuracy': next(scores)})
    cfg = tiny_config(tmp_path)
    engine.run_experiment(cfg, stop_after_epochs=1)
    best = (tmp_path / 'best.pt').read_bytes()
    result = engine.run_experiment(cfg)
    assert (tmp_path / 'best.pt').read_bytes() == best
    assert result['best_val_accuracy'] == 80.
    assert result['best_epoch'] == 1
    assert {p.name for p in tmp_path.glob('*.pt')} == {'last.pt', 'best.pt'}


def test_incompatible_resume_does_not_overwrite_checkpoint(tmp_path):
    from src.engine import run_experiment
    cfg = tiny_config(tmp_path)
    run_experiment(cfg, stop_after_epochs=1)
    original = (tmp_path / 'last.pt').read_bytes()
    cfg['lr'] = .15
    with pytest.raises(ValueError, match='configuration'):
        run_experiment(cfg)
    assert (tmp_path / 'last.pt').read_bytes() == original


def test_missing_best_is_recovered_from_last(tmp_path):
    from src.engine import run_experiment, load_checkpoint
    cfg = tiny_config(tmp_path)
    run_experiment(cfg, stop_after_epochs=1)
    expected = load_checkpoint(tmp_path / 'best.pt')
    (tmp_path / 'best.pt').unlink()
    run_experiment(cfg, stop_after_epochs=1)
    recovered = load_checkpoint(tmp_path / 'best.pt')
    assert recovered['accuracy'] >= expected['accuracy']


def test_split_is_stratified_reproducible_and_disjoint():
    from src.datasets import stratified_split
    targets = [i % 10 for i in range(1000)]
    train, val = stratified_split(targets, 10, 2026)
    assert len(train) == 900 and len(val) == 100
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(1000))
    assert all(sum(targets[i] == c for i in val) == 10 for c in range(10))
    assert (train, val) == stratified_split(targets, 10, 2026)


def test_selection_never_opens_official_test(tmp_path, monkeypatch):
    from src.config import default_config
    from src.datasets import prepare_data
    from torchvision import datasets
    class TrainOnly:
        def __init__(self, root, train, download):
            assert train is True
            self.targets = [i % 100 for i in range(1000)]
    monkeypatch.setattr(datasets, 'CIFAR100', TrainOnly)
    cfg = default_config('cifar100')
    source, split = prepare_data(cfg)
    assert len(split['validation']) == 100


def test_failed_atomic_write_preserves_last(tmp_path, monkeypatch):
    from src.engine import atomic_torch, load_checkpoint
    path = tmp_path / 'last.pt'
    atomic_torch(path, {'epoch': 7})
    original = path.read_bytes()
    def failed(*a, **kw):
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', failed)
    with pytest.raises(OSError):
        atomic_torch(path, {'epoch': 8})
    assert path.read_bytes() == original
    assert not path.with_suffix('.pt.tmp').exists()


def test_incomplete_epoch_does_not_replace_last(tmp_path, monkeypatch):
    from src import engine
    cfg = tiny_config(tmp_path)
    engine.run_experiment(cfg, stop_after_epochs=1)
    original = (tmp_path / 'last.pt').read_bytes()
    monkeypatch.setattr(engine, 'train_epoch', lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        engine.run_experiment(cfg)
    assert (tmp_path / 'last.pt').read_bytes() == original
