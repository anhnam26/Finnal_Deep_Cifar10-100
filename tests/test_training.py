import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torchvision import transforms

import train


@pytest.fixture(autouse=True)
def reset_stop():
    train.STOP_REQUESTED = False
    torch.set_num_threads(2)
    yield
    train.STOP_REQUESTED = False


@pytest.mark.parametrize('variant', ['dualpath', 'localpath'])
@pytest.mark.parametrize('classes', [10, 100])
def test_real_model_budget_and_shape(variant, classes):
    model = train.CompactMixNet(classes, variant).eval()
    profile = train.profile_model(model)
    assert 5_000_000 < profile['parameters'] <= 6_000_000
    assert 800_000_000 < profile['flops'] < 1_000_000_000
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, classes)


@pytest.mark.parametrize('classes', [10, 100])
def test_split_is_stratified_disjoint_and_repeatable(classes):
    targets = np.repeat(np.arange(classes), 50_000 // classes)
    training, validation = train.stratified_split(targets, classes, 2026)
    assert len(training) == 45_000 and len(validation) == 5000
    assert not set(training) & set(validation)
    assert set(training) | set(validation) == set(range(50_000))
    assert np.all(np.bincount(targets[validation]) == 5000 // classes)
    assert (training, validation) == train.stratified_split(targets, classes, 2026)


def test_resume_sampler_keeps_original_batch_order():
    full = list(train.RemainingBatches(33, 8, 42, 3))
    resumed = list(train.RemainingBatches(33, 8, 42, 3, 2))
    assert resumed == full[2:]
    assert len(full[-1]) == 1
    assert len(set(sum(full, []))) == 33
    assert list(train.RemainingBatches(33, 8, 42, 4)) != full


def test_sample_rng_preserves_main_rng():
    train.seed_everything(123)
    before = train.capture_rng()
    with train.sample_rng(456):
        first = (random.random(), np.random.rand(), torch.rand(1))
    after = train.capture_rng()
    assert before['python'] == after['python']
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert torch.equal(before['torch'], after['torch'])
    with train.sample_rng(456):
        second = (random.random(), np.random.rand(), torch.rand(1))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_augmented_sample_is_independent_of_fetch_order():
    image = transforms.ToPILImage()(torch.linspace(0, 1, 3 * 32 * 32).reshape(3, 32, 32))
    source = [(image, index % 10) for index in range(20)]
    transform, _ = train.image_transforms('cifar10')
    dataset = train.SeededImages(source, list(range(20)), transform, 42, 3)
    before = dataset[7][0]
    dataset[12]
    dataset[3]
    assert torch.equal(before, dataset[7][0])


def test_windows_workers_do_not_change_augmented_samples():
    image = transforms.ToPILImage()(torch.linspace(0, 1, 3 * 32 * 32).reshape(3, 32, 32))
    source = [(image, index % 10) for index in range(20)]
    transform, _ = train.image_transforms('cifar10')
    dataset = train.SeededImages(source, list(range(20)), transform, 42, 3)
    config = {'workers': 0, 'device': 'cpu', 'seed': 42, 'batch_size': 8}
    serial = list(train.make_loader(dataset, config, 3, 1))
    parallel = list(train.make_loader(dataset, {**config, 'workers': 2}, 3, 1))
    for (first_images, first_labels), (second_images, second_labels) in zip(serial, parallel):
        assert torch.equal(first_images, second_images)
        assert torch.equal(first_labels, second_labels)


def test_checkpoint_rotation_and_corrupt_recovery(tmp_path):
    path = tmp_path / 'last.pt'
    train.atomic_checkpoint(path, {'version': 1, 'step': 1}, rotate=True)
    train.atomic_checkpoint(path, {'version': 1, 'step': 2}, rotate=True)
    assert train.recover_checkpoint(tmp_path)['step'] == 2
    path.write_bytes(b'interrupted or corrupt download')
    assert train.recover_checkpoint(tmp_path)['step'] == 1
    path.unlink()
    assert train.recover_checkpoint(tmp_path)['step'] == 1


def test_lock_rejects_concurrent_training_and_releases(tmp_path):
    path = tmp_path / 'training.lock'
    with train.training_lock(path):
        with pytest.raises(RuntimeError, match='Another trainer'):
            with train.training_lock(path):
                pass
    with train.training_lock(path):
        pass


class TinyNet(nn.Module):
    def __init__(self, num_classes=10, variant='dualpath', drop_path=0.1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8),
                                    nn.SiLU(), nn.Dropout(0.1), nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(), nn.Linear(8, num_classes))

    def forward(self, inputs):
        return self.layers(inputs)


def session_directory(root):
    pointer = json.loads((root / 'latest_session.json').read_text())
    return Path(pointer['directory'])


def job_directory(root):
    session = session_directory(root)
    document = json.loads((session / 'session.json').read_text())
    return session / document['jobs'][0]['name']


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
    else:
        assert first == second


@pytest.mark.parametrize('optimizer', ['sgd', 'adam'])
def test_epoch_resume_replays_unfinished_epoch_and_matches_uninterrupted_training(tmp_path, monkeypatch, optimizer):
    monkeypatch.setattr(train, 'CompactMixNet', TinyNet)
    common = ['--smoke', '--dataset', 'cifar10', '--device', 'cpu', '--workers', '0',
              '--optimizer', optimizer]
    straight, paused = tmp_path / 'straight', tmp_path / 'paused'
    assert train.main(common + ['--new', '--output-root', str(straight)]) == 0
    assert train.main(common + ['--new', '--output-root', str(paused), '--stop-after-steps', '5']) == 0
    halfway = train.recover_checkpoint(job_directory(paused))
    assert halfway['state']['epoch'] == 1
    assert halfway['state']['next_batch'] == 0
    assert halfway['state']['global_step'] == 4
    assert halfway['state']['phase'] == 'train'
    checkpoint_path = job_directory(paused) / 'last.pt'
    previous_bytes = checkpoint_path.read_bytes()
    assert train.main(['--smoke', '--output-root', str(paused), '--stop-after-steps', '1']) == 0
    assert checkpoint_path.read_bytes() == previous_bytes
    assert train.main(['--smoke', '--output-root', str(paused)]) == 0
    first = train.recover_checkpoint(job_directory(straight))
    second = train.recover_checkpoint(job_directory(paused))
    for key in ('model', 'ema', 'optimizer', 'scaler'):
        assert_nested_equal(first[key], second[key])
    assert first['state']['global_step'] == second['state']['global_step'] == 8
    assert second['state']['phase'] == 'test'
    for left, right in zip(first['state']['history'], second['state']['history']):
        assert_nested_equal({key: value for key, value in left.items()
                             if key not in ('train_seconds', 'epoch_seconds')},
                            {key: value for key, value in right.items()
                             if key not in ('train_seconds', 'epoch_seconds')})
    result = json.loads((job_directory(paused) / 'result.json').read_text())
    assert result['synthetic'] and not result['target_met']
    assert sum(map(sum, result['test']['confusion_matrix'])) == 16
    assert len((job_directory(paused) / 'metrics.csv').read_text().splitlines()) == 3


def test_stop_before_first_epoch_creates_no_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'CompactMixNet', TinyNet)
    arguments = ['--smoke', '--dataset', 'cifar10', '--device', 'cpu', '--output-root', str(tmp_path)]
    assert train.main(arguments + ['--stop-after-steps', '2']) == 0
    directory = job_directory(tmp_path)
    assert train.recover_checkpoint(directory) is None
    assert not list(directory.glob('*.pt'))
    assert not (directory / 'metrics.csv').exists()
    assert train.main(arguments) == 0
    checkpoint = train.recover_checkpoint(directory)
    assert checkpoint['state']['epoch'] == 2
    assert checkpoint['state']['global_step'] == 8


def test_only_completed_epochs_write_checkpoints_and_progress_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train, 'CompactMixNet', TinyNet)
    checkpoint_writes = []
    original_checkpoint = train.atomic_checkpoint

    def record_checkpoint(path, payload, rotate=False):
        checkpoint_writes.append((Path(path).name, payload.get('epoch'), payload.get('state')))
        original_checkpoint(path, payload, rotate)

    monkeypatch.setattr(train, 'atomic_checkpoint', record_checkpoint)
    arguments = ['--smoke', '--dataset', 'cifar10', '--device', 'cpu', '--output-root', str(tmp_path)]
    assert train.main(arguments) == 0
    messages = capsys.readouterr().out.splitlines()
    assert len(messages) == 3
    assert 'EPOCH 1/2' in messages[0] and 'checkpoint=last.pt' in messages[0]
    assert 'EPOCH 2/2' in messages[1] and 'checkpoint=last.pt' in messages[1]
    assert 'RESULT cifar10' in messages[2]
    progress = [state for name, _, state in checkpoint_writes if name == 'last.pt']
    assert [state['epoch'] for state in progress] == [1, 2]
    assert all(state['next_batch'] == 0 for state in progress)
    assert all(len(state['history']) == state['epoch'] for state in progress)
    assert all(name in ('best.pt', 'last.pt') for name, _, _ in checkpoint_writes)
    assert all(epoch in (1, 2) for name, epoch, _ in checkpoint_writes if name == 'best.pt')
    saved_bytes = (job_directory(tmp_path) / 'last.pt').read_bytes()
    checkpoint_writes.clear()
    assert train.main(arguments + ['--resume', str(session_directory(tmp_path))]) == 0
    assert not checkpoint_writes
    assert (job_directory(tmp_path) / 'last.pt').read_bytes() == saved_bytes


def test_interrupted_validation_does_not_save_incomplete_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'CompactMixNet', TinyNet)
    original_evaluate = train.evaluate
    evaluations = []

    def interrupted_evaluate(*arguments):
        evaluations.append(True)
        if len(evaluations) == 2:
            train.request_stop(None, None)
        return original_evaluate(*arguments)

    monkeypatch.setattr(train, 'evaluate', interrupted_evaluate)
    arguments = ['--smoke', '--dataset', 'cifar10', '--device', 'cpu', '--output-root', str(tmp_path)]
    assert train.main(arguments) == 0
    assert not list(job_directory(tmp_path).glob('*.pt'))
    monkeypatch.setattr(train, 'evaluate', original_evaluate)
    assert train.main(arguments) == 0
    assert train.recover_checkpoint(job_directory(tmp_path))['state']['epoch'] == 2


def test_both_datasets_complete_and_new_runs_do_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'CompactMixNet', TinyNet)
    arguments = ['--smoke', '--device', 'cpu', '--output-root', str(tmp_path)]
    assert train.main(arguments) == 0
    original = session_directory(tmp_path)
    results = json.loads((original / 'results.json').read_text())
    assert len(results) == 2
    assert all(result['synthetic'] and not result['target_met'] for result in results)
    assert train.main(arguments) == 0
    assert session_directory(tmp_path) != original
    assert (original / 'results.json').exists()


def test_resume_rejects_changed_source_without_touching_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'CompactMixNet', TinyNet)
    arguments = ['--smoke', '--dataset', 'cifar10', '--device', 'cpu', '--output-root', str(tmp_path)]
    assert train.main(arguments + ['--stop-after-steps', '5']) == 0
    path = job_directory(tmp_path) / 'last.pt'
    previous_bytes = path.read_bytes()
    monkeypatch.setattr(train, 'source_hash', lambda: 'different-source')
    assert train.main(arguments) == 1
    assert path.read_bytes() == previous_bytes


def test_matrix_contains_required_optimizers_and_learning_rates(tmp_path):
    args = train.parse_args(['--matrix', '--device', 'cpu'])
    _, session = train.create_session(args, tmp_path)
    assert len(session['jobs']) == 12
    for dataset in ('cifar10', 'cifar100'):
        jobs = [job['config'] for job in session['jobs'] if job['config']['dataset'] == dataset]
        assert {job['lr'] for job in jobs if job['optimizer'] == 'sgd'} == {0.05, 0.1, 0.15}
        assert {job['lr'] for job in jobs if job['optimizer'] == 'adam'} == {0.0003, 0.001, 0.003}


@pytest.mark.parametrize('arguments', [['--epochs', '0'], ['--lr', 'nan'], ['--workers', '-1'],
                                      ['--batch-size', '0'], ['--new', '--resume', 'old']])
def test_invalid_cli_settings_fail(arguments):
    with pytest.raises(SystemExit):
        train.parse_args(arguments)
