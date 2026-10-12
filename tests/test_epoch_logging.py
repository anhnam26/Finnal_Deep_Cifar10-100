from pathlib import Path
import subprocess
import yaml
import pytest


@pytest.fixture
def config(tmp_path):
    from src.config import default_config
    cfg = default_config('cifar10')
    cfg.update(smoke=True, epochs=2, batch_size=8, workers=0, device='cpu',
               tensorboard=False, run_dir=str(tmp_path / 'run'))
    cfg['model'].update(widths=[16, 32, 64], depths=[1, 1, 1])
    return cfg


def test_console_and_file_have_one_line_per_completed_epoch(config, capsys):
    from src.engine import run_experiment
    run_experiment(config)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert all('EPOCH' in line and 'cifar10' in line and 'saved last.pt' in line for line in lines)
    assert '1/2' in lines[0] and '2/2' in lines[1]
    assert (Path(config['run_dir']) / 'training.log').read_text().splitlines() == lines


def test_interrupted_validation_emits_no_log_or_checkpoint(config, monkeypatch, capsys):
    from src import engine
    config['tensorboard'] = True
    def interrupt(*args, **kwargs):
        assert capsys.readouterr().out == ''
        raise KeyboardInterrupt
    monkeypatch.setattr(engine, 'evaluate', interrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.run_experiment(config)
    directory = Path(config['run_dir'])
    assert not list(directory.glob('*.pt'))
    assert not (directory / 'metrics.csv').exists()
    assert not (directory / 'tensorboard').exists()
    assert (directory / 'training.log').read_text() == ''


def test_missing_best_is_not_written_before_the_next_epoch(config, monkeypatch, capsys):
    from src import engine
    engine.run_experiment(config, stop_after_epochs=1)
    capsys.readouterr()
    directory = Path(config['run_dir'])
    (directory / 'best.pt').unlink()
    original = engine.train_epoch
    def check(*args, **kwargs):
        assert not (directory / 'best.pt').exists()
        assert engine.load_checkpoint(directory / 'last.pt')['epoch'] == 1
        assert capsys.readouterr().out == ''
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, 'train_epoch', check)
    engine.run_experiment(config)
    assert (directory / 'best.pt').exists()
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_completed_run_is_skipped_without_an_extra_log(config, capsys):
    from src.engine import run_experiment
    run_experiment(config)
    capsys.readouterr()
    run_experiment(config)
    assert capsys.readouterr().out == ''


def test_failed_best_write_does_not_advance_last(config, monkeypatch, capsys):
    from src import engine
    engine.run_experiment(config, stop_after_epochs=1)
    capsys.readouterr()
    directory = Path(config['run_dir'])
    previous_last = (directory / 'last.pt').read_bytes()
    monkeypatch.setattr(engine, 'evaluate', lambda *a, **kw: {'loss': 1., 'accuracy': 100.})
    original = engine.atomic_torch
    def fail_best(path, payload):
        if Path(path).name == 'best.pt':
            raise OSError('disk full')
        return original(path, payload)
    monkeypatch.setattr(engine, 'atomic_torch', fail_best)
    with pytest.raises(OSError, match='disk full'):
        engine.run_experiment(config)
    assert (directory / 'last.pt').read_bytes() == previous_last
    assert capsys.readouterr().out == ''


def test_training_cli_prints_only_completed_epochs(tmp_path):
    from src.python_policy import SELECTED_PYTHON
    path = tmp_path / 'tiny.yaml'
    path.write_text(yaml.safe_dump({'model': {'family': 'compactmix',
                    'widths': [16, 32, 64], 'depths': [1, 1, 1]}}))
    result = subprocess.run([str(SELECTED_PYTHON), '-u', 'train.py', '--smoke',
                             '--dataset', 'cifar10', '--epochs', '2', '--device', 'cpu',
                             '--no-tensorboard', '--config', str(path),
                             '--output-root', str(tmp_path / 'jobs')],
                            capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    assert all('EPOCH' in line for line in lines)
