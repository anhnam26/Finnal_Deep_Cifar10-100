import copy
import json
from pathlib import Path
import pytest
import torch


def research_config(tmp_path):
    from src.config import default_config
    config = default_config('cifar10')
    config.update(smoke=True, epochs=2, workers=0, batch_size=8,
                  tensorboard=False, device='cpu', run_dir=str(tmp_path))
    config['model'].update(widths=[16, 32, 64], depths=[1, 1, 1])
    return config


def test_optuna_trial_resumes_without_creating_another_trial(tmp_path):
    from src.search import search
    config = research_config(tmp_path / 'trial')
    db = tmp_path / 'optuna.db'
    study = search(config, trials=1, study_name='resume', storage_path=db, stop_after_epochs=1)
    assert len(study.trials) == 1
    assert study.trials[0].state.name == 'RUNNING'
    study = search(config, trials=1, study_name='resume', storage_path=db)
    assert len(study.trials) == 1
    assert study.trials[0].state.name == 'COMPLETE'
    trial = study.trials[0]
    assert 0 <= trial.value <= 100
    assert trial.user_attrs['parameters'] < 6_000_000
    assert trial.user_attrs['test_accuracy'] is None
    assert trial.user_attrs['checkpoint_path'].endswith('best.pt')


def test_finalization_rejects_synthetic_evidence(tmp_path):
    from src.engine import run_experiment
    from src.finalize import finalize
    cfg = research_config(tmp_path)
    run_experiment(cfg)
    with pytest.raises(ValueError, match='synthetic|subset'):
        finalize(tmp_path)


def test_final_retraining_and_test_are_cached(tmp_path, monkeypatch):
    from src.config import default_config
    from src.engine import run_experiment
    from src.finalize import finalize, evaluate_final
    from torchvision import datasets
    calls = []
    class SmallCifar:
        def __init__(self, root, train, download):
            calls.append(train)
            self.targets = [i % 10 for i in range(200 if train else 20)]
            self.fake = datasets.FakeData(size=len(self.targets), image_size=(3,32,32), num_classes=10)
        def __len__(self):
            return len(self.targets)
        def __getitem__(self, i):
            return self.fake[i][0], self.targets[i]
    monkeypatch.setattr(datasets, 'CIFAR10', SmallCifar)
    cfg = default_config('cifar10')
    cfg.update(epochs=1, batch_size=20, workers=0, device='cpu', tensorboard=False,
               run_dir=str(tmp_path / 'selection'))
    cfg['model'].update(widths=[16,32,64], depths=[1,1,1])
    run_experiment(cfg)
    assert all(calls)  # Official test was never opened during model selection.
    final = finalize(Path(cfg['run_dir']))
    assert final['phase'] == 'final'
    assert final['test_accuracy'] is not None
    assert calls.count(False) == 1
    split = json.loads((tmp_path / 'selection_final' / 'split.json').read_text())
    assert len(split['training']) == 200 and split['validation'] == []
    evaluate_final(tmp_path / 'selection_final')
    assert calls.count(False) == 1


def test_pdf_and_csv_do_not_turn_missing_test_into_zero(tmp_path):
    from src.reporting import build_reports
    roots = tmp_path / 'runs'
    cfg = research_config(roots / 'smoke')
    from src.engine import run_experiment
    run_experiment(cfg)
    output = tmp_path / 'report'
    build_reports(roots=[roots], report_root=output, result_path=tmp_path / 'results.csv')
    assert (output / 'final_report.pdf').read_bytes().startswith(b'%PDF')
    assert 'synthetic' in (output / 'leaderboard.md').read_text().lower()
    import csv
    rows = list(csv.DictReader((tmp_path / 'results.csv').open()))
    assert rows[0]['test_accuracy'] == ''
