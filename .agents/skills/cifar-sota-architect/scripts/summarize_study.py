"""Read the project SQLite study without running new trials."""
import argparse
from pathlib import Path
import site
import sys

ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(str(ROOT / '.deps'))
sys.path.insert(0, str(ROOT / '.deps'))

if __name__ == '__main__':
    import optuna
    parser = argparse.ArgumentParser()
    parser.add_argument('--study')
    args = parser.parse_args()
    database = ROOT / 'experiments' / 'optuna.db'
    if not database.exists():
        print('No project Optuna database yet.')
        raise SystemExit(0)
    storage = 'sqlite:///' + database.as_posix()
    names = [args.study] if args.study else optuna.get_all_study_names(storage)
    for name in names:
        study = optuna.load_study(study_name=name, storage=storage)
        print(name, 'directions=', [d.name for d in study.directions])
        for t in study.trials:
            print(f'trial={t.number} state={t.state.name} values={t.values} params={t.params} failure={t.user_attrs.get("failure_reason", "")}')
