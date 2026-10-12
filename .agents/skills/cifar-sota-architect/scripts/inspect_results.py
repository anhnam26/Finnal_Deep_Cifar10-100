"""Read real experiment scores; optionally include flagged technical checks."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.python_policy import ensure_selected_python
ensure_selected_python()

if __name__ == '__main__':
    from src.reporting import collect_results, score
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-smoke', action='store_true')
    args = parser.parse_args()
    records = collect_results([ROOT / 'checkpoints', ROOT / 'experiments' / 'smoke', ROOT / 'experiments' / 'verification'])
    if not args.include_smoke:
        records = [r for r in records if not r['synthetic'] and not r['subset']]
    for r in records:
        print(f"{r['experiment_id']} | {r['dataset']} | val={score(r['best_val_accuracy']) or 'not used'} | test={score(r['test_accuracy']) or 'not evaluated'} | synthetic={r['synthetic']} subset={r['subset']}")
    if not records:
        print('No completed full-data experiments. Accuracy targets remain unverified.')
