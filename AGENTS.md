# Repository instructions

- Keep two runtime entry points: `train.py` and `evaluate.py`. Model blocks,
  forward passes, data and the complete training workflow belong in `train.py`.
- Use only the existing Microsoft Store Python 3.13 base installation at
  `%LOCALAPPDATA%\Microsoft\WindowsApps\python3.13.exe` or its verified resolved
  executable. Never create/use another interpreter, virtual environment or
  alternate package directory, and do not change the IDE interpreter. Install
  missing packages directly with that executable and `-m pip`; preserve CUDA.
- Model L must have <=6M learnable parameters and <1G inference FLOPs for one
  RGB32x32 image. Report the FLOP convention. Never invent accuracy or confuse
  validation, subset/synthetic checks and official test results.
- Select with validation, freeze decisions, retrain on all training data and
  evaluate final test once. Historical 97.18%/82.98% test results belong to
  45k-trained selection checkpoints.
- Never automatically start long training. Reuse `datasets/`; keep local
  checkpoints in `checkpoints/` and measured shareable evidence in `results/`.
- Retain only `best.pt` and `last.pt` per run. Update checkpoints/metrics and
  emit one normal summary only after a completed train/validation epoch and
  successful writes. No batch/startup/resume chatter during normal training.
- Preserve exact optimizer/EMA/AMP/RNG resume. Reject changed source/config/
  runtime. Legacy compatibility is explicitly pinned and tested; do not bypass
  it or delete checkpoints to silence a mismatch.
- Keep datasets, weights, secrets and machine-specific config out of Git.
  Do not modify global tools/settings or unrelated processes.
- Validate each meaningful unit, review status/diff, stage only related files
  and immediately create a local Conventional Commit using the installed
  `git-commit` skill. Report hashes. Never automatically push, change visibility,
  amend, reset, rebase, force-push or merge main/master.
