# CIFAR research and resumable training

Authority: `Final exam.docx`, the attached project-local research instructions,
and the user's request to build now and leave long training to them.

## Design

Keep one public entry point, `python train.py`, using the user's selected
Microsoft Store Python 3.13 and existing CUDA PyTorch. Missing dependencies go
into repository `.deps`, without creating or downloading a Python environment.
Train CIFAR-10 followed by CIFAR-100, from scratch. Rebuild
the existing CompactMixNet idea as configurable CompactMix-L: CIFAR stem,
split depthwise branches, grouped spatial mixing, channel gate and residual
blocks. Count inference operations using two FLOPs per multiply-add, including
normalization/activation/pooling/residual operations. Enforce <=6M learnable
parameters and <1G estimated FLOPs for exam models before training.

Use stratified 45k/5k training/validation splits. Model selection and Optuna see
validation only. A frozen validation winner is retrained on all 50k training
images for its frozen number of epochs before one final test evaluation.

Atomic epoch checkpoints hold weights, EMA, optimizer, AMP scaler, scheduler
position, RNG, split/config/code identities and metric history. Exactly one
last.pt and one best.pt per experiment; no rotations or numbered checkpoints.
Ctrl+C requests a stop at an epoch boundary; force termination replays an
unfinished epoch. Resume rejects changed experiment settings, environment or
training source instead of silently discarding an existing run. Logs include
batch progress, epoch metrics, ETA, VRAM and checkpoint paths.

## Work units and verification

1. Local dependencies: reuse the selected interpreter, before/after
   inventories, local MCP launcher. Verify CUDA forward/backward and imports.
2. Model: configurable custom model, WRN, PreActResNet and CIFAR ConvNeXt
   baselines. Verify output dimensions, gradients and counted budgets.
3. Training: data split, transforms, recipes, atomic best/last, automatic queue,
   strict resume and logs. Verify uninterrupted/resumed CPU equivalence, best
   retention, interruption behavior and a short real-data GPU run.
4. Research: SQLite Optuna, pruning/persistent interrupted trial recovery,
   SGD/Adam LR matrix, frozen retraining/evaluation, result CSV, leaderboard
   and PDF. Verify tiny objective and PDF generation without test leakage.
5. Project skill and handoff: local skill/references/helpers, AGENTS.md and
   Vietnamese README with exact commands. Validate skill and full test suite.

Create a Conventional Commit on the current branch after each validated unit.
Preserve unrelated deleted documentation/tests until replaced as part of the
authorized rebuild; never stage unrelated deletions. No pushes, global installs
or global Codex configuration changes. Accuracy targets are unverified goals.

User steering: creation of `.venv` was cancelled. Its deletion was requested
explicitly but automatic command review rejected deletion with `blocked by
policy`. Do not use that partial directory. Continue with the existing Python.
