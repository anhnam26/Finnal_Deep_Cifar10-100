# Verification on October 13, 2026

`python -X utf8 -m pytest -q`: **36 passed**. Tests cover:

- Custom L / WRN / PreActResNet / CIFAR ConvNeXt forward, gradients and budgets
  for both 10/100 classes; configurable blocks and rejected oversized model.
- Ten timm family adapters: 32x32 forward, output dimensions and profiling.
- Stratified disjoint reproducible split, train-only dataset access in selection.
- Exact CPU SGD and Adam resumed/uninterrupted weights, EMA and optimizer state.
- Best retention on declining accuracy, only two checkpoint names, missing-best
  recovery, failed-write preservation and incomplete-epoch preservation.
- Same augmented images/labels after Windows two-worker restart.
- Persistent Optuna interrupted-trial recovery without creating another trial.
- Frozen full-training-set retraining and cached final evaluation against a
  controlled test fixture; official CIFAR test was not opened by setup.
- CSV blank missing test scores and valid PDF generation.

Real-data CUDA verification used each dataset's training set with only 128
training and 64 validation images, two epochs, batch 128, deterministic FP16
AMP and seed 42. These checks are explicitly flagged subset and excluded from
benchmark rankings. No full-data training or target accuracy is demonstrated.

The command below was invoked three times. It stopped after CIFAR-10 epoch 1,
resumed CIFAR-10 epoch 2 and stopped after CIFAR-100 epoch 1, then skipped the
completed CIFAR-10 run and resumed CIFAR-100 epoch 2.

```powershell
python -u train.py --dataset both --epochs 2 --limit-train 128 --limit-val 64 --batch-size 128 --workers 2 --no-tensorboard --run-name final-gpu-check --output-root experiments/verification --stop-after-epochs 1
```

An uninterrupted run used the same model/recipe/data/seed with workers 0 and
run name whole-gpu-check. Tensor-by-tensor comparison on both datasets showed
raw/EMA weights, optimizer state, AMP scaler and history exactly equal. Each
run directory contains exactly best.pt and last.pt. Peak allocated GPU memory
was approximately 1.84-1.88 GiB; no AMP optimizer skips occurred. This does not
guarantee bitwise equality across different hardware/software environments.

Model L profiles: CIFAR-10 5,248,172 parameters / 842,406,944 FLOPs;
CIFAR-100 5,271,302 / 842,453,114. WRN YAML and ConvNeXt YAML were separately
profiled and comply with the exam budget. The operation convention is explicit
in README/ARCHITECTURE; profiler estimates are separate from measured VRAM.

Project skill validator passed; both skill helpers executed. Optuna MCP STDIO
initialized, listed 26 tools and queried study names successfully. `codex mcp
list` showed the project Optuna server enabled with the selected interpreter.
Source compilation and git whitespace checks passed. Package setup was rerun
idempotently without downloading an interpreter, creating a venv, or replacing
the existing CUDA PyTorch. Global packages/config audit unchanged; observed
external Codex metadata changes are recorded in ENVIRONMENT.md.

Final filesystem check: `.venv` absent. The rejected deletion command is not
represented as a successful agent action. The default full user run has not
been started. Generated PDF is a protocol/architecture report until actual
full-data optimizer/LR comparisons and frozen test results are available.
