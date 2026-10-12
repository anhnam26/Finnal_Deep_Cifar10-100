# Verification on October 13, 2026

## Current epoch-only logging and checkpoint timing

Current suite: **48 passed**. Console and training.log have exactly one line
per completed training/validation epoch, emitted after checkpoint/metrics writes.
No batch, startup, queue, resume or automatic-report progress. A validation
interruption emits no epoch log, metrics or checkpoint. Missing-best recovery
waits for the next epoch. A failed best write does not advance durable last.
SGD/Adam resume equivalence passes with TensorBoard both enabled and disabled.

Short synthetic CUDA runs on both datasets stopped/resumed over two epochs and
printed only completed epoch summaries; these are execution checks, not accuracy
evidence. Model/recipe budgets and the selected Python environment are unchanged.

The user explicitly requested deletion of old training history. The validated
targets were checkpoints/, experiments/, outputs/ if present and reports/generated/.
Native PowerShell deletion was rejected by automatic command review with
`blocked by policy`; the existing histories remain. No alternate deletion
method was used. The dataset, source and Git history are not reset. A new
`--run-name epoch-only` can start immediately while cleanup remains pending.

## Current single-environment policy

After the user's all-project Python instruction, the project uses only the
existing Microsoft Store Python 3.13 and its own package environment. The
global AGENTS.md and project instructions match. New runtime guards reject a
different executable, version or venv prefix. No `.deps` import hooks remain.

The shared-environment migration suite had **40 passed** (the prior 36 plus
four Python policy cases); the newer epoch-only verification is recorded above.
The setup script was run successfully and idempotently in the selected Python.
Thirteen missing distributions were installed in its normal user-site; no
existing packages changed version or were removed. torch 2.7.1+cu118,
torchvision 0.22.1+cu118 and NumPy 2.2.6 stayed unchanged; checked package imports
all resolve within that environment (`shared-python-audit.json`).

One synthetic CUDA epoch per dataset completed using normal global imports,
with valid last/best checkpoints and no AMP skips. This is a technical smoke
check, not accuracy evidence. MCP launches with the exact selected alias.
`.venv` is absent; native deletion of old `.deps` was rejected by automatic
review, so its unused directory cleanup remains pending.

## Prior research pipeline verification

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
Source compilation and git whitespace checks passed. Earlier package setup was
rerun idempotently without downloading an interpreter, creating a venv, or
replacing CUDA PyTorch. That earlier isolation audit is historical; the current
policy explicitly authorizes package installation in the existing shared Python
and the global AGENTS.md change, as recorded above and in ENVIRONMENT.md.

Final filesystem check: `.venv` absent. The rejected deletion command is not
represented as a successful agent action. The default full user run has not
been started. Generated PDF is a protocol/architecture report until actual
full-data optimizer/LR comparisons and frozen test results are available.
