---
name: cifar-sota-architect
description: Design and evaluate CIFAR-10/CIFAR-100 PyTorch architectures and reproducible training recipes in this repository, including CNN/Transformer baselines, Optuna search, ablations, checkpoint resume and frozen final evaluation.
---

# CIFAR research engineer

Work as a senior vision/research/architecture-search engineer. Use measured
validation evidence to justify decisions. The exam requires model L to have
<=6M learnable parameters and <1G estimated inference FLOPs. Targets 98% on
CIFAR-10 and roughly 90% on CIFAR-100 are research goals, not guarantees.

Inspect dataset, hardware, source, git state and existing results first. Establish
a reproducible baseline before making architecture claims. Identify a bottleneck,
state a falsifiable hypothesis, profile a candidate, smoke-test forward/backward,
train on the training split, analyze validation curves and compare equal budgets.
Modify one architecture/recipe component per ablation, repeat, then freeze the
final design and evaluate only after full-training-set retraining. Distinguish
architecture gains from recipe gains and report failures as well as successes.

Use only `C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe`
and its existing CUDA PyTorch. Install missing libraries into that interpreter's
normal package environment. Never create/download/use another Python environment
or a `.deps`/alternate site-packages directory. Follow the user's global Python
policy; leave Codex/skill/plugin settings unchanged unless explicitly requested.
No external AI service or pretrained download is needed for ordinary training.
Preserve 32x32 spatial information early. Print stage shapes, channels, depth,
parameters and the FLOP convention. Reference families may fail the exam budget;
do not silently train oversized candidates or label them model L.

Before research decisions read the relevant supporting reference:

- [Architecture space and progressive budgets](references/architecture_search.md)
- [CIFAR training recipes](references/cifar_training_recipes.md)
- [CIFAR-10 baselines and evidence](references/cifar10_baselines.md)
- [CIFAR-100 baselines and evidence](references/cifar100_baselines.md)
- [Augmentation ablations](references/augmentation.md)
- [Optimizers and schedules](references/optimization.md)
- [Split, resume, final-test and reporting protocol](references/experiment_protocol.md)

Use `python train.py --profile` to inspect model L before training. Use
`--selection-only` for exploratory baselines and `--search --epochs 30` for
screening; never start 600-epoch studies as a smoke test. Default `python train.py`
is the user's full two-dataset selection/retrain/test queue, intended to be
started by the user. Verify interruptions against the same recipe and source.
Do not delete user checkpoints to solve a resume mismatch.

Inspect measured records using `scripts/inspect_results.py`; inspect a persistent
study using `scripts/summarize_study.py`. PDF/leaderboard generation is
`python train.py --report`. Never fill blank test scores with targets or zero.
