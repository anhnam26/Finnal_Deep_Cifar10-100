# Lightweight CIFAR final — StageFusionNet

Project root: `D:\DeepLearning\Finnal`. Follow `D:\DeepLearning\Finnal\Final exam.docx`: <=6M learnable parameters, <1 GFLOP, both CIFAR datasets, SGD/Adam with several initial LRs, PDF and oral defense. **98% CIFAR-10 / 95% CIFAR-100 validation are goals, not results or promises.** Architecture non-duplication must be confirmed with the instructor.

## Architecture and protocol

`D:\DeepLearning\Finnal\models\stagefusion.py`: stride1 3x3 stem; 32/16/8 stages with widths40/104/240, depths2/3/4; preactivated dense residual blocks. Each pooled stage descriptor is projected to240 and LayerNorm-normalized. The fusion head adds two image-gated early descriptors to the final descriptor. `no_fusion` removes only early descriptors/gates. Established residual/GAP/gating ideas are acknowledged; this is not a claim of research novelty.

One backbone for both datasets, classifier10/100. Training from scratch, official data only; no distillation, ensemble or TTA. Fixed stratified45k/5k split2026, initialization seed42. Test is not used by training/queue or selection. BT1 is retained unchanged as a classroom template, not copied wholesale; weighted sample metrics and separate validation dataset objects avoid issues in the sample scripts.

Existing Python3.13 / PyTorch2.7.1+cu118 environment; no new environment/dependency. Run commands below in the root or use absolute paths.

```powershell
python -B -m pytest -q
powershell -NoProfile -ExecutionPolicy Bypass -File D:\DeepLearning\Finnal\tests\test_watch_training.ps1
python -B D:\DeepLearning\Finnal\profile_model.py --variant fusion --classes 100
python -B D:\DeepLearning\Finnal\profile_model.py --variant no_fusion --classes 10
```

Profiler input[1,3,32,32], conv/linear2 FLOPs/MAC, BN unfused4/element, LayerNorm8/element+3/group, sigmoid4, residual/fusion add/multiply1, GAP and biases counted. Flatten/memory ignored. Scalar transcendental conventions are estimates, not hardware latency. Never divide MACs by2 to derive FLOPs.

## Experiments

`D:\DeepLearning\Finnal\configs\candidate\manifest.json` screens no_fusion/fusion on CIFAR-100 first (2x150epochs). Four full2epoch real-data smoke runs cover both heads/variants. Initial recipe: batch128, workers0, SGD0.1 momentum0.9 Nesterov, WD5e-4, crop/flip, plain CE, per-step cosine with5epoch warmup (no warmup in smoke). CUDA AMP initial scale4096; skips logged and >1%/epoch blocked.

```powershell
python -B -u D:\DeepLearning\Finnal\study.py --manifest D:\DeepLearning\Finnal\configs\candidate\manifest.json --stage smoke
python -B -u D:\DeepLearning\Finnal\study.py --manifest D:\DeepLearning\Finnal\configs\candidate\manifest.json
powershell -NoProfile -ExecutionPolicy Bypass -File D:\DeepLearning\Finnal\watch_training.ps1
```

Do not start another queue when one is alive. Source fingerprint guards resume; keep executable Python frozen during training. Global/per-study locks, fail-stop and `STOP_AFTER_RUN` are retained. Stale locks require checking both queue and trainer PIDs, not blindly deleting them. Smoke budget includes checkpoint/validation and25% reserve; thermals/host memory can change runtime. Keep unrelated programs/processes untouched.

`D:\DeepLearning\Finnal\configs\candidate_matrix\manifest.json` is a **prepared, not launched**12run template: both datasets, SGD LR0.05/0.1/0.15, Adam LR0.0003/0.001/0.003 (not AdamW),150epochs. Adam betas0.9/0.999 eps1e-8 WD1e-4; optimizer-specific WD difference is disclosed. Finalize architecture/recipe and regenerate into a new directory before matrix launch; do not run all configs before selection.

Generator refuses config/output reuse:
```powershell
python -B D:\DeepLearning\Finnal\prepare_study.py --mode matrix --variant fusion --destination D:\DeepLearning\Finnal\configs\selected_matrix --output-root D:\DeepLearning\Finnal\outputs\selected_matrix
```

No fusion superiority is assumed. Compare architecture under identical recipe, recipe changes under identical architecture. Augmentation/EMA experiments, additional seeds, frozen final50k training and official test follow only after validation evidence.

## Resume, exports, PDF

```powershell
python -B D:\DeepLearning\Finnal\train.py --config <absolute-config-path> --resume <absolute-last.pt-path>
python -B D:\DeepLearning\Finnal\evaluate_full.py --checkpoint <absolute-best.pt-path> --split validation --output-dir <new-absolute-export-directory>
python -B D:\DeepLearning\Finnal\reporting.py --study-root D:\DeepLearning\Finnal\outputs\candidate
```

Checkpoint stores model/optimizer/scaler/RNG/loader RNG/config/split/source. Resume only original output and same schedule. Validation exports include5k predictions, per-class/confusion/calibration metrics, learning curves, model-only weights and PDF. Aggregate PDF is **provisional**, not final submission; smoke/unfinished runs are excluded from architecture ranking. Matched fusion ablations use paired prediction comparisons. Official test requires `--split test --confirm-final-test` after freezing decisions; outputs cannot be overwritten. Never tune using test results.

## Retired experiments

Old model code, launchers, configs, ignored weights/optimizer states/logs and source snapshots were retired after inventory. Historical best validation (single seed): baseline15095.50%, baseline30095.46%, preactivation15095.22%; different schedules are not matched evidence. Old ignored weights cannot be recovered from Git. Source history remains in Git; detailed internal inventory is ignored in `D:\DeepLearning\Finnal\plan_action`.

Preserved: datasets, KnowLedge, BT1, exam, download verification, reusable data/train/evaluation/reporting and Git history. The pre-existing `.gitignore` changes were not staged. No automatic push. Long-term exam/report protocol: `D:\DeepLearning\Finnal\docs\FINAL_PLAN.md`. Verified profiles/smoke timings and the current memory gate: `D:\DeepLearning\Finnal\docs\CANDIDATE_VALIDATION.md`.