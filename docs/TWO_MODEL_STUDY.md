# Two-model study: training and evaluation protocol

Baseline and Multiscale are candidates; neither is assumed better. Both use the same backbone per dataset except the 10/100-class classifier. Originality/non-duplication and FLOPs convention still require instructor confirmation.

## Stages
1. Four real-data smoke runs: two full epochs, 45k train + 5k stratified validation. CIFAR10 SGD .1 and CIFAR100 Adam .001 for both variants, workers0; two matched CIFAR100 Adam repeats workers2. Debug/budget measurements, not model ranking. Compare workers 0/2 before fixing grid workers.
2. Base grid: 2 variants x 2 datasets x 2 optimizers x 3 LRs = 24 runs, 150 epochs each, seed42/split2026, batch128, AMP, warmup5/cosine. SGD LRs .05/.1/.15, momentum .9, Nesterov, WD5e-4; Adam LRs .0003/.001/.003, betas .9/.999, eps1e-8, WD1e-4. Crop padding4/flip, hard CE. Same budgets/splits/recipe across variants. Checkpoint: highest validation top1; ties retain earliest.
3. Recipe trials: up to 8 additional runs at same budget, CutMix alpha1/probability.5, then smoothing .05 only if supported by validation. Separate architecture and recipe effects.
4. Four long development runs up to 300 epochs to fix final schedule using validation.
5. Final: 2 variants x 2 datasets x seeds42/43/44 =12 runs, from scratch on full50k at frozen epoch/recipe. Predetermined final epoch, no test-dependent selection. Test only after freeze. Final architecture L selected from both datasets' validation, not separate winners from test.

Maximum plan ~9600 epochs excluding smoke; actual time based on real measured epoch wall time, plus 20–30% reserve. Queue started in this checkpoint executes only the base grid. No automatic test evaluation or recipe/final decisions.

## Metrics and exports
Top1/error/top5 (%), NLL; perclass and macro/micro/weighted precision/recall/F1 (fractions), support, balanced accuracy (%), raw/row-normalized confusion; summed multiclass Brier, ECE15 equal-width confidence bins (fractions), confidence histogram/reliability; OvR ROC-AUC/AP perclass and macro/weighted/micro where applicable; Wilson95% interval for individual accuracy. Missing-class AUC is null/undefined, never silently zero. AP is average precision, not trapezoidal PR-AUC.

Single-label micro-F1=accuracy fraction; balanced CIFAR mean recall=accuracy. Explain redundancies. Evaluation always hard-label float32 single view; mixed training loss/accuracy are not comparable to deterministic hard-label train metrics. Full-data final has no validation curve.

Store dataset indices, labels, logits/probabilities, top5/confidence and class names. CSV/JSON/NPZ plus PNG/PDF charts. Error examples deterministic high-confidence wrong predictions. C100 full perclass tables/curves data; readable selected and macro/micro curve plots rather than 100 overlapped lines. No tuning after examining final test errors.

Resource metrics: params/MACs/FLOPs explicit 2FLOPs/MAC convention; model-only weights file size (not optimizer checkpoint); training/validation/checkpoint wall time, throughput, allocated/reserved CUDA peaks; batch1 median/p95 latency and batch128 throughput with warmup/sync/precision/device disclosed, no preprocessing/I/O included. Desktop GPU contention must be noted.

AMP stability: GradScaler dynamic scaling can skip a bounded number of overflow updates without nonfinite model weights. Record initial/final scale, skipped updates and fraction; fail-stop if >1% updates in any epoch or nonfinite loss/weights. Real two-epoch smoke initially observed1–2 skipped updates out of704 on baseline, all losses finite and curves improving; a zero-skip prerequisite was too strict and corrected before starting grid. Smoke source fingerprint is preserved; grid uses frozen updated logging/guard source. cuDNN benchmark can choose different kernels across processes, so float32 evaluation may show tiny top1/NLL differences around decision ties; saved indexed logits are authoritative for full metrics.

Compare matched configs for architecture ablation and independently tuned recipes with same search budget separately. Repeated final seeds mean/std (not best-test seed). Paired bootstrap95% CI on same indexed images of preselected checkpoint pairs is sampling uncertainty, not seed variability. No significance claims from multiple exploratory comparisons.

## Safety and reproducibility
Official train45k/val5k indices/split fingerprint fixed. Train loop never loads test. Evaluation requires split selection and test acknowledgement. Full-data runs must finish frozen schedule. Files saved in unique run directory; fail on mismatch/overwrite. Resume restores optimizer/scaler/RNG/loader RNG at completed epoch boundaries; source fingerprints and logs recorded. One active queue via lock, sequential subprocesses, fail-stop on errors, stop-request checked between runs. No stopping other projects.

Report is provisional until grid/recipe/seeds/final test are complete. Never present smoke or unfinished 150-epoch runs as final performance.

## Readable live training monitor (Windows PowerShell)
Open a VS Code PowerShell terminal and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\DeepLearning\Finnal\watch_training.ps1"
```

This separate viewer never starts/stops training or changes Python/config/checkpoints. It displays the grid overview, queue/child PID state, saved epoch metrics, validation exports and stderr; automatically follows the next run. Smoke metrics are excluded from the grid table. `Ctrl+C` stops only the viewer. ExecutionPolicy Bypass applies only to this process, not the machine policy.

The viewer appends `outputs/study_two_models/logs/training_readable.log`; original CSV/JSON/raw logs remain authoritative. It records only while open. Observation timestamps are not original epoch timestamps; initial recent rows and stderr are labeled history. Each reopening starts a new session and may repeat historical context. Train+Val time excludes checkpoint overhead; metrics arrive per epoch, not per batch.

Use `-Once` for one snapshot, `-HistoryRows 10` for more context, or `-NoFileLog` to avoid writing the readable log. `-PollCount 3` limits polling for diagnostics. No global environment or execution policy changes are needed.