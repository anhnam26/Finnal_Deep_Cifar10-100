# Matched CIFAR-10 architecture confirmation

Project root: `D:\DeepLearning\Finnal`.

## Scope
Run the missing150epoch CIFAR-10 comparisons for `no_fusion` and `fusion`, sequentially. Both CIFAR-100 architecture runs already completed; they are not requeued. The12run optimizer/LR matrix remains separate and is not started here.

## Frozen settings
New configs under `D:\DeepLearning\Finnal\configs\candidate_cifar10\grid` copy the completed CIFAR-100 screening recipe; only dataset/output directory change. Between the two CIFAR-10 configs only variant/output differ:

- SGD, initialLR0.1, momentum0.9, Nesterov, weightdecay5e-4.
-150epochs,5epoch warmup, per-step cosine to1e-5.
- batch128/workers0, seed42/split2026, stratified45k train/5k validation.
- Croppadding4/flip, CE without smoothing or CutMix; existing CUDA AMP4096initialscale with recorded skips and persistentoverflow guard.
- Same StageFusionNet backbone and10class head. Source fingerprint4861727ff906658c933ebc551c2786ed6a3172e7cb3bf6ae551490cd0d84bff6 remains unchanged; no executable source edits.

The manifest references existing completed full2epoch CIFAR-10 smoke configs under `D:\DeepLearning\Finnal\configs\candidate\smoke`. Their settings match except epoch count/warmup/output. Full45000/5000 sample counts, source hash and complete validation exports were checked before reuse. The queue will reuse these artifacts rather than retrain smoke or alter old outputs.

## Launch and monitor
Require no active project trainer/queue or global lock, RAM>=1536MiB/free system commit>=5120MiB and disk>=12GiB before launch. Existing smoke timing gives a conservative4.01hours for300epochs including25%reserve, not a runtime promise. Thermals/external applications can change speed.

```powershell
python -B -u D:\DeepLearning\Finnal\study.py --manifest D:\DeepLearning\Finnal\configs\candidate_cifar10\manifest.json
powershell -NoProfile -ExecutionPolicy Bypass -File D:\DeepLearning\Finnal\watch_training.ps1 -StudyRoot D:\DeepLearning\Finnal\outputs\candidate_cifar10 -ManifestPath D:\DeepLearning\Finnal\configs\candidate_cifar10\manifest.json -Details
```

Use explicit paths above; the default viewer still points to the completed CIFAR-100 study. Logs/status/checkpoints/validation exports and provisionalPDF for the new runs are under `D:\DeepLearning\Finnal\outputs\candidate_cifar10`. The second model starts automatically after the first finishes and exports validation. Closing the viewer does not stop training. Keep executable Python frozen during training; do not start a duplicate queue. `STOP_AFTER_RUN` in the new output root requests a safe between-run stop.

## Interpretation
Completed CIFAR-100 log bestvalidation: no_fusion75.86%, fusion76.54%; not95%. Fusion's single-seed improvement is not established as robust. CIFAR-10 confirmation is a new measurement;98% remains aspirational, not promised. No official-test access occurs in the queue. Choose later architecture/recipe on validation, preserve checkpoint/config/history, then finalize SGD/Adam/LR/seeds/frozenfinal training and official-test evaluation separately.