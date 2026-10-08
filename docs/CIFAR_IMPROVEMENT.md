# Validation-only CIFAR improvement

Target: **official-test top-1 >97%** on CIFAR-10. This is a goal, not a measured result or guarantee. Instructor approval/architecture non-duplication remains unconfirmed. No external data, pretrained model, distillation, ensemble or TTA.

## Candidate and rationale

`LightNet(10 or 100, "hybrid")` uses the existing dense residual stages at 32/16/8 pixels, followed by four pointwise-depthwise-pointwise (PDP) residual blocks at **8x8**. The latter expand to twice the output width, apply depthwise5x5 context, project, then channel-reweight with SE before adding a projected/identity shortcut. All heads share the backbone. Unlike the old fourth stage, there is no reduction to4x4.

| Stage | Output | Width | Blocks |
|---|---|---|---|
| Stem | 32x32 | 48 | conv3x3/BN/ReLU |
| Dense residual1 | 32x32 | 48 | 2 |
| Dense residual2 | 16x16 | 96 | 3 |
| Dense residual3 | 8x8 | 192 | 3 |
| PDP/SE residual | 8x8 | 384 | 4 |
| GAP + linear | 1 | 384 | head10/100 |

Measured with the repository's unfused inference convention (2 FLOPs/MAC, BN4/element; includes SE/add/mul): C10 **4,850,458 parameters / 952,495,306 FLOPs**, C100 **4,885,108 / 952,564,516**. Both satisfy <=6M/<1G, though FLOPs margin is only ~4.7%. PDP is a capacity/compute tradeoff, not proof of better accuracy. The model has fewer parameters but more FLOPs than baseline; disclose both when comparing. Residual/PDP/SE are established ideas, not claimed as original inventions.

## Four matched development trials

Baseline/hybrid x plain/CutMix, each300epochs with SGD LR0.1, momentum0.9/Nesterov, WD5e-4 on all learnable parameters, batch128, workers0, AMP, warmup5/per-step cosine to1e-5, seed42/split_seed2026. Plain is crop padding4/flip + CE. CutMix alpha1, probability0.5; no smoothing. Train45k/validation5k from official training data. These are new experiment IDs, not re-execution of deleted runs. Select only on validation; a single seed is provisional, not a robustness estimate.

Two new full-epoch smoke runs use CutMix and the same architectures. Their timing, finite losses, full45000/5000 counts and <=1% AMP skips gate long training. Smoke accuracy is not a ranking or final result. RandAugment/smoothing remain future controlled trials, not enabled without evidence. The final candidate still needs SGD/Adam/multiple LR comparisons and CIFAR100 ablations before full-data freeze/test.

## Safe old/new queue transition

Completed CIFAR10 outputs/logs (10grid +2smoke) were removed on request. Configs remain in `excluded_completed_configs`, not in executable lists. Remaining14 old grid jobs and4 CIFAR100 smoke jobs are retained. Budget is rebuilt from retained CIFAR100 measurements. Mixed old reports are regenerated without removed results. No removed accuracy results are archived for later selection.

The protected old Adam0.003 run stopped after121/150epochs because of system low virtual memory, not CUDA OOM. Byte-identical Python/model source is preserved at `D:\DeepLearning\Finnal\outputs\legacy_source_59c2376`; the old manifest declares this `source_root` and its required fingerprint. **Do not delete that snapshot until legacy jobs finish.** No checkpoint hash is rewritten or bypassed. Absolute config data/output paths allow resume from the snapshot.

The supervisor is stdlib-only; budget/report helpers run in short-lived subprocesses instead of keeping PyTorch/reporting resident throughout training. Per-study queue lock plus `D:\DeepLearning\Finnal\outputs\training.lock` prevent simultaneous project queues. Inspect process command lines before touching stale locks: Windows can reuse PIDs. Never kill unrelated applications/change pagefile automatically. Source remains frozen once jobs start. `STOP_AFTER_RUN` in a study output root stops its queue at a run boundary; it does not interrupt an epoch.

Training entry points never read official test. Freeze recipe/epoch budget, retrain from scratch on50k, then use the existing explicit test-evaluation gate; do not tune against repeated test scores.

## Research provenance and limits

- Internal TickNet paper (`D:\DeepLearning\Finnal\KnowLedge\1-s2.0-S0925231224007136-main.pdf`): efficiency/PDP/residual motivation; paper's95.01% CIFAR10 is not >97% evidence.
- `D:\DeepLearning\Finnal\KnowLedge\CNNs_NT.pdf` and selected sections of `UnderstandingDeepLearning_08_05_24_C.pdf`: CNN/optimization/generalization background. The541-page book was **not** read end-to-end; unavailable video transcripts are not treated as reviewed.
- Wide Residual Networks: https://arxiv.org/abs/1605.07146 (width/depth tradeoffs; not copied wholesale).
- Squeeze-and-Excitation Networks: https://arxiv.org/abs/1709.01507 (channel gating).
- CutMix: https://arxiv.org/abs/1905.04899 (patch/label mixing, area-corrected lambda).
- Cutout: https://arxiv.org/abs/1708.04552 and RandAugment: https://arxiv.org/abs/1909.13719 (future regularization options, not current recipe).

Full training takes hours/days. GPU thermals, other processes and commit pressure can change estimates; launching a queue is not completion or achievement of97%.

## Launch sequence

`D:\DeepLearning\Finnal\run_approved_training.ps1` first checks system memory (freeRAM>=1536MiB/freecommit>=5120MiB, grounded in the observed trainer footprint). It runs the protected legacy resume manifest, then the new smoke+four trials, then all remaining old jobs. A failure, budget block or stop request stops the chain; there is no automatic retry. Original old configs stay unchanged; `D:\DeepLearning\Finnal\configs\study\legacy_launch` contains equivalent absolute-path copies so the snapshot does not interpret `datasets` relative to its own directory. Queue source fingerprint still checks the snapshot byte-for-byte.

```powershell
& 'D:\DeepLearning\Finnal\run_approved_training.ps1' -PreflightOnly
& 'D:\DeepLearning\Finnal\run_approved_training.ps1'
```

Do not launch this again if `approved_chain.lock` or a live project queue exists. Monitor new trials via `D:\DeepLearning\Finnal\outputs\study_improvement\status.json`; old runs via `D:\DeepLearning\Finnal\outputs\study_two_models\status.json`; chain progress via `D:\DeepLearning\Finnal\outputs\approved_chain_status.json`. Do not edit executable Python source while a queue using it is active.