# CIFAR backbone redesign: matched evidence before longer training

## Why replace the previous strategy?

The completed Baseline Plain300 run reached best **95.46% validation** at epoch292,
not official-test accuracy. Its epoch150 result was90.20%, with LR0.05134 instead
of the0.00001 endpoint of a150epoch schedule. Extending the cosine schedule is
not an architectural improvement. Deleted older outputs cannot substantiate old
accuracy curves. Hybrid feasibility/smoke is not superiority evidence.

The previous chain was deliberately paused with user approval, not a spontaneous
training failure: Baseline CutMix saved133/300, best validation89.06%. Its source,
checkpoint, RNG/optimizer/scaler/config/split/history are preserved. Both Hybrid
trials and remaining legacy jobs are no longer automatically scheduled.

## Candidate hypothesis (`preact`)

- Dense3x3 convolutions, no depthwise/PDP/SE/CutMix in this comparison.
- Preactivation BN-ReLU-conv residual blocks; identity shortcuts keep their
  unactivated signal; shape-changing1x1 projections use preactivated inputs.
- Stage widths **40/104/248**, depths **2/3/4**, resolution **32/16/8**.
- Bare3x3 stem; terminal BN/ReLU, global mean and10/100-way classifier.
- This custom asymmetric allocation spends more capacity at8x8 and avoids the
  old4x4 bottleneck. It is a hypothesis to test, not a novel block or proven winner.
- Shape-only profiling considered40/104/256 (0.9560G FLOPs at100classes) and
  40/112/256 (1.00015G, rejected).40/104/248 leaves greater FLOP margin.
- Profile: CIFAR-10 **4,727,778 parameters / 922,362,218 FLOPs**;
  CIFAR-100 **4,750,188 / 922,406,948**, same counting convention.

Known foundations must be cited: [Identity Mappings in Deep Residual Networks,
He et al.](https://arxiv.org/abs/1603.05027) and [Wide Residual Networks,
Zagoruyko & Komodakis](https://arxiv.org/abs/1605.07146).
The layout is not a copied named reference architecture. Distinctiveness and
acceptance under the assignment still require instructor review; changing widths
alone is not a guarantee of acceptance. No accuracy above97% is promised.

## Locked screening protocol

Two new run IDs, **Baseline vs Preact**, plain crop/flip augmentation. Each uses
SGD0.1, momentum0.9, Nesterov, WD5e-4 on all learnable parameters, batch128,
AMP, workers0, seed42, split seed2026,150epochs,5warmup epochs then per-step
cosine to1e-5.45k/5k stratified split; no debug batch limits, no extra data.
Only `variant` and `output_dir` differ. Baseline remeasurement is intentional and
does not overwrite deleted runs or the completed300epoch result.

Two full2epoch smoke runs precede long training. They check45000/5000 samples,
finite losses, bounded AMP skips, exports, throughput, and host memory. Low smoke
accuracy is not an architecture selection criterion. The existing supervisor
derives a conservative time budget and refuses fingerprint mismatches.

Compare validation curves at matched epochs (50/100/150), best and final accuracy,
losses, AMP skips, total wall time, parameters and the same transparent FLOP
convention (2FLOPs/MAC, plus BN/activation/add/GAP). Equal epochs are not equal
compute. One seed is screening only. If the candidate does not beat this
reference, do not claim improvement or extend only its schedule to rescue it.

Then, conditional on evidence: additional paired seeds and CIFAR-100 validation,
one-at-a-time Cutout ablation, required SGD/Adam LR matrix. Selection remains
validation-only. Freeze the final recipe before full50k training and explicitly
gated official-test evaluation; no iterative test tuning, ensemble, TTA or
distillation without approval. Official-test >97% remains an aspiration.

## Operation and preservation

Project root: `D:\DeepLearning\Finnal`.

```powershell
python -B D:\DeepLearning\Finnal\profile_model.py --variant preact --classes 100
python -B D:\DeepLearning\Finnal\study.py --manifest D:\DeepLearning\Finnal\configs\redesign\manifest.json --stage smoke
python -B D:\DeepLearning\Finnal\study.py --manifest D:\DeepLearning\Finnal\configs\redesign\manifest.json --stage all
powershell -NoProfile -ExecutionPolicy Bypass -File D:\DeepLearning\Finnal\watch_training.ps1 -ManifestPath D:\DeepLearning\Finnal\configs\redesign\manifest.json -StudyRoot D:\DeepLearning\Finnal\outputs\study_redesign
```

Do not run duplicate queues. Keep Python frozen after the new queue starts.
The default viewer also discovers `configs/redesign/manifest.json`; reopen only
the viewer to reload its updated discovery (not the trainer).
The smoke/all commands are sequential alternatives to managed background launch,
not instructions to start another process beside an already running queue.
Source fingerprints include all root/model Python files, including preparers.

Old-source snapshot: `D:\DeepLearning\Finnal\outputs\improvement_source_61af26cc`.
Paused-run evidence and snapshot-aware, absolute-data-path resume manifest:
`D:\DeepLearning\Finnal\outputs\redesign_transition`.
Original old configs/manifests remain untouched. For deliberate old resumption,
first stop the new queue, review/remove the old STOP_AFTER_RUN, and use
`resume_manifest.json` from that evidence directory; never bypass fingerprint
checks or resume old checkpoints using redesigned root source. Retain the separate
legacy snapshot59c2376 as well. No checkpoint/output deletions, environment,
dependency or pagefile changes are part of this task.