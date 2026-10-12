# CompactMixNet: design and evaluation protocol

This replaces ScaleGateNet and the earlier candidate networks, not their saved
historical results. Neither target accuracy has been demonstrated by this rebuild.
The final examination document remains the authority for the 6M-parameter and
1-GFLOP constraints. Confirm architecture acceptance/non-duplication with the
instructor; a custom composition is not a claim of research novelty.

## Backbone

- Input: RGB 32x32; 3x3 stride-one convolution to 40 channels.
- Stage resolutions: 32x32, 16x16, 8x8.
- Stage widths: 40, 112, 256; stage depths: 2, 3, 8.
- Transitions: pre-activation, 2x2 average pooling, 1x1 channel projection.
- Each residual block: pre-activation, dense 1x1 expansion by two, split into
  two depthwise spatial branches, concatenate, 3x3 eight-group mixing,
  squeeze/excitation, dense 1x1 projection, stochastic-depth residual addition.
- dualpath: branch kernels 3x3 and 5x5.
- localpath: branch kernels 3x3 and 3x3, a controlled receptive-field ablation.
- Head: BatchNorm, SiLU, global average pooling, linear 10/100-class classifier.
- Stochastic depth increases linearly from zero to 0.1 across the 13 blocks.

| Variant | Classes | Parameters | GFLOPs |
| --- | ---: | ---: | ---: |
| dualpath | 10 | 5,248,172 | 0.842407 |
| dualpath | 100 | 5,271,302 | 0.842453 |
| localpath | 10 | 5,208,748 | 0.832839 |
| localpath | 100 | 5,231,878 | 0.832885 |

Measured by the included shape-aware analytical profiler at batch size one.
Multiply-add = two FLOPs; conv/linear biases, BN (4/element), SiLU (6/element),
sigmoid (4/element), average pools, residual adds and channel multiplications
are included. These are explicit operation-count conventions, not hardware
latency or an assertion that transcendental functions cost exactly N cycles.
Memory-only split/cat/flatten are excluded. Stochastic depth is identity at
inference. The script fails immediately if a model exceeds either budget.

## Training recipe

The default runs dualpath on CIFAR-10 then CIFAR-100 independently, from scratch,
without an external teacher, pretrained weights, test augmentation or ensemble.
Each official training set is stratified into 45,000 training and 5,000 validation
images using split seed 2026. Every run saves the actual indices and their SHA256.
Validation uses a separate deterministic transform, not the training augmentation.
Official 10,000-image test data is opened only after all epochs have finished.

- 600 epochs, batch size 128, initialization seed 42, two Windows workers.
- SGD: LR 0.1, momentum 0.9, Nesterov, weight decay 0.0005.
- Adam option: LR 0.001, betas (0.9,0.999), epsilon 1e-8, weight decay 0.0001.
- Biases and one-dimensional normalization weights have zero weight decay.
- Linear warmup from 10% of the base LR for min(5, epochs//10) epochs,
  then per-batch cosine decay to 0.1% of the base LR.
- Reflect-padded random crop, random horizontal flip, torchvision CIFAR10
  AutoAugment policy (also applied to CIFAR-100; not a separately searched policy).
- CutMix alpha=1 on 50% of batches, MixUp alpha=0.2 on 25%, none on 25%.
- Mixing is disabled for the final min(30, epochs//10) epochs; AutoAugment remains.
- Soft-target cross-entropy, label smoothing 0.05, gradient norm clipping at 5.
- CUDA float16 AMP with initial scale 4096; optimizer skips/scale are logged.
- EMA decay min(0.999, (1+updates)/(10+updates)); floating buffers are averaged,
  integer buffers copied. No extra BatchNorm calibration or hidden test tuning.
- Raw and EMA models are evaluated on validation every epoch; the best validation
  score alone selects weights. Ties retain the earlier checkpoint/raw candidate.
- Default deterministic algorithms; --fast trades reproducibility for speed.

Training accuracy is label-weighted under MixUp/CutMix and is NOT directly
comparable to clean validation/test accuracy. Epoch metrics use sample-weighted
means. Test output includes top-1/top-5, macro F1, per-class recall and confusion
matrix. Targets are evaluated against test accuracy, never fabricated or used
to stop training as soon as test performance looks favorable.

--matrix generates SGD LRs 0.05/0.1/0.15 and Adam LRs 0.0003/0.001/0.003:
12 independent runs for both datasets. Compare recipes using validation, not
the best test score. Report all trials, compute multiple seeds if budget allows,
and do not claim a single favorable trial proves reliable 98%/95% performance.

## Durability and reproducibility

Checkpoint contents include raw/EMA weights, optimizer, AMP scaler, epoch,
next batch, completed optimizer-step count, partial-epoch weighted metrics,
full epoch history, RNG states, configuration, split hash and source hash.
The epoch shuffle is reproducible, and each image's augmentation has its own
seed derived from seed/epoch/original index. Worker prefetch and worker restart
therefore cannot alter which augmented samples are used after a mid-epoch resume.
DataLoader has a separate RNG so recreating its iterator does not perturb the
training RNG. The per-batch scheduler is reconstructed from the saved position.

Writes go to a temporary file, flush/fsync, then atomically replace last.pt.
The previous durable checkpoint is retained as previous.pt, and recovered if
last.pt is absent or unreadable. best.pt stores inference weights only.
Checkpoint rotation can briefly leave last.pt absent, so previous.pt is always
considered during recovery. PyTorch deserialization is trusted-local-only:
never point resume at downloaded/untrusted checkpoint files.

Ctrl+C/SIGTERM request a safe stop after the in-flight batch, rather than raising
KeyboardInterrupt in the middle of optimizer updates. Validation/test restarts
from its beginning if interrupted; completed training batches are not replayed.
An OS-managed exclusive file lock prevents two trainers in the same output root;
the OS releases it after a crash, so stale PID text does not require deletion.

A hard kill, power failure or hardware failure cannot execute a new save.
It resumes from the last completed durable checkpoint: by default every 50
batches or 120 seconds, whichever threshold is reached first at a batch boundary.
--checkpoint-every 1 minimizes lost completed batches at significant disk-I/O
cost. Unfinished writes never become the active checkpoint. No software can
guarantee disk survival or save the currently executing CUDA kernel on power loss.
Exact bitwise recovery is tested in the same environment; PyTorch does not
promise identical results across different platforms, versions or devices.

## Sources for established components

- Residual learning: https://arxiv.org/abs/1512.03385
- Squeeze-and-Excitation: https://arxiv.org/abs/1709.01507
- Stochastic depth: https://arxiv.org/abs/1603.09382
- AutoAugment: https://arxiv.org/abs/1805.09501
- MixUp: https://arxiv.org/abs/1710.09412
- CutMix: https://arxiv.org/abs/1905.04899
- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html

These sources motivate components, not accuracy predictions for this untrained
architecture. Lightweight, from-scratch CIFAR-100 at 95% is an exceptionally
ambitious target; architecture implementation is not evidence of reaching it.
