# CompactMix-L

Input RGB 32x32; stride-one 3x3 stem; stages 32/16/8 with widths 40/112/256
and depths 2/3/8. Transitions use normalization/SiLU, average pooling and 1x1
projection. Each block expands twofold, splits equally into 3x3 and 5x5
depthwise paths, concatenates, applies eight-group 3x3 mixing, SE and dense
1x1 projection, then stochastic-depth residual addition. Final normalization,
SiLU, global average pooling and 10/100-class linear head.

| Classes | Parameters | FLOPs | MACs |
|---:|---:|---:|---:|
| 10 | 5,248,172 | 842,406,944 | 409,492,864 |
| 100 | 5,271,302 | 842,453,114 | 409,515,904 |

Count convention: multiply-add is two operations; convolution/linear biases,
BN (4/element), SiLU (6), sigmoid (4), pooling, residual addition and channel
multiply are included. Split/cat/reshape are memory operations. These are
analytical estimates, not cycle costs or hardware latency. Inference DropPath
and dropout are identity. Both class heads satisfy <=6M and <1G constraints.

Configurable ablations include branch kernel, width/depth/expansion,
SE/ECA/none, BN/GN/spatial LN, activation, residual scale, stem, pool, dropout
and stochastic depth. `norm=ln` means GroupNorm(1,C), normalization over
channels and spatial dimensions, rather than ConvNeXt's per-pixel channel LN.
The separate ConvNeXt baseline uses per-pixel channel LayerNorm.

Reference families use CIFAR-aware stems or explicit small patches. Profile
and reject over-budget models before training. Timm reference operation counts
add functional attention estimates and a 10% margin; custom L's explicit count
is the exam budget evidence. Code supports references, not claims of their
accuracy or of every arbitrary timm override being fully profiled.

Keep a fixed stratified training/validation split. Raw/EMA validation select
best. Freeze recipe, weight type and winning epoch count, initialize afresh on
all 50k training images, then evaluate the official test once. Final-phase best
is the fixed final epoch rather than test-selected weights. Checkpoints are
epoch-atomic, exactly best.pt/last.pt per experiment, with strict config/source/
environment matching and deterministic sample-seeded replay. See README for
commands and `reports/architecture_notes.md` for untested hypotheses.

Normal logs are emitted once per completed training/validation epoch after
durable checkpoint and metrics writes. Both best and last are updated at epoch
boundaries, including deferred best recovery. No batch/startup/resume logs.
