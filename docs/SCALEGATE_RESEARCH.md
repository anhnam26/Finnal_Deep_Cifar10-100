# Evidence-first ScaleGateNet redesign

## Internet results and their limitations

Checked primary paper tables, not leaderboard numbers detached from protocols:

| Published configuration | CIFAR-10 test accuracy | CIFAR-100 test accuracy | Conditions |
|---|---:|---:|---|
| PyramidNet + ShakeDrop + AutoAugment | 98.5% | 89.3% | Table2 reports1.5/10.7% error; no extra training data |
| ViT-H/14 JFT | 99.50% | 94.55% | 632M parameters; JFT pretraining, far outside exam budget |
| EfficientNet-L2 + SAM | 99.70% | 96.08% | Table3 reports0.30/3.92% error; ImageNet plus JFT pretraining, resolution475 |

Sources:
- AutoAugment: https://arxiv.org/html/1805.09501v3 (Table2).
- ViT: https://arxiv.org/html/2010.11929v2 (Tables1/2).
- SAM: https://arxiv.org/html/2010.01412v3 (Section3.2/Table3).
- Res2Net: https://arxiv.org/abs/1904.01169 (hierarchical split-channel features).
- Squeeze-and-Excitation: https://arxiv.org/abs/1709.01507 (channel recalibration).
- CutMix: https://arxiv.org/abs/1905.04899 and https://github.com/clovaai/CutMix-PyTorch .

Published test results cannot be substituted for this project's45k/5k validation. No verified configuration in the inspected evidence simultaneously satisfies98%/95%, <=6M, <1G, and CIFAR-only scratch training. Do not infer that shrinking a pretrained network preserves accuracy. No external implementation/checkpoint was copied or installed. Concepts are cited; implementation is written in the project's existing PyTorch style. No novelty or instructor acceptance is claimed.

## Architecture

`D:\DeepLearning\Finnal\models\scalegate.py`: stride1 dense3x3 stem; widths40/120/320 and depths2/3/4 at32/16/8. Each preactivated block has dense3x3 feature extraction, four spatially convolved channel splits with hierarchical residual links, BN/ReLU and dense1x1 remixing, then a raw-identity/projection residual addition. Unlike standard Res2Net bottlenecks, the first spatial extraction is dense3x3 and every split is convolved. This is a composition hypothesis, not proof of originality or better accuracy.

Stages2/3 optionally apply image-conditioned channel recalibration using GAP, a C/8 bottleneck, sigmoid and2x scaling initialized near identity. No stage-descriptor fusion or LayerNorm head. GAP+linear outputs10/100 classes.

The compatibility labels `fusion` and `no_fusion` now mean **local gate on/off**, respectively. Reporting and config metadata must identify ScaleGateNet, not StageFusionNet. Old weights are incompatible. Both variants retain exactly the same non-gate topology; neutral gate parameters reproduce the no-gate forward output in a regression test.

| Variant | Classes | Parameters | FLOPs |
|---|---:|---:|---:|
| gate on (`fusion`) | 10 | 5,096,535 | 890,209,140 |
| gate on (`fusion`) | 100 | 5,125,425 | 890,266,830 |
| gate off (`no_fusion`) | 10 | 4,981,490 | 889,624,330 |
| gate off (`no_fusion`) | 100 | 5,010,380 | 889,682,020 |

Input1x3x32x32; Conv/Linear2FLOPs/MAC; all hierarchical/residual adds and gate multiplications counted through explicit modules; BN4/element, ReLU1, sigmoid4, GAP/biases counted. Split/cat/flatten are memory operations excluded, not free hardware latency. Unfused inference convention, no training FLOPs claim.

## First matched experiment

Four300epoch development runs: C100 gate on/off, then C10 gate on/off. Fixed45k/5k stratified split2026 and seed42. SGD0.1, momentum0.9, Nesterov, WD5e-4 on all learnable parameters, batch128/workers0; warmup5 and per-step cosine to1e-5. Crop/flip plus CutMix alpha1/probability0.5, plain CE without smoothing; train accuracy is lambda-weighted. No AutoAugment in the first experiment. torchvision CIFAR10 AutoAugment policy is available for a separate controlled recipe trial, including policy transfer to C100, but is not automatically launched.

Four full two-epoch smoke runs use the same recipe except schedule length/no warmup/output. Smoke establishes feasibility, not final accuracy. Existing source freeze, fail-stop locks, <=1% AMP skips/epoch and full sample checks remain. Long launch requires >=1536MiB free RAM, >=5120MiB free commit and >=12GiB disk. Runtime estimate must come from smoke, not these parameter counts.

SGD/Adam multi-LR matrix, repeated seeds, full50k retraining and official test remain subsequent work. Matrix generator can prepare configs but they are not automatically launched. Gate gains require same-recipe paired runs and seed replication; old StageFusionNet150/plain results are historical context, not a matched architecture comparison to300/CutMix.

## Retirement

User authorized deletion of unsuccessful old models. Old StageFusionNet source/config entrypoints and24 inventoried checkpoint/model-weight files are retired; historical metrics, logs, predictions and PDFs remain. BT1, KnowLedge, exam, datasets, download evidence and Git history remain protected. Ignored weights cannot be restored from Git; old export completion markers describe historical exports, not current availability of deleted weights.