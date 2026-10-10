# ScaleGateNet preflight — 2026-10-10

## Checks completed before long launch

-65Python tests passed: shapes/finite all-parameter gradients, profile budget both variants/heads, neutral-gate equality, raw residual identity, SGD/Adam exact CPU/CUDA AMP epoch-boundary resume with CutMix, augmentation isolation and queue/evaluation guards.
-PowerShell fixture tests passed after setting fixture Out-File width240. The headless console had narrowed to16columns and wrapped redirected strings; this was a fixture rendering issue, not training/viewer logic failure. Defaults/CLI discovery updated to ScaleGateNet.
-All modified Python parsed; config pairs differ only variant/output; fingerprints agree; protected BT1/KnowLedge/exam/download SHA256 checks passed. No packages installed/environment or machine settings changed.

## Full real-data smoke

`D:\DeepLearning\Finnal\configs\scalegate\manifest.json`, four configurations2epochs each; all eight epochs45000training/5000validation, finite losses, zeroAMPskips. Four validation exports each5000predictions/complete marker/PDF; best/last checkpoint ZIP format verified. All trainer/queue stderr empty. This is feasibility evidence, NOT final accuracy evidence.

| Dataset / variant | Epoch wall seconds | Max allocated MiB | Max reserved MiB |
|---|---|---:|---:|
| C100 gate on | 35.667 / 32.504 | 1283.10 | 1324 |
| C100 gate off | 32.645 / 30.153 | 1243.38 | 1282 |
| C10 gate on | 33.365 / 31.342 | 1282.86 | 1324 |
| C10 gate off | 31.655 / 30.390 | 1243.14 | 1282 |

Measured conservative35.667s/epoch x1200epochs x1.25reserve = **14.861hours**, below24h configured allowance. Separate inference export/report overhead and changing laptop thermals can affect runtime; no completion-time promise. Latest prelaunch observation6241MiB freeRAM,13882MiB freecommit,69GiB free disk. Launch rechecks thresholds; smoke ended and both locks removed.

## Long protocol

C100 gated300 -> C100 ungated300 -> C10 gated300 -> C10 ungated300. Same seed42/split2026/SGD/crop-flip+CutMix across pair. Official test,12-run LR matrix and AutoAugment trials NOT launched. Executable source frozen at `9a4fd7528bf908f30c23963b0d6a42e35d13504a7e9f5f2607394ee5c9f9482f` once long queue starts.

Monitor `D:\DeepLearning\Finnal\outputs\scalegate\status.json`, per-run metrics/timing and `D:\DeepLearning\Finnal\outputs\scalegate\logs\training_readable.log`; details viewer shows LR/AMP/VRAM/timing. Initial low accuracy is expected, not a reason to reject/delete a run. Final accuracy/ablation superiority remain unmeasured.