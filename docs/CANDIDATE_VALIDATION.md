# StageFusionNet — implementation and smoke evidence

Root: `D:\DeepLearning\Finnal`. Measured on the existing Python3.13/PyTorch2.7.1+cu118 environment and RTX4060 Laptop8GB. These measurements validate feasibility and execution, **not final accuracy or fusion superiority**.

## Inference budget

Input[1,3,32,32], inference eval, conv/linear2 FLOPs/MAC; includes unfused BN, LayerNorm, pooling, gate sigmoid, broadcasting products and residual/fusion sums. Scalar operation convention is documented in `D:\DeepLearning\Finnal\profile_model.py`.

| Variant | Classes | Learnable parameters | MACs | FLOPs |
|---|---:|---:|---:|---:|
| fusion | 10 | 4,510,068 | 443,593,280 | 890,274,525 |
| fusion | 100 | 4,531,758 | 443,614,880 | 890,317,815 |
| no_fusion | 10 | 4,473,778 | 443,558,240 | 889,794,125 |
| no_fusion | 100 | 4,495,468 | 443,579,840 | 889,837,415 |

Fusion adds36,290parameters and480,400FLOPs to either head. Both heads satisfy <=6M/<1G with approximately11% FLOP headroom. Instructor convention and architecture non-duplication remain to be confirmed.

## Validation checks

- 62 Python tests passed, including both heads/variants finite gradients, sample-weighted metrics, stratified disjoint splits and train/validation transform isolation, no official-test access without confirmation, tiny overfit, config guards and source locks.
- Exact interrupted/resumed model identity for both variants, SGD/Adam, CPU/CUDA AMP with CutMix; checkpoint/scaler/best validation state retained.
- PowerShell viewer fixtures passed: defaults from unrelated cwd/spaces, automatic matrix discovery, saved-epoch dedup/reminders, smoke exclusion, waiting/running/completed/failure states and exports.
- Protected28files (BT1, KnowLedge, exam) matched pre-cleanup SHA256. Datasets and download verification retained; no new environment/packages, no external process kill, no push.

The initial two-step Adam AMP test overflowed once at scale65536. Initial scale is now4096 for training/benchmark; persistent >1% skipped-step guard remains unchanged. Rerun passed and real-data smoke recorded zero skips in all8epochs.

## Full real-data smoke

Each row is a fresh2epoch SGD0.1 plain crop/flip run, batch128/workers0, same seed42/split2026. Two-epoch cosine is a runtime check, not the150epoch screening schedule. Every epoch:45000train/5000validation, finite losses,0AMPskips; each completed validation export contains5000predictions.

| Dataset | Variant | Full epoch wall seconds | Peak allocated MiB | Peak reserved MiB |
|---|---|---|---:|---:|
| CIFAR-100 | no_fusion | 35.45 / 33.47 | 781.46 | 810 |
| CIFAR-100 | fusion | 36.81 / 35.71 | 798.24 | 850 |
| CIFAR-10 | no_fusion | 34.40 / 33.36 | 781.27 | 808 |
| CIFAR-10 | fusion | 38.50 / 37.97 | 798.06 | 850 |

Durable artifacts: `D:\DeepLearning\Finnal\outputs\candidate\smoke`, `budget.json`, `status.json`; each run stores weights/optimizer/RNG/config/split/source/logs, evaluation metrics/curves/predictions/PDF. Smoke completed, no long runs started and no official test evaluated. Peak memory is PyTorch allocation/reservation, not total GPU usage of desktop applications.

## Next execution gate

Conservative screening budget:38.505s x300epochs x1.25 reserve =4.01hours, excluding additional final stages/export costs. Actual throughput may drift with thermals and external apps. The separate12run LR matrix is prepared only; it has not been launched or chosen as a final recipe.

At the last post-smoke snapshot, free physical RAM was approximately1055MiB and free system commit approximately2550MiB. Earlier project runs exhausted host commit memory, not CUDA memory. Long screening is **blocked on memory preflight**, despite smoke passing. Free unrelated application memory manually; do not kill arbitrary processes or change environments/pagefile from the agent. Require at least1536MiB physical RAM and5120MiB free commit before long launch. Also check queue/trainer PIDs and `D:\DeepLearning\Finnal\outputs\training.lock` to avoid duplicate training.

After preflight, continue:
```powershell
python -B -u D:\DeepLearning\Finnal\study.py --manifest D:\DeepLearning\Finnal\configs\candidate\manifest.json
powershell -NoProfile -ExecutionPolicy Bypass -File D:\DeepLearning\Finnal\watch_training.ps1
```
Completed smoke is reused, not retrained. Keep executable Python frozen: required source fingerprint4861727ff906658c933ebc551c2786ed6a3172e7cb3bf6ae551490cd0d84bff6. Select on CIFAR-100 validation, confirm CIFAR-10, then test incremental recipe changes before the12run matrix/seeds/frozen50k and official-test/PDF finalization. CIFAR-10095% remains unproven; no measured long-run target accuracy exists.

## Retirement record

Old configs/models/generators/launcher/source snapshots/weights/optimizer states/logs were removed after exact inventory and historical metrics capture. Ignored weights are not recoverable from Git; tracked code/config history remains. Historical CIFAR-10 best held-out validation: baseline15095.50%, baseline30095.46%, preactivation15095.22%. These are not new results or official-test scores. Full knowledge book/video review and final report remain incomplete.