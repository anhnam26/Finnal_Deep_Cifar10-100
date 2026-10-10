# Retired StageFusionNet results

Single seed42, split2026, stratified45k/5k;150epochs SGD0.1/crop-flip/plain CE. These are **recorded training validation**, not official test scores.

| Dataset | Variant | Best validation | Best epoch | Last validation |
|---|---|---:|---:|---:|
| CIFAR-10 | no_fusion | 95.18% | 146 | 95.14% |
| CIFAR-10 | fusion | 95.30% | 137 | 95.22% |
| CIFAR-100 | no_fusion | 75.86% | 141 | 75.58% |
| CIFAR-100 | fusion | 76.54% | 124 | 76.26% |

C10 independent best-checkpoint exports agree with logged95.18/95.30%. C100 no_fusion export75.84% vs logged75.86% differs by one image; differing cuDNN settings were noted, cause unconfirmed. Single-seed paired bootstrap C10 fusion-minus-plain0.12 points, CI approximately[-0.4005,+0.6605]; C100 CI approximately[-0.32,+1.68]. Image bootstrap is not seed uncertainty; no stable fusion superiority established.

User authorized retirement after targets98/95 were not achieved.24 inventoried ignored weights/checkpoints (722,734,408bytes) deleted, including smoke/export weights; these are not Git-recoverable. Old model source remains available in Git history, but runnable root/configs now describe a different model. Old entrypoint configs were removed to prevent accidental restart. Historical metrics/logs/predictions/PDFs and embedded frozen configs remain in `D:\DeepLearning\Finnal\outputs\candidate` and `D:\DeepLearning\Finnal\outputs\candidate_cifar10`. Historical export completion markers do not imply deleted weights are still present. Do not reuse old checkpoint paths with ScaleGateNet.

These150/plain results are not a matched architecture comparison against new300/CutMix runs. Official test remains untouched.