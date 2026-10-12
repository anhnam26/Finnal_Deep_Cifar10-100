# Augmentation choices

Supported image policies: basic random crop/flip, CIFAR10 AutoAugment,
RandAugment (two ops, magnitude nine), TrivialAugmentWide and optional random
erasing. Validation/test use tensor conversion and fixed dataset normalization
only. CIFAR10 AutoAugment on CIFAR-100 is an explicit initial policy, not a
searched CIFAR-100 optimum.

Supported label mixing: Mixup and area-corrected CutMix. Soft labels use the
configured smoothing. Logged training accuracy weights both source labels and
is not directly comparable with clean validation/test accuracy. Disable mixing
for the last min(30,epochs//10) epochs. Optuna tests none/Mixup/CutMix separately;
the default combined recipe should be ablated under fixed seeds and budgets.
Per-image augmentation seeds depend on seed/epoch/original index, so worker
prefetch/restart does not change the replayed epoch's samples.
