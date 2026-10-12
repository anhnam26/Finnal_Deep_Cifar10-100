# CIFAR-10 evidence

Model L: 5,248,172 learnable parameters, 0.842407 GFLOPs (2 FLOPs/MAC plus
explicit elementwise operations), 32/16/8 stage resolutions, widths 40/112/256,
depths 2/3/8. No full-data accuracy is established by setup tests.

Ready baselines: WRN-16-4, PreActResNet, CIFAR ConvNeXt and configured timm
reference families. Compare the same split, seeds, epoch budgets and recipe.
Do not treat a library model name as a measured score.

Reference inspected: https://github.com/akira-n-28/convnext-muon-cifar10
README claims 98.10% test accuracy with 12.37M parameters and 600 epochs;
this is a repository claim, not independently reproduced evidence, and exceeds
the exam's 6M limit. Useful ideas are stride-one stem, stage-specific kernels,
selective SE, calibrated EMA and warmup/cosine. Its Muon/AdamW and test-driven
checkpoint selection are not copied into this exam protocol.
