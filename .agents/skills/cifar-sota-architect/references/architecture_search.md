# Search implementation

`src/models.py` builds custom CompactMix-L, WRN-16-4, PreActResNet and CIFAR
ConvNeXt. `src/timm_models.py` adapts ResNet, ResNeXt, ConvNeXt V2, EfficientNet,
RegNet, MobileNet, ViT, DeiT, Swin and MobileViT reference families, from scratch.
Convolutional reference stems preserve 32x32. ViT/DeiT use 4x4 patches; Swin
uses 2x2 patches and a small configured backbone. Reference size is not evidence
of exam compliance: profile each candidate and prune over-budget candidates.

Default Optuna searches four custom/baseline families; `--search-families`
selects additional supported families. CompactMix dimensions: widths, three
stage depths, simple/stacked stem, context kernel 3/5/7, expansion 2/4,
none/SE/ECA attention, BN/GN/spatial LN, ReLU/SiLU/GELU, residual scale,
DropPath, dropout and average/average-max pooling. Other reference model kwargs
are accepted through YAML `model`; generic block replacements, CBAM, SAM and
Muon are not implemented or validated and must be separate hypotheses.

Progressive sequence: 1-3 epoch smoke, 20-30 epoch screening, 100-200 epoch
medium budget, 600 epoch finalist, repeated seeds. Each budget is a separate
study name; resume within the same budget. HPO reports validation accuracy,
parameters, FLOPs, seed, checkpoint, time and failures in local SQLite. Multiple
objectives maximize validation and minimize parameters; epoch pruning applies
only to single-objective studies because Optuna cannot report/prune steps for
multiple objectives. Budget pruning applies to both.
