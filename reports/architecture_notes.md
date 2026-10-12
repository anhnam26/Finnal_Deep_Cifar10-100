# CompactMix-L research notes

The proposed L extends the repository's previous CompactMixNet idea with
configurable ablations and a strict validation protocol. No full training result
or accuracy gain has been measured during setup. The final examination forbids
duplicate student architectures; this implementation cannot verify other groups'
models and is not a research-novelty claim.

Hypothesis to test: combining local 3x3 and contextual 5x5 depthwise branches
before eight-group spatial mixing improves validation at a fixed sub-6M budget.
Compare against 3x3/3x3 and no SE using identical recipe, seeds and training
duration. Follow with no mixing, no EMA and no DropPath to separate architecture
and recipe effects. Log successful and failed hypotheses after actual runs.

The reference ConvNeXt/Muon repository claims CIFAR-10 98.10% with 12.37M
parameters; it is not independently reproduced, exceeds this exam's limit, and
its score does not predict L's accuracy. Its stride-one stem, kernel schedule,
selective SE and EMA calibration motivate experiments. No Muon implementation,
pretrained weights, teacher, TTA or ensemble was imported into default training.

Implemented exploration includes four custom/CIFAR baselines and ten adapted
timm reference families. Oversized references are rejected before training.
The analytical budget count for custom L includes explicit elementwise costs;
timm references use additional functional estimates and a 10% operation margin.
Reference counts are estimates, not measured latency or complete profiler proof
for every possible arbitrary timm kwargs combination.
