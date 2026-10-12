# CIFAR-100 evidence

Model L: 5,271,302 learnable parameters and 0.842453 GFLOPs; same backbone as
CIFAR-10 with a 100-class head. Train it independently from scratch.
Approximately 90% is an ambitious research target under this budget; no
full-data accuracy is established by setup tests. Do not transfer CIFAR-10
scores into expectations for CIFAR-100 or claim results from 128-image checks.

Use 450 samples per class for training and 50 per class for validation before
freezing. Final retraining uses all 500 training samples per class. Analyze
macro F1, per-class recall and confusion matrix from the single frozen test
evaluation. Log optimizer/LR settings and compare equal validation budgets.
If regularization underfits, adjust it using validation and ablations.
