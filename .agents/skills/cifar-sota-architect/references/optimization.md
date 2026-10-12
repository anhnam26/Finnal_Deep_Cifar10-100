# Optimization choices

SGD/momentum, Adam and AdamW are implemented. The exam matrix uses SGD
0.05/0.1/0.15 and Adam 0.0003/0.001/0.003. These ranges differ because identical
numeric LRs do not imply equivalent update scales. Print momentum, betas,
epsilon, weight decay, norm/bias exclusions, batch size, warmup and epochs.
Compare using validation; never select optimizer from official test results.

Search LR, decay, batch, warmup, cosine/one-cycle-shaped schedule, smoothing,
mixing policy, augmentation, EMA and architecture regularization. The schedule
is evaluated per batch from completed epoch and batch index; resume restores
optimizer moments/momentum, AMP scale and successful update count. A new epoch
budget changes cosine and requires a new run/study. Muon and SAM remain future
opt-in experiments rather than defaults or claimed improvements.
