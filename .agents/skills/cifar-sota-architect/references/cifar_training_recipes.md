# Initial recipe

600 epochs, batch 128, seed 42, split seed 2026, deterministic CUDA, FP16 AMP,
gradient clipping 5. Warmup is min(configured 5, epochs/10) epochs; cosine ends
near 0.1% of peak LR. Default SGD is 0.1 with momentum 0.9, Nesterov and weight
decay 0.0005; Adam is 0.001 with betas 0.9/0.999, epsilon 1e-8, decay 0.0001.
Norm/bias weights have no decay. Raw and EMA validation candidates compete;
strict accuracy improvement replaces best and ties retain the earlier choice.

Crop/flip/AutoAugment, smoothing 0.05, mixing probability 0.75 with equal
Mixup/CutMix choice, alphas 0.2/1.0. EMA decay is min(0.999,(1+steps)/(10+steps)).
Random erasing is disabled initially. These are starting hypotheses, not evidence
that every component helps. Compare against basic crop/flip and no mixing/EMA
before attributing a gain to architecture. No external teacher or pretrained
weights. Hardware latency and VRAM must be measured separately from FLOP counts.
