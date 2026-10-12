# Experimental protocol

Training and validation are stratified from official training data, 45k/5k.
Store actual indices, split SHA256, config, training-source fingerprint, git
commit, Python/PyTorch/NumPy/CUDA/device, seed and UTC start time. Choose best by
validation only. Ties keep the earlier checkpoint. Repeated seeds and ablations
come before freezing; official test stays sealed throughout.

Each run has exactly one best.pt and one last.pt. Last holds model, EMA,
optimizer, AMP scaler, epoch, global step, RNG, history and best weights for
recovery. Flush/fsync and atomic replacement preserve the previous last on a
failed write. No numbered/rotating checkpoint. A hard stop replays the unfinished
epoch. First Ctrl+C finishes and saves the epoch; second stops immediately.
Unreadable last errors explicitly; there is no hidden fallback to older epochs.

Finalization fixes recipe, winning weight type and best validation epoch count,
reinitializes and trains on all 50k images. Final-phase best aliases the fixed
last epoch, because no validation/test selection occurs in that phase. Evaluate
official test once, record checkpoint SHA256 and cache the result. Changing a
model from test evidence invalidates the held-out claim; do not do it.

Raw JSON/CSV persist measured results, including technical checks flagged
synthetic/subset. The leaderboard excludes those checks, and PDF says explicitly
when full-data results or exam comparisons are absent. Blank test accuracy stays
blank. Setup verification is not an accuracy result or completed final report.
