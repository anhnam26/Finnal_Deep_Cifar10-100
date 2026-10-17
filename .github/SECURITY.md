# Security

Only load PyTorch checkpoints from sources you trust: restoring full optimizer
and RNG state uses Python serialization, which can execute code in a malicious
checkpoint.

Do not post credentials or private data in public issues. Use GitHub private
vulnerability reporting when enabled; otherwise open a minimal issue requesting
a private reporting channel, without exploit payloads or confidential material.
