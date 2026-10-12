# Environment verified October 13, 2026

Use the existing Microsoft Store Python 3.13.14 selected in VS Code. The
`python`, `python3.13.exe` aliases and its resolved package `python.exe` refer
to this same installation. No new Python interpreter is downloaded or used.

- NVIDIA RTX 4060 Laptop GPU, 8188 MiB VRAM, driver 546.92.
- Existing PyTorch 2.7.1+cu118 and torchvision 0.22.1+cu118; CUDA available.
- Existing NumPy 2.2.6, pandas 2.3.2, scikit-learn 1.6.1, PyYAML 6.0.3,
  matplotlib 3.10.5, TensorBoard 2.20.0 and pytest 9.1.1.
- Added only in `.deps`: timm 1.0.30, Optuna 5.0.0, optuna-mcp 0.2.0,
  MCP 1.30.0, fpdf2 2.8.9 and missing transitive libraries.
- CUDA 11.8 wheels work with the installed driver; no driver/toolkit upgrade.
- `scripts/setup_environment.py` reuses this interpreter and installs missing
  libraries with `pip --target .deps --no-deps`, resolving their requirements.
  It never creates a venv, installs Python or changes global packages.
- `.codex/config.toml` is project-only. `codex mcp list` sees Optuna enabled,
  using the existing Python and `experiments/optuna.db`.

The original `.venv` creation/download was stopped following user steering.
The user then explicitly requested deletion. Automatic command review rejected
the agent's native PowerShell deletion with `blocked by policy`. The final
filesystem check confirms `.venv` no longer exists; no claim is made that the
rejected agent command removed it. All commands use the existing interpreter.

Before/after inventories in `reports/generated/isolation-audit.json` confirm
global Python distributions and `~/.codex/config.toml` unchanged. No files were
added to `~/.agents` or the audited global plugin/skill folders. Two existing
Codex plugin installation-marker files changed during the session (plugin-
management and work-pets); the task did not invoke a global plugin installer.
These observed metadata changes prevent a claim that every external file was
unchanged. Ordinary installer temporary files and initial pytest temp folders
were outside the repository; subsequent tests use a project-local basetemp.
