# Shared Python environment verified October 13, 2026

The user's latest global policy supersedes the old project-local package policy.
Use only the existing Microsoft Store Python 3.13.14 selected in VS Code. The
`python`, `python3.13.exe` aliases and its resolved package `python.exe` refer
to this same installation. No new Python interpreter is downloaded or used.

- NVIDIA RTX 4060 Laptop GPU, 8188 MiB VRAM, driver 546.92.
- Existing PyTorch 2.7.1+cu118 and torchvision 0.22.1+cu118; CUDA available.
- Existing NumPy 2.2.6, pandas 2.3.2, scikit-learn 1.6.1, PyYAML 6.0.3,
  matplotlib 3.10.5, TensorBoard 2.20.0 and pytest 9.1.1.
- Installed in that same interpreter's normal user-site: timm 1.0.30, Optuna
  5.0.0, optuna-mcp 0.2.0, MCP 1.30.0, fpdf2 2.8.9 and missing transitive
  libraries. Thirteen distributions were added; none already installed changed
  version or was removed.
- CUDA 11.8 wheels work with the installed driver; no driver/toolkit upgrade.
- `scripts/setup_environment.py` invokes the exact selected `python3.13.exe -m
  pip install`, using normal package installation and constraints preserving
  working torch/torchvision/NumPy. It never creates a venv, installs another
  Python or uses an alternate package destination. Entry points refuse a
  different executable, Python version or virtual-environment prefix.
- `.codex/config.toml` is project-only. `codex mcp list` sees Optuna enabled,
  using the exact selected Python alias and `experiments/optuna.db`.

The original `.venv` creation/download was stopped following user steering.
The user then explicitly requested deletion. Automatic command review rejected
the agent's native PowerShell deletion with `blocked by policy`. The final
filesystem check confirms `.venv` no longer exists; no claim is made that the
rejected agent command removed it. All commands use the existing interpreter.

The obsolete `.deps` import hooks are removed from train, setup, MCP, tests and
skill helpers. Deleting the old directory with native PowerShell was rejected
by automatic command review (`blocked by policy`); it is unused, but on-disk
cleanup remains pending. No alternative deletion method was used.

Current audit: `reports/generated/shared-python-audit.json`. It verifies
`sys.prefix == sys.base_prefix`, import locations within this same interpreter's
site-packages, CUDA availability and unchanged protected package versions.
Microsoft Store redirects physical user-site storage to WpSystem; this is the
same Python313 user-site, not another environment.

User-authorized global changes: `C:/Users/anhna/.codex/AGENTS.md` now contains
the all-project Python rule; missing libraries are installed in the selected
Python's package directory. No global MCP registration or config.toml edit.
Older `isolation-*.json` reports are historical and describe the superseded
setup. The current suite has 40 passing tests and both dataset smoke runs
complete on CUDA using this one shared environment.
