# Python environment audit: October 12, 2026

## Preserved development interpreter

The currently selected VS Code interpreter for this workspace is:

    C:/Users/anhna/AppData/Local/Microsoft/WindowsApps/python3.13.exe

This was confirmed from the Python Environments extension's workspace selection
and its environment-change log, not the legacy Python extension's stale 3.14
auto-discovery entry. It resolves to Microsoft Store Python 3.13.14. Its user
packages remain under the PythonSoftwareFoundation.Python.3.13 local package
directory. PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118 and RTX 4060 Laptop CUDA
support are intact. The workspace .vscode settings were left unchanged and were
not included in the project's commits.

python.exe, python3.exe and python3.13.exe in WindowsApps are entry points to this
same installation, not three different environments. Do not remove these aliases
or the selected installation's packages to simplify the interpreter list.

## Cleanup completed

- Uninstalled uv-managed CPython 3.11.16 and 3.13.15 using uv python uninstall.
- The uv-managed python3.13.exe launcher in .local/bin was also removed by uv.
- After explicit user approval, uninstalled free-claude-code 5.15.0 and all 12
  fcc-* entry points, then uninstalled uv-managed CPython 3.14.0 and its launcher.
- Verified the tool environment, 3.14 installation and python3.14.exe launcher
  are absent. uv tool list is empty, and uv python list --only-installed now
  finds only the preserved Microsoft Store Python 3.13.14 installation.
- uv cache prune removed temporary cached environments/unused entries and
  reported approximately 1.1 GiB of cached files removed.
- No project source, datasets, model results or selected-interpreter packages
  were deleted during this system-wide environment audit.

## Remaining exceptions and incomplete cleanup

free-claude-code previously required Python >=3.14 and could not be migrated to
the selected Python 3.13. The user subsequently approved removing both the tool
and its 3.14 runtime. Neither remains installed, and neither was a training
dependency. Existing application configuration outside the managed tool
environment was not removed.

LM Studio's embedded Python/runtime is application-owned, not a separately
selected development environment, and was not removed.

Recursive removal of the following standalone venv directories was refused by
this session's command policy. These directories are therefore NOT deleted:

    C:/Users/anhna/.venv
    D:/TestOpus/.venv
    D:/kagglegemma/.venv
    D:/Kaggle999/enveda-CASMI26-molecule-id-mass-spectra/.venv
    D:/DataMakeTrue/internRAG/.venv

An orphaned Lib-only directory also remains at:

    C:/Users/anhna/AppData/Local/Programs/Python/Python313

The kagglegemma/internRAG venvs referenced the removed uv-managed 3.13 runtime;
they are now stale and must not be selected for training. Their project source
is untouched. No alternate shell or Python deletion script was used to bypass
the policy refusal. System-wide cleanup must not be described as fully complete.

## Verification

- All 22 tests pass, including model budgets, Windows worker reproducibility,
  corrupted-checkpoint recovery and exact CPU SGD/Adam resume equivalence.
- A short real CIFAR-100 CUDA run used batch size 128 and approximately 1.86 GiB
  peak allocated GPU memory without AMP skips. It stopped after batch two and
  resumed at batch three, then saved after batch four.
- Its raw/EMA weights, optimizer state, AMP scaler and partial-epoch metrics
  match an uninterrupted four-batch CUDA run exactly (tensor-by-tensor).
- These are execution/resume checks only, not evidence of 98%/95% accuracy.
- Verification runs are isolated in outputs/rebuilt_verification. Normal
  python train.py uses outputs/rebuilt and starts the intended two-dataset queue.
