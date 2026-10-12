# Repository instructions

For CIFAR research, read `.agents/skills/cifar-sota-architect/SKILL.md`.
`Final exam.docx` governs the <=6M parameter and <1G FLOP limits for model L.
Never invent accuracy or confuse validation, synthetic/subset checks and test.
Keep official test sealed during architecture/HPO/recipe/model selection. Freeze
the final recipe and epoch count, retrain on all training images, then test once.
Record source/config/split/environment/seed and preserve resumable experiments.

Use ONLY `C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe`,
the user's existing Microsoft Store Python 3.13, and its own installed packages.
Follow the user-wide Python policy in `C:\Users\anhna\.codex\AGENTS.md`.
Never create/download/use another interpreter, venv, conda, pipx/uv/Poetry tool
environment, `.deps` or alternate third-party package directory. Do not change
the IDE interpreter. Install missing libraries directly into this one environment
using the selected executable with `-m pip`; its normal user-site is permitted.
Preserve working CUDA/PyTorch. Use `python scripts/setup_environment.py` when
dependencies are missing. Global skills/plugins and Codex configuration remain
unchanged unless explicitly requested by the user. Reuse `datasets/`; generated
artifacts stay in this repository. Do not automatically launch long training.

Each experiment retains only `last.pt` and `best.pt`. Last is the last completed
training/validation epoch; best uses validation accuracy. No previous.pt or
numbered checkpoints. Changed config/source/runtime must not silently overwrite
or abandon an existing run: reject resume and explain `--run-name`.

## Git commit rules

After each independent meaningful unit, validate and immediately make a local
commit using the installed `git-commit` skill. Use small Conventional Commits
(feat/fix/refactor/docs/test/chore/perf/build/ci) with clear English descriptions.
Before every commit inspect status and diff, run relevant checks, and stage
only that unit's files. Never blindly `git add .` / `git add -A`, commit secrets
or `.env`, or include unrelated user changes.
Commit on the current working branch. Never automatically push, force-push,
merge main/master, rebase, reset or amend without explicit permission. Report
failed hooks/commits instead of bypassing them. Report commit hashes at handoff.
