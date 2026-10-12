# Repository instructions

For CIFAR research, read `.agents/skills/cifar-sota-architect/SKILL.md`.
`Final exam.docx` governs the <=6M parameter and <1G FLOP limits for model L.
Never invent accuracy or confuse validation, synthetic/subset checks and test.
Keep official test sealed during architecture/HPO/recipe/model selection. Freeze
the final recipe and epoch count, retrain on all training images, then test once.
Record source/config/split/environment/seed and preserve resumable experiments.

Use the user's existing Microsoft Store Python 3.13. Do not download another
Python interpreter, create `.venv`, change the selected IDE interpreter, install
global packages, install global skills/plugins or edit `~/.codex` / `~/.agents`.
Missing dependencies belong in `.deps`; entry points add it to their import
path. Use `python scripts/setup_environment.py` only when dependencies are
missing. Reuse the existing dataset under `datasets/`. Generated artifacts stay
under this repository. Never automatically launch long training during setup.

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
