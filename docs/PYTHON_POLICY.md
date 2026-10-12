# One Python installation for every project

The user explicitly replaced the earlier project-isolation dependency policy
on October 13, 2026. The persistent global rule is in
`C:/Users/anhna/.codex/AGENTS.md`; the repository AGENTS.md and CIFAR skill follow
the same rule. Existing Git commit rules are preserved.

Allowed executable:
`C:/Users/anhna/AppData/Local/Microsoft/WindowsApps/python3.13.exe`.
It resolves to the existing Microsoft Store Python 3.13 application. Its normal
Python313 user-site package directory is part of this same environment, not a
new venv. Missing libraries must be installed there through this executable's
`-m pip`. No new interpreter, venv, `.deps`, isolated tool environment or
alternate package destination is permitted.

Codex reads global AGENTS.md when a run/session starts. This session follows the
user's explicit instruction immediately; future sessions inherit the saved rule.
Official instruction discovery:
https://learn.chatgpt.com/docs/agent-configuration/agents-md

The current project migration removes `.deps` import hooks, guards entry points
against another interpreter/venv, and uses the existing CUDA torch/torchvision.
Old isolation reports describe the prior setup and are historical, not the
current dependency policy. No model architecture/training recipe is changed.
