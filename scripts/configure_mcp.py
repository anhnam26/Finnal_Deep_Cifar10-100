"""Write project config only; never register a global MCP or edit trust."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / '.codex' / 'config.toml'
target.parent.mkdir(exist_ok=True)
database = ROOT / 'experiments' / 'optuna.db'
database.parent.mkdir(exist_ok=True)
quote = lambda path: json.dumps(str(path).replace('\\', '/'))
content = ('# Generated for this project and the existing selected Python.\n'
           '[mcp_servers.optuna]\n'
           f'command = {quote(sys.executable)}\n'
           f'args = [{quote(ROOT / "scripts" / "start_optuna_mcp.py")}, "--storage", '
           f'{json.dumps("sqlite:///" + database.as_posix())}]\n'
           f'cwd = {quote(ROOT)}\n'
           'startup_timeout_sec = 30\n')
target.write_text(content, encoding='utf-8')
print(target)
