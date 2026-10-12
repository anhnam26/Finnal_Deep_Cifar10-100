"""Project Optuna MCP, using only the user's selected Python environment."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.python_policy import ensure_selected_python
ensure_selected_python()
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))

if __name__ == '__main__':
    from optuna_mcp.server import main
    main()
