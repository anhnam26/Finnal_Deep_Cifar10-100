"""Local Optuna MCP, using the user's current Python and .deps."""
import os
from pathlib import Path
import site
import sys

ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(ROOT / '.deps'))
sys.path.insert(0, str(ROOT / '.deps'))
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))

if __name__ == '__main__':
    from optuna_mcp.server import main
    main()
