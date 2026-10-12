import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'experiments').mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from src.python_policy import ensure_selected_python
ensure_selected_python()
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
