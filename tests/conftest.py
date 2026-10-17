import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".cache").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
import train
train.ensure_python()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
