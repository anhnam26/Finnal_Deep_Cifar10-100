import os
from pathlib import Path
import sys
import site

ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(ROOT / '.deps'))
sys.path.insert(0, str(ROOT / '.deps'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
