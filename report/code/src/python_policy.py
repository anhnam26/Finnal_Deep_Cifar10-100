"""Only the user's existing Microsoft Store Python 3.13 is permitted."""
import os
from pathlib import Path
import sys

SELECTED_PYTHON = Path(r'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe')
RESOLVED_PYTHON = SELECTED_PYTHON.parent / 'PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0' / 'python.exe'


def ensure_selected_python():
    def normalized(path):
        return os.path.normcase(os.path.abspath(str(path)))
    allowed = {normalized(SELECTED_PYTHON), normalized(RESOLVED_PYTHON)}
    if (normalized(sys.executable) not in allowed
            or normalized(sys.prefix) != normalized(sys.base_prefix)
            or sys.version_info[:2] != (3, 13)):
        raise RuntimeError(
            'Use only the existing Python environment: '
            f'"{SELECTED_PYTHON}". Current interpreter: {sys.executable}; '
            'virtual/alternate Python environments are not permitted.')
