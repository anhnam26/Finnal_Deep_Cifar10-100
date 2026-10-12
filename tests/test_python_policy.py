import sys
import pytest


def test_selected_base_interpreter_is_allowed():
    from src.python_policy import ensure_selected_python
    ensure_selected_python()


@pytest.mark.parametrize('changed', ['executable', 'prefix', 'version'])
def test_alternate_interpreter_or_environment_is_rejected(monkeypatch, changed):
    from src.python_policy import ensure_selected_python
    if changed == 'executable':
        monkeypatch.setattr(sys, 'executable', r'D:\OtherProject\.venv\Scripts\python.exe')
    elif changed == 'prefix':
        monkeypatch.setattr(sys, 'prefix', r'D:\OtherProject\.venv')
    else:
        monkeypatch.setattr(sys, 'version_info', (3, 12, 0))
    with pytest.raises(RuntimeError, match='python3.13.exe'):
        ensure_selected_python()
