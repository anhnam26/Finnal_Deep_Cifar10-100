"""Install missing libraries into the one permitted Python environment."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import site
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.python_policy import SELECTED_PYTHON, ensure_selected_python

REPORTS = ROOT / 'reports' / 'generated'
PROTECTED = ('torch', 'torchvision', 'numpy')


def inventory():
    return {'executable': sys.executable, 'prefix': sys.prefix,
            'base_prefix': sys.base_prefix, 'user_site': site.getusersitepackages(),
            'packages': {d.metadata['Name'].lower().replace('_', '-'): d.version
                         for d in importlib.metadata.distributions()}}


def verify_imports():
    code = """
import json, sys, site
from pathlib import Path
import torch, torchvision, numpy, timm, optuna, optuna_mcp, mcp, fpdf
modules = (torch, torchvision, numpy, timm, optuna, optuna_mcp, mcp, fpdf)
roots = [Path(p).resolve() for p in [site.getusersitepackages(), *site.getsitepackages()]]
imports = {module.__name__: str(Path(module.__file__).resolve()) for module in modules}
assert sys.prefix == sys.base_prefix
for path in imports.values():
 assert any(Path(path).is_relative_to(root) for root in roots), path
print(json.dumps({'executable':sys.executable, 'prefix':sys.prefix,
 'base_prefix':sys.base_prefix, 'imports':imports,
 'torch':torch.__version__, 'torchvision':torchvision.__version__,
 'numpy':numpy.__version__, 'timm':timm.__version__, 'optuna':optuna.__version__,
 'cuda_available':torch.cuda.is_available(),
 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))
"""
    output = subprocess.check_output([str(SELECTED_PYTHON), '-X', 'utf8', '-c', code],
                                     cwd=ROOT, text=True, encoding='utf-8')
    return json.loads(output)


def main():
    ensure_selected_python()
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-after', action='store_true')
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    before_path = REPORTS / 'shared-python-before.json'
    if args.audit_after:
        before = json.loads(before_path.read_text(encoding='utf-8'))
        after = inventory()
        old, new = before['packages'], after['packages']
        audit = {'runtime': verify_imports(),
                 'added_packages': {k: v for k, v in new.items() if k not in old},
                 'changed_packages': {k: {'before': old[k], 'after': new[k]}
                                      for k in old.keys() & new.keys() if old[k] != new[k]},
                 'removed_packages': sorted(old.keys() - new.keys()),
                 'protected_packages_unchanged': all(old.get(k) == new.get(k) for k in PROTECTED),
                 'policy': 'one existing Microsoft Store Python 3.13; direct package installation'}
        (REPORTS / 'shared-python-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print(json.dumps(audit, indent=2))
        return
    before = inventory()
    if not before_path.exists():
        before_path.write_text(json.dumps(before, indent=2), encoding='utf-8')
    if not all(name in before['packages'] for name in PROTECTED):
        raise RuntimeError('Existing CUDA torch/torchvision/numpy must be present in the selected Python')
    constraints = REPORTS / 'shared-python-constraints.txt'
    constraints.write_text('\n'.join(f'{name}=={before["packages"][name]}' for name in PROTECTED)+'\n', encoding='utf-8')
    env = dict(os.environ, PIP_CACHE_DIR=str(ROOT / '.cache' / 'pip'),
               PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run([str(SELECTED_PYTHON), '-m', 'pip', 'install',
                    '-r', str(ROOT / 'requirements.txt'), '--constraint', str(constraints),
                    '--upgrade-strategy', 'only-if-needed', '--only-binary=:all:'],
                   cwd=ROOT, env=env, check=True)
    verification = verify_imports()
    if any(verification[name] != before['packages'][name] for name in PROTECTED):
        raise RuntimeError('Protected CUDA/PyTorch/NumPy versions changed unexpectedly')
    from packaging.requirements import Requirement
    lines = [line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
             if line.strip() and not line.startswith('#')]
    versions = [f'{Requirement(line).name}=={importlib.metadata.version(Requirement(line).name)}' for line in lines]
    (ROOT / 'requirements-lock.txt').write_text('\n'.join(versions)+'\n', encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
