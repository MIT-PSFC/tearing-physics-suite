"""physics/ and wrappers/ must not import upward (layering rule in CLAUDE.md)."""
import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / 'tearing_physics_suite'
FORBIDDEN = {'physics': ('wrappers', 'drivers'), 'wrappers': ('physics', 'drivers')}


def _tps_imports(path):
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith('tearing_physics_suite'):
            yield n.module, n.lineno
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith('tearing_physics_suite'):
                    yield a.name, n.lineno


@pytest.mark.parametrize('layer', FORBIDDEN)
def test_no_upward_imports(layer):
    bad = [f'{p.relative_to(PKG)}:{ln} imports {m}'
           for p in (PKG / layer).rglob('*.py') for m, ln in _tps_imports(p)
           if any(m.startswith(f'tearing_physics_suite.{f}') for f in FORBIDDEN[layer])]
    assert not bad, bad


def test_package_imports_without_tpshome():
    """TPSHOME is read lazily (utils.tps_home), so every module imports without it."""
    import os
    import subprocess
    import sys
    code = ("import importlib, pkgutil, tearing_physics_suite as t\n"
            "for m in pkgutil.walk_packages(t.__path__, 'tearing_physics_suite.'):\n"
            "    if 'build_tearing' not in m.name: importlib.import_module(m.name)\n")
    env = {k: v for k, v in os.environ.items() if k != 'TPSHOME'}
    env['PYTHONPATH'] = str(PKG.parent)
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
