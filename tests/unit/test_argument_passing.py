"""Positional arguments whose name matches a *different* parameter of the TPS function they are passed to.

Catches swaps like te/ti: `f(..., te_keV_spline, ti_keV_spline)` into `def f(..., ti_spline, te_spline)`.
Names are compared after dropping unit/type suffixes (_keV, _spline, _xr, ...). Static (AST) check of the
package and the tests; calls to functions defined more than once are skipped.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = re.compile(r'_(kev|spline|xr|xarray|da|vals?|surf|dict|in|local)(?=_|$)')
ALLOWED = {('tests/unit/physics/test_mre_hyperparams.py', 'extract_critical_mre_factors_on_modes', 'rdcon')}


def _norm(name):
    return SUFFIXES.sub('', name.lower()).strip('_')


def _arg_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _swapped_arguments():
    pkg = sorted((ROOT / 'tearing_physics_suite').rglob('*.py'))
    defs = {}
    for path in pkg:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef):
                a = node.args
                params = [x.arg for x in a.posonlyargs + a.args if x.arg not in ('self', 'cls')]
                defs.setdefault(node.name, []).append((params, params + [x.arg for x in a.kwonlyargs]))
    found = []
    for path in pkg + sorted((ROOT / 'tests').rglob('*.py')):
        rel = str(path.relative_to(ROOT))
        for call in ast.walk(ast.parse(path.read_text())):
            if not isinstance(call, ast.Call):
                continue
            fn = getattr(call.func, 'id', None) or getattr(call.func, 'attr', None)
            if len(defs.get(fn, [])) != 1:
                continue
            positional, every = defs[fn][0]
            for param, arg in zip(positional, call.args):
                if isinstance(arg, ast.Starred):
                    break
                name = _arg_name(arg)
                if name is None or _norm(name) == _norm(param) or (rel, fn, name) in ALLOWED:
                    continue
                other = [q for q in every if _norm(q) == _norm(name)]
                if other:
                    found.append(f'{rel}:{call.lineno} {fn}: `{name}` passed as `{param}` (the function has `{other[0]}`)')
    return found


def test_no_swapped_positional_arguments():
    assert _swapped_arguments() == []


def test_detects_a_swap(tmp_path, monkeypatch):
    src = 'def f(a, ti_spline, te_spline):\n    pass\n\n\ndef g(te_keV_spline, ti_keV_spline):\n    f(1, te_keV_spline, ti_keV_spline)\n'
    pkg = tmp_path / 'tearing_physics_suite'
    pkg.mkdir()
    (pkg / 'm.py').write_text(src)
    (tmp_path / 'tests').mkdir()
    monkeypatch.setitem(globals(), 'ROOT', tmp_path)
    assert len(_swapped_arguments()) == 2
