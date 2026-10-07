# Unit tests for PEST3_wrappers that need no PEST3 binary (pytest).
import os
import stat

import numpy as np
import xarray as xr
import pytest

os.environ.setdefault('TPSHOME', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tearing_physics_suite.PEST3_wrappers as p3w


def _fake_pest3(tmp_path, body='exit 0'):
    """Directory holding a stand-in pest3x shell script, plus a dummy equilibrium file."""
    exe_dir = tmp_path / 'bin'
    exe_dir.mkdir()
    exe = exe_dir / 'pest3x'
    exe.write_text('#!/bin/sh\n' + body + '\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    eq = tmp_path / 'g000001.00001'
    eq.write_text('dummy')
    return str(exe_dir), str(eq)


def test_stale_pest3_nc_not_read(tmp_path):
    """A pest3.nc left by an earlier run must not be read when the new run writes none."""
    exe_dir, eq = _fake_pest3(tmp_path)
    work = tmp_path / 'work'
    work.mkdir()
    xr.Dataset({'qa': ('x', [1.0, 2.0])}).to_netcdf(work / 'pest3.nc')

    cwd = os.getcwd()
    try:
        pest3_xr, pest3_ran, _ = p3w.PEST3_resistive_calculation(
            eq, 1, working_dir=str(work), pest3_dir=exe_dir, verbose=False,
            save_input=False, clean_netcdf=False, vacuum_source_pest='pest3')
    finally:
        os.chdir(cwd)
    assert pest3_xr is None
    assert not pest3_ran
    assert not (work / 'pest3.nc').exists()


def _run_loop(monkeypatch, qa_last, ran=True, truncimax=4):
    """Run pest3_special_truncation_loop with PEST3 calls replaced by a fixed qa result."""
    monkeypatch.setattr(p3w, 'pest3_special_truncation_single', lambda *a, **k: (0.9, True))

    def fake_run(*args, **kwargs):
        if not ran:
            return None, False, {}
        return xr.Dataset({'qa': ('x', [1.0, qa_last])}), True, {}
    monkeypatch.setattr(p3w, 'PEST3_resistive_calculation', fake_run)
    return p3w.pest3_special_truncation_loop('eq', 1, 5.0, {}, verbose=False, truncimax=truncimax)


def test_truncation_loop_not_converged(monkeypatch):
    """No convergence returns the last lower bracket and False (was a NameError)."""
    psihigh, ok = _run_loop(monkeypatch, qa_last=3.0, truncimax=4)
    assert not ok
    # qa stays below qlim, so the lower bracket rises from 0.9 by bisection toward 1.
    assert psihigh == pytest.approx(1.0 - 0.1 / 2**4)


def test_truncation_loop_converged(monkeypatch):
    psihigh, ok = _run_loop(monkeypatch, qa_last=5.005)
    assert ok
    assert psihigh == pytest.approx(0.9)


def test_truncation_loop_failed_run(monkeypatch):
    """A failed PEST3 run returns (0, False) instead of indexing None."""
    assert _run_loop(monkeypatch, qa_last=3.0, ran=False) == (0, False)


def _run_gpec_vac(tmp_path, body, vac_in=True):
    exe_dir, eq = _fake_pest3(tmp_path, body)
    work = tmp_path / 'work'
    work.mkdir()
    if vac_in:
        (work / 'vac.in').write_text('&vac\n/\n')
    cwd = os.getcwd()
    try:
        return p3w.PEST3_resistive_calculation(
            eq, 1, working_dir=str(work), pest3_dir=exe_dir, verbose=False,
            save_input=False, clean_netcdf=False, mthvac_pest=480), work
    finally:
        os.chdir(cwd)


def test_gpec_vacuum_needs_gpec_build(tmp_path):
    """The default GPEC vacuum refuses a pest3x linked without it."""
    with pytest.raises(RuntimeError, match='without GPEC vacuum'):
        _run_gpec_vac(tmp_path, 'exit 0')


def test_gpec_vacuum_needs_vac_in(tmp_path):
    with pytest.raises(FileNotFoundError, match='vac.in'):
        _run_gpec_vac(tmp_path, '# vacuum_mod_MOD_mscvac\nexit 0', vac_in=False)


def test_gpec_vacuum_passes_V_flag(tmp_path):
    """-V<mthvac_pest> reaches pest3x, and the input dict records the source."""
    (_, _, inputs), work = _run_gpec_vac(tmp_path, '# vacuum_mod_MOD_mscvac\necho "$@" > args.txt')
    assert ' -V480 ' in ' ' + (work / 'args.txt').read_text() + ' '
    assert inputs['vacuum_source_pest'] == 'gpec'
    assert inputs['mthvac_pest'] == 480
