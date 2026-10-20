"""GPEC_resistive_calculation with a fake rdcon (no Fortran): staging, run, read, save."""
import os
import resource
import stat

import pytest
import xarray as xr

from tearing_physics_suite.wrappers.gpec import GPEC_resistive_calculation
from tearing_physics_suite.wrappers import run_codes
from tearing_physics_suite.wrappers.gpec_common import executable, run_gpec_code
from tearing_physics_suite.wrappers.run_codes import _rel_diff, run_resistive_calculation

FIX = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fixtures', 'g147131.02300_DIIID_KEFIT_rdcon_n1.nc')


def _fake_gpec(tmp_path, exit_code=0):
    d = tmp_path / 'GPEC' / 'rdcon'
    d.mkdir(parents=True)
    exe = d / 'rdcon'
    exe.write_text(f'#!/bin/sh\ncp {os.path.abspath(FIX)} rdcon_output_n1.nc\nexit {exit_code}\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(tmp_path / 'GPEC')


def test_executable_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        executable(str(tmp_path), 'stride')


def test_rdcon_only_runs_without_stride_executable(tmp_path, eq_file):
    gpec_dir = _fake_gpec(tmp_path)
    out = tmp_path / 'out'
    rd, st, rd_ran, st_ran, inputs = GPEC_resistive_calculation(
        eq_file, 1, run_rdcon=True, run_stride=False, working_dir=str(tmp_path / 'work'), gpec_dir=gpec_dir,
        output_location=str(out), output_prefix='p_', save_input=True, save_terminal_output=True)
    assert rd_ran and not st_ran and st is None
    ref = xr.load_dataset(FIX)
    ref['Delta_prime'].attrs['long_name'] = 'Delta prime matrix'  # relabelled on read (item F)
    xr.testing.assert_identical(rd.load(), ref)
    assert (out / 'p_g147131.02300_DIIID_KEFIT_rdcon_n1.nc').exists()
    assert (out / 'p_g147131.02300_DIIID_KEFIT_rdcon_terminal_output_n1.txt').exists()
    assert (out / 'p_g147131.02300_DIIID_KEFIT_rdcon_stride_input_n1.pkl').exists()
    assert inputs['nn'] == 1 and (tmp_path / 'work' / 'rdcon.in').exists() and not (tmp_path / 'work' / 'stride').exists()


@pytest.mark.parametrize('code', ['stride', 'rdcon'])
def test_stride_large_stack_rdcon_one_thread(tmp_path, monkeypatch, code):
    """STRIDE (OpenMP) gets OMP_STACKSIZE 1G and a 1 GB stack limit; RDCON (serial) gets OMP_NUM_THREADS 1."""
    monkeypatch.delenv('OMP_STACKSIZE', raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    monkeypatch.chdir(tmp_path)
    exe = tmp_path / code
    exe.write_text('#!/bin/sh\necho "$(ulimit -s) $OMP_NUM_THREADS $OMP_STACKSIZE" > env.txt\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    assert run_gpec_code(code, str(tmp_path), 1) == (None, False)
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    stack = min(1 << 30, hard) if code == 'stride' and hard != resource.RLIM_INFINITY else 1 << 30 if code == 'stride' else soft
    want = 'unlimited' if stack == resource.RLIM_INFINITY else str(stack // 1024)
    assert (tmp_path / 'env.txt').read_text().split() == [want] + (['8', '1G'] if code == 'stride' else ['1'])


def test_failed_run_returns_none(tmp_path, eq_file):
    rd, _, rd_ran, _, _ = GPEC_resistive_calculation(
        eq_file, 1, run_rdcon=True, working_dir=str(tmp_path / 'work'), gpec_dir=_fake_gpec(tmp_path, exit_code=3))
    assert rd is None and not rd_ran


def test_make_results_dir_false_saves_nothing(tmp_path, eq_file):
    """make_results_dir reaches GPEC_resistive_calculation: no output_location, no copies in working_dir."""
    work = tmp_path / 'work'
    res = run_resistive_calculation(eq_file, 1, run_stride=False, run_pest3=False, make_results_dir=False,
                                    working_dir=str(work), gpec_dir=_fake_gpec(tmp_path), verbose=False)
    assert res.rdcon_ran
    assert not list(work.glob('*_rdcon_n1.nc')) and not list(work.glob('*_rdcon_stride_input_n1.pkl'))


def test_jgpec_skipped_when_rdcon_fails(tmp_path, eq_file, monkeypatch):
    def no_jgpec(*a, **k):
        raise AssertionError('jGPEC must not run without RDCON')
    monkeypatch.setattr(run_codes, 'run_jgpec_solvers', no_jgpec)
    res = run_resistive_calculation(eq_file, 1, run_stride=False, run_pest3=False, run_jgpec=True,
                                    working_dir=str(tmp_path / 'work'), gpec_dir=_fake_gpec(tmp_path, exit_code=3),
                                    verbose=False)
    assert not res.rdcon_ran and res.jgpec_xrs == {}


def test_rel_diff():
    assert _rel_diff(2.0, 2.0) == 0.0 and _rel_diff(0.0, 0.0) == 0.0
    assert _rel_diff(1.0, 3.0) == 1.0 and _rel_diff(0.0, 1e-3) == 2.0


def test_rdcon_run_is_timed(tmp_path, eq_file):
    """GPEC_resistive_calculation records each code's wall time under '<code>_n<n>' (utils.CODE_WALL_S)."""
    from tearing_physics_suite.utils import CODE_WALL_S
    CODE_WALL_S.clear()
    GPEC_resistive_calculation(eq_file, 1, run_rdcon=True, working_dir=str(tmp_path / 'work'), gpec_dir=_fake_gpec(tmp_path))
    assert list(CODE_WALL_S) == ['rdcon_n1'] and CODE_WALL_S['rdcon_n1'] > 0
    CODE_WALL_S.clear()
