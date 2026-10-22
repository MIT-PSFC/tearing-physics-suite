"""GPEC_resistive_calculation with a fake rdcon (no Fortran): staging, run, read, save."""
import os
import resource
import stat

import pytest
import xarray as xr

from tearing_physics_suite.wrappers.gpec import GPEC_resistive_calculation
from tearing_physics_suite.wrappers import run_codes
from tearing_physics_suite.wrappers.gpec_common import executable, peak_gib, run_gpec_code
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


@pytest.mark.parametrize('save_terminal_output', [True, False])
def test_killed_run_is_recorded(tmp_path, monkeypatch, save_terminal_output):
    """A code killed by a signal (e.g. out of memory) is a failed run in CODE_KILLED; a plain failure is not."""
    from tearing_physics_suite.utils import CODE_KILLED
    CODE_KILLED.clear()
    monkeypatch.chdir(tmp_path)
    exe = tmp_path / 'stride'
    for body in ('exit 3', 'kill -9 $$'):
        exe.write_text(f'#!/bin/sh\n{body}\n')
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        assert run_gpec_code('stride', str(tmp_path), 4, save_terminal_output=save_terminal_output) == (None, False)
    assert CODE_KILLED == {'stride': [4]}
    CODE_KILLED.clear()


def test_peak_gib_matches_measured_peaks():
    """The predictor against measured peaks (GiB): STRIDE n=4 at mpert 77 / 44 surfaces 7.7, at 73 / 38 5.7 (CG, W4),
    banded 1.80 at 77 (CCG); RDCON's Delta' at 69 / 44 37.9 (CCG, 204323_2306 d0), and within W4's 28.7-33.6 GiB check
    steps at 65 / 38."""
    assert peak_gib('stride', 77, 44) == pytest.approx(7.7, abs=0.2)
    assert 1.80 <= peak_gib('stride', 77, 44, banded=True) <= 1.9
    assert peak_gib('stride', 73, 38) == pytest.approx(5.7, abs=0.2)
    assert peak_gib('rdcon', 69, 44, nx=256) == pytest.approx(38.0, abs=0.1)
    assert 28.7 <= peak_gib('rdcon', 65, 38, nx=256) <= 33.6


@pytest.mark.parametrize('mem, banded, ran', [(2, 'f', False), (4, 'f', True), (2, 't', True)])
def test_stride_not_started_when_predicted_too_big(tmp_path, eq_file, monkeypatch, mem, banded, ran):
    """Before STRIDE, its peak is predicted from RDCON's surface pass (fixture: mpert 27 + delta_mhigh 8, 5 surfaces:
    2.02 GiB, banded 0.62); in a session too small for it, STRIDE is not started and is recorded killed and skipped."""
    from tearing_physics_suite.utils import CODE_KILLED, CODE_SKIPPED
    gpec_dir = _fake_gpec(tmp_path)
    exe = tmp_path / 'GPEC' / 'stride' / 'stride'
    exe.parent.mkdir()
    exe.write_text(f'#!/bin/sh\ntouch {tmp_path}/stride_ran\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('TPS_MEM_GIB', str(mem))
    CODE_KILLED.clear()
    CODE_SKIPPED.clear()
    GPEC_resistive_calculation(eq_file, 1, run_rdcon=True, run_stride=True, working_dir=str(tmp_path / 'work'),
                               gpec_dir=gpec_dir, save_terminal_output=True, dprime_banded=banded)
    assert (tmp_path / 'stride_ran').exists() == ran
    assert CODE_SKIPPED == CODE_KILLED == ({} if ran else {'stride': [1]})
    CODE_KILLED.clear()
    CODE_SKIPPED.clear()


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
