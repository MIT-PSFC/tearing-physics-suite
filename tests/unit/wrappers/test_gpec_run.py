"""GPEC_resistive_calculation with a fake rdcon (no Fortran): staging, run, read, save."""
import os
import stat

import pytest
import xarray as xr

from tearing_physics_suite.wrappers.gpec import GPEC_resistive_calculation
from tearing_physics_suite.wrappers.gpec_common import executable

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
    xr.testing.assert_identical(rd.load(), xr.load_dataset(FIX))
    assert (out / 'p_g147131.02300_DIIID_KEFIT_rdcon_n1.nc').exists()
    assert (out / 'p_g147131.02300_DIIID_KEFIT_rdcon_terminal_output_n1.txt').exists()
    assert (out / 'p_g147131.02300_DIIID_KEFIT_rdcon_stride_input_n1.pkl').exists()
    assert inputs['nn'] == 1 and (tmp_path / 'work' / 'rdcon.in').exists() and not (tmp_path / 'work' / 'stride').exists()


def test_failed_run_returns_none(tmp_path, eq_file):
    rd, _, rd_ran, _, _ = GPEC_resistive_calculation(
        eq_file, 1, run_rdcon=True, working_dir=str(tmp_path / 'work'), gpec_dir=_fake_gpec(tmp_path, exit_code=3))
    assert rd is None and not rd_ran
