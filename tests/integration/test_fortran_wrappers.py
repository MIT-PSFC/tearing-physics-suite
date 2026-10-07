"""RDCON/STRIDE/PEST3 wrapper runs at full grids, with reference Delta'_21 values (n=1, g147131)."""
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tearing_physics_suite.utils import eq_stem
from tearing_physics_suite.wrappers.pest3 import pest3_special_truncation_loop
from tearing_physics_suite.wrappers.run_codes import run_resistive_calculation

pytestmark = pytest.mark.fortran
COMMON = dict(run_rdcon=True, run_stride=True, run_pest3=True, etol=1e-10, mpsi_pest=400, kband_pest=15,
              gal_flag='t', ode_flag='t', save_input=True, nx_string_pest='''-k"100 70 140 50 120"''')


def _dp21(ds):
    return float(ds.Delta_prime.isel(r=0, r_prime=0, i=0))


@pytest.fixture(scope='module')
def wall_run(tmp_path_factory, eq_file):
    d = tmp_path_factory.mktemp('wall')
    out = run_resistive_calculation(eq_file, 1, working_dir=str(d / 'work'), vac_flag='f',
                                    output_location=str(d), output_prefix='test_1',
                                    pest_match_truncation=True, **COMMON)
    return d, out


@pytest.fixture(scope='module')
def nowall_run(tmp_path_factory, eq_file):
    d = tmp_path_factory.mktemp('nowall')
    out = run_resistive_calculation(eq_file, 1, working_dir=str(d / 'work'), vac_flag='t',
                                    output_location=str(d), output_prefix='test_2', **COMMON)
    return d, out


def test_pest3_truncation_loop(eq_file, tmp_path):
    os.chdir(tmp_path)
    _, ran = pest3_special_truncation_loop(eq_file, 1, 5.2, {'psihigh_pest': 1}, debug=True,
                                           output_prefix_special='trunctest1')
    assert ran


def test_wall_all_codes_ran(wall_run):
    rd_ran, st_ran, p3_ran = wall_run[1][3:6]
    assert rd_ran and st_ran and p3_ran


@pytest.mark.parametrize('code,ref,atol', [('rdcon', 2.0, 0.05), ('stride', 2.0, 0.05), ('pest3', 2.1, 0.15)])
def test_wall_dp21(wall_run, code, ref, atol):
    ds = dict(zip(('rdcon', 'stride', 'pest3'), wall_run[1][:3]))[code]
    assert np.isclose(_dp21(ds), ref, atol=atol), _dp21(ds)


def test_wall_outputs_round_trip(wall_run, eq_file):
    d, (rd, st, p3, _, _, _, rs_in, p3_in) = wall_run[0], wall_run[1][:8]
    stem = 'test_1' + eq_stem(eq_file)
    for code, ds in (('rdcon', rd), ('stride', st), ('pest3', p3)):
        assert ds.equals(xr.open_dataset(d / f'{stem}_{code}_n1.nc')), code
    assert pd.read_pickle(d / f'{stem}_rdcon_stride_input_n1.pkl') == rs_in
    assert pd.read_pickle(d / f'{stem}_pest3_input_n1.pkl') == p3_in


@pytest.mark.parametrize('code,ref,atol', [
    ('rdcon', 7.36, 0.03), ('stride', 7.3, 0.3),
    pytest.param('pest3', 7.3, 0.03, marks=pytest.mark.xfail(
        strict=False, reason='PEST3 gives ~2.57 with no wall (ongoing_handoff.md); re-baseline after item D')),
])
def test_nowall_dp21(nowall_run, code, ref, atol):
    ds = dict(zip(('rdcon', 'stride', 'pest3'), nowall_run[1][:3]))[code]
    assert np.isclose(_dp21(ds), ref, atol=atol), _dp21(ds)
