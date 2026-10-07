"""jGPEC Delta' (both solvers) against RDCON on g147131, n=1, through run_resistive_calculation and the pipeline."""
import numpy as np
import pytest

from tearing_physics_suite.physics.combine import compile_xarrays
from tearing_physics_suite.wrappers.run_codes import run_resistive_calculation

pytestmark = [pytest.mark.julia, pytest.mark.fortran]
COMMON = dict(run_rdcon=True, run_stride=False, run_pest3=False, run_jgpec=True, etol=1e-10, verbose=False)
RTOL = 0.03  # set from the first run (2/1: Riccati 7.45, Galerkin 7.43 vs RDCON 7.36 no wall); needs sign-off


@pytest.fixture(scope='module')
def nowall(tmp_path_factory, eq_file):
    d = tmp_path_factory.mktemp('jgpec_nowall')
    return run_resistive_calculation(eq_file, 1, working_dir=str(d / 'work'), vac_flag='t',
                                     jgpec_solvers=('galerkin', 'riccati'), output_location=str(d), **COMMON)


@pytest.fixture(scope='module')
def wall(tmp_path_factory, eq_file):
    d = tmp_path_factory.mktemp('jgpec_wall')
    return run_resistive_calculation(eq_file, 1, working_dir=str(d / 'work'), vac_flag='f',
                                     jgpec_solvers=('galerkin', 'riccati'), **COMMON)


def _dp21(ds):
    return float(ds.Delta_prime.sel(i=0).isel(r=0, r_prime=0))


@pytest.mark.parametrize('code', ['jGPEC_galerkin', 'jGPEC_riccati'])
def test_nowall_dp21(nowall, code):
    assert nowall.rdcon_ran and nowall.jgpec_ran[code]
    assert np.isclose(_dp21(nowall.jgpec_xrs[code]), _dp21(nowall.rdcon_xr), rtol=RTOL)


def test_wall_galerkin_dp21_and_riccati_nan(wall):
    assert wall.jgpec_ran == {'jGPEC_galerkin': True, 'jGPEC_riccati': False}
    assert np.isclose(_dp21(wall.jgpec_xrs['jGPEC_galerkin']), _dp21(wall.rdcon_xr), rtol=RTOL)
    assert wall.jgpec_xrs['jGPEC_riccati'].Delta_prime.isnull().all()


def test_orientation_matches_rdcon(nowall):
    """Off-diagonal Delta' entries sit where RDCON's do (h5 transposition handled by the reader)."""
    rd = nowall.rdcon_xr.Delta_prime.sel(i=0).values
    for code in ('jGPEC_galerkin', 'jGPEC_riccati'):
        jg = nowall.jgpec_xrs[code].Delta_prime.sel(i=0).values
        for a, b in ((0, 1), (1, 0), (0, 2)):
            assert abs(jg[a, b] - rd[a, b]) < abs(jg[b, a] - rd[a, b]), (code, a, b)


@pytest.mark.xfail(strict=True, reason="jGPEC Galerkin, no wall: the 4/1 row (psi_n 0.93) differs from RDCON and "
                                       "Riccati (e.g. [4/1, 2/1] -12.9 vs 3.6 and 3.2). Cause: unscaled banded LU in "
                                       "jGPEC galerkin_solve (diagonal spans ~1e-15..1e10); see META_PLAN 5")
def test_galerkin_nowall_41_row(nowall):
    rd = nowall.rdcon_xr.Delta_prime.sel(i=0).values
    jg = nowall.jgpec_xrs['jGPEC_galerkin'].Delta_prime.sel(i=0).values
    assert np.allclose(jg[2, :3], rd[2, :3], rtol=0.3)


def test_combined_codes(nowall):
    comb, _, _ = compile_xarrays(*nowall)
    assert list(comb.code.values) == ['rdcon', 'jGPEC_galerkin', 'jGPEC_riccati']
    np.testing.assert_array_equal(comb.r_value if 'r_value' in comb else comb.r, nowall.rdcon_xr.r)
