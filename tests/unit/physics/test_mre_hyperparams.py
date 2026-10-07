"""k0, k1, C0 as dims of the MRE outputs (item C2)."""
import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.physics.combine import combine_codes
from tearing_physics_suite.physics.global_quantities import global_mre_quantities
from tearing_physics_suite.physics.mre_model import extract_critical_mre_factors_on_modes

GRID = dict(k0=[0.8227, 1.0], k1=[1.2, 1.7], C0=[0.6, 0.8])
K_VARS = ['w_marg_surf', 'w_sat_surf', 'w_max_loc_surf', 'dwdtau_max_surf', 'wd_at_marg_surf',
          'X0_on_w_marg_surf', 'X0_on_wd_at_marg_surf']


def _mre(code_datasets, **k):
    rd = extract_critical_mre_factors_on_modes(code_datasets['rdcon'], code_datasets['rdcon'], **k)
    out = {'rdcon': rd}
    for c in ('stride', 'pest3'):
        out[c] = extract_critical_mre_factors_on_modes(code_datasets[c], rd, **k)
    combined, dropped = combine_codes(out, nn=1)
    assert not dropped
    return combined


@pytest.fixture(scope='module')
def grid(code_datasets):
    return _mre(code_datasets, **GRID)


def test_dims(grid):
    for v in K_VARS:
        assert grid[v].dims[-2:] == ('k1', 'C0'), v
    assert grid.prefac_surf.dims[-1] == 'k0'
    assert 'k1' not in grid.dwdt_surf.dims and 'k0' not in grid.dwdt_surf.dims


def test_grid_slice_equals_scalar_run(code_datasets, grid):
    one = _mre(code_datasets, k0=1.0, k1=1.2, C0=0.8)
    for v in K_VARS:
        xr.testing.assert_identical(grid[v].sel(k1=[1.2], C0=[0.8]), one[v])
    xr.testing.assert_identical(grid.prefac_surf.sel(k0=[1.0]), one.prefac_surf)


def test_k0_only_rescales_prefac(grid):
    p = grid.prefac_surf
    np.testing.assert_allclose(p.sel(k0=1.0) * 1.0, p.sel(k0=0.8227) * 0.8227, rtol=1e-14)


def test_x0_ratios_filled_for_every_code(grid):
    for c in grid.code.values:
        assert np.isfinite(grid.X0_on_w_marg_surf.sel(code=c)).any(), c


def test_global_ranks_per_k(grid):
    g = global_mre_quantities(xr.concat([grid, grid.assign_coords(nn=[2])], dim='nn'), psi_pedestal_cutoff=0.8)
    for k1 in GRID['k1']:
        for C0 in GRID['C0']:
            one = global_mre_quantities(
                xr.concat([grid, grid.assign_coords(nn=[2])], dim='nn').sel(k1=[k1], C0=[C0]), psi_pedestal_cutoff=0.8)
            for v in ('min_w_marg_rank', 'max_dwdtau_rank', 'min_w_marg_allsurf'):
                xr.testing.assert_identical(g[v].sel(k1=[k1], C0=[C0]), one[v])
    assert np.nanmax(g.min_w_marg_rank.sel(k1=1.2, C0=0.6, code='rdcon', Delta_prime_type='full coupled')) <= 2 * 5
