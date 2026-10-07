"""Recompute of the k-dependent MRE outputs from a zarr database (item C3).

Uses the golden reference's compiled zarr (made before C1, so the C1 settings are added here).
"""
import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.drivers.mre_recompute import (
    GLOBAL_OUTPUTS, K_OUTPUTS, check_recompute_inputs, dwdt_curves, recompute_mre_hyperparams)

ZARR = 'work/multi_run_zarr/compiled_combined_xr.zarr'


@pytest.fixture(scope='module')
def db(golden_dir, tmp_path_factory):
    src = xr.open_zarr(golden_dir / ZARR).load()
    n = src.sizes['run_idx']
    src = src.assign(wd_static=('run_idx', np.zeros(n, bool)), force_lmfp=('run_idx', np.zeros(n, bool)),
                     psi_pedestal_cutoff=('run_idx', np.full(n, 0.9)))  # settings of the golden multi_run case
    path = tmp_path_factory.mktemp('db') / 'db.zarr'
    src.to_zarr(path)
    return src, path


def test_inputs_present(db):
    assert check_recompute_inputs(db[0]) == []
    assert 'wd_static' in check_recompute_inputs(db[0].drop_vars('wd_static'))


def test_recompute_at_stored_k_equals_stored(db, tmp_path):
    src, path = db
    new = xr.open_zarr(recompute_mre_hyperparams(path, 0.8227, 1.7, 0.6, out_path=str(tmp_path / 'o.zarr'))).load()
    new = new.squeeze(['k0', 'k1', 'C0'], drop=True)
    for v in K_OUTPUTS + GLOBAL_OUTPUTS:
        a, b = src[v], new[v].transpose(*src[v].dims)
        if v.startswith('X0_on_'):  # stored before C2: only filled on rdcon
            a, b = a.sel(code='rdcon'), b.sel(code='rdcon')
        np.testing.assert_array_equal(a.values, b.values, err_msg=v)


@pytest.mark.parallel
def test_parallel_equals_serial(db, tmp_path):
    k = dict(k0=[0.8, 1.0], k1=[1.2, 1.7], C0=[0.6, 0.8])
    one = xr.open_zarr(recompute_mre_hyperparams(db[1], out_path=str(tmp_path / 's.zarr'), **k)).load()
    two = xr.open_zarr(recompute_mre_hyperparams(db[1], out_path=str(tmp_path / 'p.zarr'), n_workers=2,
                                                 run_chunk=1, **k)).load()
    xr.testing.assert_identical(one, two)
    np.testing.assert_allclose(one.prefac_surf.sel(k0=1.0) * 1.0, one.prefac_surf.sel(k0=0.8) * 0.8, rtol=1e-14)


def test_dwdt_curves_match_stored(db):
    src = db[0].isel(run_idx=[0], nn=[0])
    curves = dwdt_curves(src, w_bar=src.w_bar.values).squeeze(['k0', 'k1', 'C0'], drop=True)
    np.testing.assert_array_equal(curves.transpose(*src.dwdt_surf.dims).values, src.dwdt_surf.values)
