import numpy as np
import xarray as xr
from scipy.interpolate import Akima1DInterpolator, CubicSpline

from tearing_physics_suite.drivers.multi_run import DROP_KEYS, meta_dict_to_dataset
from tearing_physics_suite.drivers.zarr_store import add_to_zarr_store


def _run(r_vals, idx):
    r = np.arange(len(r_vals), dtype=float)
    return xr.Dataset({'w_marg_surf': ('r', np.asarray(r_vals, float)), 'qlim': ((), 3.0 + idx)},
                      coords={'r': r, 'run_idx': idx})


def test_add_to_zarr_store_pads_ragged_dim(tmp_path):
    z = tmp_path / 's.zarr'
    add_to_zarr_store(_run([1.0, 2.0, 3.0], 0), z, 'run_idx')
    add_to_zarr_store(_run([4.0, 5.0], 1), z, 'run_idx')
    add_to_zarr_store(_run([6.0, 7.0, 8.0, 9.0], 2), z, 'run_idx')
    ds = xr.open_zarr(z).load()
    assert ds.sizes['run_idx'] == 3 and ds.sizes['r'] == 4
    np.testing.assert_array_equal(ds.w_marg_surf.isel(run_idx=1).values[:2], [4.0, 5.0])
    assert np.isnan(ds.w_marg_surf.isel(run_idx=1).values[2:]).all()
    np.testing.assert_array_equal(ds.w_marg_surf.isel(run_idx=2).values, [6.0, 7.0, 8.0, 9.0])
    np.testing.assert_array_equal(ds.qlim.values, [3.0, 4.0, 5.0])


def test_meta_dict_to_dataset():
    x = np.linspace(0, 1, 11)
    meta = {'Zeff': 1.5, 'average_ion_mass': 2.5, 'te_keV_spline': CubicSpline(x, 1 - x),
            'chi_perp_spline': Akima1DInterpolator(x[:6], x[:6] ** 2),
            'IDA_ne': {'x': x, 'y': 2 * x}, 'IDA_te': {'x': x, 'y': 3 * x}}
    ds = meta_dict_to_dataset(meta)
    assert ds.sizes['run_idx'] == 1
    assert ds.average_ion_mass.item() == 2.5
    # DROP_KEYS (profiles and Zeff already in the main dataset) are skipped
    assert {'Zeff', 'te_keV_spline'} <= DROP_KEYS and 'Zeff' not in ds and 'te_keV_spline' not in ds
    # IDA_* profiles share one indexed grid
    assert ds.IDA_ne.dims == ds.IDA_te.dims == ('run_idx', 'psi_n_IDA')
    np.testing.assert_allclose(ds.psi_n_IDA.values, x)
    np.testing.assert_allclose(ds.IDA_te.squeeze().values, 3 * x)
    # Other splines get their own knot grid
    assert ds.chi_perp_spline.dims == ('run_idx', 'chi_perp_spline_knot')
    np.testing.assert_allclose(ds.chi_perp_spline_knot_x.squeeze().values, x[:6])  # stored per run
    np.testing.assert_allclose(ds.chi_perp_spline.squeeze().values, x[:6] ** 2)
