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


def test_add_to_zarr_store_ragged_with_strings(tmp_path):
    z = tmp_path / 's.zarr'
    for i, n in enumerate((2, 3)):
        ds = _run([1.0] * n, i).assign(label=('r', np.array(['ab'] * n, dtype=object)), key=np.array(f'k{i}', dtype=object))
        add_to_zarr_store(ds, z, 'run_idx')
    ds = xr.open_zarr(z).load()
    assert ds.sizes['r'] == 3 and list(ds.key.values) == ['k0', 'k1'] and ds.label.values[1, 2] == 'ab'


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


def test_multi_compile_zarr_aligns_codes_by_name(tmp_path):
    import pickle as pkl

    from tearing_physics_suite.drivers import multi_run as mr
    runs = [['rdcon', 'stride', 'jGPEC_galerkin'], ['rdcon', 'jGPEC_galerkin'], ['rdcon', 'stride', 'jGPEC_galerkin']]
    for i, codes in enumerate(runs):
        val = [{'rdcon': 1.0, 'stride': 2.0, 'jGPEC_galerkin': 3.0}[c] for c in codes]
        ds = xr.Dataset({'Delta_prime': (('nn', 'code'), [val])}, coords={'nn': [1], 'code': codes})
        with open(tmp_path / f'result_{i}.pkl', 'wb') as f:
            pkl.dump({'combined_xr': ds, 'input_dict_out': {'nx': 64}}, f)
    out, _ = mr.multi_compile_zarr([f'eq{i}' for i in range(3)], str(tmp_path), report_errs=False)
    out = out.load()
    assert list(out.code.values) == ['rdcon', 'stride', 'jGPEC_galerkin']    # unused codes (pest3, ...) dropped
    dp = out.Delta_prime.isel(nn=0)
    np.testing.assert_array_equal(dp.sel(code='jGPEC_galerkin').values, [3.0, 3.0, 3.0])
    assert np.isnan(dp.sel(code='stride').values[1]) and dp.sel(code='stride').values[0] == 2.0
