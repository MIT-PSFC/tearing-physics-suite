import numpy as np
import xarray as xr

from tearing_physics_suite.physics.xr_utils import interp_to_surfaces, like


def test_like_matches_zero_times_idiom():
    t = xr.DataArray([0.2, np.nan, 0.7], dims='r', coords={'r': [2.0, 3.0, 4.0]})
    for v in (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0], dtype=np.float32), 5.0):
        want = v + 0.0 * t
        got = like(v, t)
        xr.testing.assert_identical(got, want.rename(None))
        assert got.dtype == np.float64


def test_like_dataarray_aligns_by_name():
    t = xr.DataArray([0.2, 0.7], dims='r')
    v = xr.DataArray([1.0, 2.0, 3.0], dims='psi_n')
    assert like(v, t).dims == ('psi_n', 'r')


def test_interp_to_surfaces():
    psi = np.linspace(0, 1, 41)
    ds = xr.Dataset({'a': ('psi_n', psi**2), 'psi_n_rational': ('r', [0.25, 0.5])}, coords={'psi_n': psi})
    out = interp_to_surfaces(ds, ['a'])['a']
    assert out.dims == ('r',)
    np.testing.assert_allclose(out.values, [0.0625, 0.25], rtol=1e-12)
