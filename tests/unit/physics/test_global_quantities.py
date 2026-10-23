import numpy as np
import xarray as xr

from tearing_physics_suite.physics.global_quantities import delta_prime_variability


def test_code_comparisons_nan_where_a_code_has_no_delta_prime():
    """Across codes, a missing Delta' leaves the comparison NaN (not 0); across a scan, missing values are skipped."""
    dp = xr.DataArray([[1.0, 2.0], [1.5, 2.0], [np.nan, 3.0]], dims=('code', 'r'),
                      coords={'code': ['rdcon', 'stride', 'jGPEC_galerkin']})
    ds = delta_prime_variability(dp)
    np.testing.assert_array_equal(ds.Delta_prime_diff_across_code, [np.nan, 1.0])
    np.testing.assert_array_equal(ds.Delta_prime_diff_across_jGPEC_galerkinvsRDCON, [np.nan, 1.0])
    np.testing.assert_array_equal(ds.Delta_prime_diff_across_GPEC, [0.5, 0.0])
    scan = delta_prime_variability(dp.rename(code='mtheta').assign_coords(mtheta=[1, 2, 3]), comparison_var='mtheta')
    np.testing.assert_array_equal(scan.Delta_prime_diff_across_mtheta, [0.5, 1.0])
