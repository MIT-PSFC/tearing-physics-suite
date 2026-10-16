import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.analysis.dist_plot import dist_values


def _ds():
    v = np.arange(12.).reshape(4, 1, 3)   # (run_idx, nn, r)
    return xr.Dataset({'Delta_prime_surf': (('run_idx', 'nn', 'r'), v),
                       'r_value': (('run_idx', 'nn', 'r'), np.tile([2., 3., 4.], (4, 1, 1))),
                       't_ms': ('run_idx', [1051., 1051., 1350., np.nan]), 'draw': ('run_idx', [0, 1, 0, 2])},
                      coords={'nn': [1]})


def test_groups_by_t_ms_and_filters():
    t, vals = dist_values(_ds(), 'Delta_prime_surf', sel={'nn': 1, 'code': 'rdcon'},
                          where={'r_value': 3, '__or': {'or': [{'var': 'draw', 'val': 0},
                                                               {'var': 'Delta_prime_surf', 'op': '>', 'val': 3}]}})
    assert list(t) == [1051., 1350.]
    assert [list(v) for v in vals] == [[1., 4.], [7.]]


def test_missing_filter_variable_is_skipped():
    with pytest.warns(UserWarning, match='Delta_prime_ImRatio'):
        _, vals = dist_values(_ds(), 'Delta_prime_surf', where={'Delta_prime_ImRatio': [{'op': '<', 'val': 0.05}]})
    assert [v.size for v in vals] == [6, 3]
