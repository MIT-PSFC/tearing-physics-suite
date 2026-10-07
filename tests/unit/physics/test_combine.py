import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.physics.combine import (
    _counts_along_dim,
    _uniquify_r,
    add_unique_label,
    collapse_to_primary,
    sel_rational,
)

R = [2.0, 3.0, 3.0, 4.0]  # degenerate m=3 (two surfaces with the same m)


def _ds(r=R):
    n = len(r)
    return xr.Dataset({'Delta_prime_surf': ('r', np.arange(n, dtype=float)),
                       'Delta_prime': (('r_prime', 'r'), np.arange(n * n, dtype=float).reshape(n, n))},
                      coords={'r': r, 'r_prime': r})


def test_uniquify_r_index_and_labels():
    u = _uniquify_r(_ds())
    np.testing.assert_array_equal(u.r.values, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(u.r_value.values, R)
    np.testing.assert_array_equal(u.r_occ.values, [0, 1, 0, 0])  # counted from the right
    np.testing.assert_array_equal(u.r_unique.values, [True, False, False, True])
    np.testing.assert_array_equal(u.r_prime_value.values, R)


def test_uniquify_r_rejects_mismatched_r_prime():
    ds = _ds().assign_coords(r_prime=[2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        _uniquify_r(ds)


def test_collapse_keeps_outermost_occurrence():
    c = collapse_to_primary(_uniquify_r(_ds()))
    np.testing.assert_array_equal(c.r.values, [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(c.Delta_prime_surf.values, [0.0, 2.0, 3.0])  # occ==0 copy of m=3 is index 2
    assert c.Delta_prime.shape == (3, 3)


def test_collapse_per_nn_with_nan_padding():
    a = _uniquify_r(_ds([2.0, 3.0, 4.0]))
    b = _uniquify_r(_ds([3.0, 4.0]))
    ds = xr.concat([a, b], dim='nn', join='outer').assign_coords(nn=[1, 2])
    c = collapse_to_primary(ds)
    np.testing.assert_array_equal(c.r.values, [2.0, 3.0, 4.0])
    assert np.isnan(c.Delta_prime_surf.sel(nn=2, r=2.0))
    assert c.Delta_prime_surf.sel(nn=2, r=3.0) == 0.0


def test_sel_rational():
    s = sel_rational(_uniquify_r(_ds()), 3.0)
    assert s.sizes['r'] == 1 and s.Delta_prime_surf.item() == 2.0


def test_counts_along_dim_and_add_unique_label():
    u = _uniquify_r(_ds()).drop_vars(['r_unique', 'r_prime_unique'])
    counts, other = _counts_along_dim(u.r_value, 'r')
    np.testing.assert_array_equal(counts, [1, 2, 2, 1])
    assert other == []
    np.testing.assert_array_equal(add_unique_label(u).r_unique.values, [True, False, False, True])


# combine_codes / add_code_dim / merge_input_dicts (A2.1)
from tearing_physics_suite.physics.combine import add_code_dim, combine_codes, merge_input_dicts  # noqa: E402


def _code_ds(code, bad=False):
    # bad: a degenerate r index (m=3 twice) that differs from the others cannot be aligned
    r = [2.0, 3.0, 3.0] if bad else [2.0, 3.0]
    ds = xr.Dataset({'Delta_prime_surf': ('r', np.arange(len(r), dtype=float))}, coords={'r': r},
                    attrs={'qlim': 3.0})
    return add_code_dim(ds, code)


def test_add_code_dim_does_not_mutate():
    ds = xr.Dataset({'a': ('r', [1.0])}, attrs={'qlim': 3.0})
    out = add_code_dim(ds, 'rdcon')
    assert 'qlim' not in ds and ds.attrs == {'qlim': 3.0}
    assert out.qlim.dims == ('code',) and out.attrs == {} and list(out.code.values) == ['rdcon']


def test_merge_input_dicts():
    a, b = {'nn': 1, 'x': 1}, {'x': 2, 'y': 3}
    assert merge_input_dicts(a, None, b) == {'nn': 1, 'x': 2, 'y': 3}
    assert a == {'nn': 1, 'x': 1}


def test_combine_codes_all():
    out, dropped = combine_codes({'rdcon': _code_ds('rdcon'), 'stride': _code_ds('stride'), 'pest3': None}, nn=2)
    assert out.Delta_prime_surf.dims == ('nn', 'code', 'r') and list(out.code.values) == ['rdcon', 'stride']
    assert out.nn.values.tolist() == [2] and dropped == {}


@pytest.mark.parametrize('bad', ['pest3', 'stride'])
def test_combine_codes_drops_failing_code_by_name(bad):
    ds = {c: _code_ds(c, bad=(c == bad)) for c in ('rdcon', 'stride', 'pest3')}
    out, dropped = combine_codes(ds, nn=1)
    assert list(dropped) == [bad]
    assert list(out.code.values) == [c for c in ('rdcon', 'stride', 'pest3') if c != bad]


def test_combine_codes_debug_raises():
    with pytest.raises(Exception):
        combine_codes({'rdcon': _code_ds('rdcon'), 'pest3': _code_ds('pest3', bad=True)}, nn=1, debug=True)


def test_combine_codes_empty():
    assert combine_codes({'rdcon': None}, nn=None) == (None, {})
