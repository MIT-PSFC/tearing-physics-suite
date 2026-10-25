"""What the compiled zarr keeps (item F): drivers/store_schema.py."""
import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.drivers.mre_recompute import INPUTS, K_OUTPUTS, OPTIONAL_INPUTS, recompute_mre_hyperparams
from tearing_physics_suite.drivers.store_schema import (
    KEEP_FLOAT64,
    PROFILE_KEEPBITS,
    bitround,
    blank_duplicates,
    keep_least_stable_eigenvectors,
    drop_store_duplicates,
    prepare_for_store,
    to_float32,
)
from tearing_physics_suite.physics.combine import _uniquify_r, collapse_to_primary


def _run(codes=('rdcon', 'stride', 'pest3')):
    r = np.array([2.0, 3.0, 3.0])
    psi = np.linspace(0, 1, 5)
    nc = len(codes)
    ds = xr.Dataset(
        {'f': (('code', 'psi_n'), np.tile(psi + 1, (nc, 1))),
         'q': (('code', 'psi_n'), np.tile(psi + 2, (nc, 1))),
         'betan': (('code',), np.full(nc, 1.5)),
         'Delta_prime_surf': (('code', 'r'), np.arange(3 * nc, dtype=float).reshape(nc, 3)),
         'w_marg_surf': (('code', 'r'), np.ones((nc, 3))),
         'dwdt_surf': (('code', 'r', 'w_bar'), np.ones((nc, 3, 4))),
         'Delta_prime_diff_across_code': (('r',), np.ones(3)),
         'Delta_prime_diff_across_GPECvsPEST': (('r',), np.ones(3)),
         'mpsi': (('code',), np.full(nc, 128)),
         'cell_x1': (('code', 'icell'), np.zeros((nc, 6))),
         'te_keV': (('psi_n',), psi + 1)},
        coords={'code': list(codes), 'psi_n': psi, 'r': r, 'r_prime': r})
    return _uniquify_r(ds)


def test_dwdt_dropped_and_float32_except_recompute_inputs():
    out = prepare_for_store(_run())
    assert 'dwdt_surf' not in out and 'cell_x1' not in out and 'icell' not in out.dims and 'te_keV' not in out
    assert out.w_marg_surf.dtype == np.float32 and out.f.dtype == np.float32
    assert out.Delta_prime_surf.dtype == np.float64  # k0/k1/C0 recompute input
    assert out.psi_n.dtype == np.float64 and out.mpsi.dtype == np.int64  # coords and ints unchanged
    assert {'Delta_prime_surf', 'psi_n_rational', 'Dr_surf', 'wd_static'} <= KEEP_FLOAT64 == set(INPUTS) | set(OPTIONAL_INPUTS)


def test_options_off_keep_everything_but_store_drop():
    ds = _run()
    out = prepare_for_store(ds, store_float32=False, drop_duplicates=False)
    xr.testing.assert_identical(out, ds.drop_vars(['dwdt_surf', 'cell_x1', 'te_keV']).assign(shear=(('code', 'r'), np.full((3, 3), np.nan))))


def test_stride_v160_repeats_dropped_and_shear_in_every_run():
    """GPEC v1.6's STRIDE writes di/dr_rational, resm, msing (stored already) and shear (kept; NaN without STRIDE)."""
    ds = _run().assign(shear=(('code', 'r'), np.ones((3, 3))), resm=(('code', 'r'), np.ones((3, 3), dtype=np.int32)),
                       di_rational=(('code', 'r'), np.ones((3, 3))), dr_rational=(('code', 'r'), np.ones((3, 3))),
                       msing=(('code',), np.ones(3)))
    out, bare = prepare_for_store(ds), prepare_for_store(_run())
    assert not {'resm', 'di_rational', 'dr_rational', 'msing'} & set(out) and set(out) == set(bare)
    assert out.shear.dims == bare.shear.dims == ('code', 'r') and np.isnan(bare.shear).all()


def test_stride_equil_copy_blanked_rdcon_kept():
    out = blank_duplicates(_run())
    assert np.isnan(out.f.sel(code='stride')).all() and np.isnan(out.betan.sel(code='stride'))
    np.testing.assert_array_equal(out.f.sel(code='rdcon'), _run().f.sel(code='rdcon'))
    np.testing.assert_array_equal(out.f.sel(code='pest3'), _run().f.sel(code='pest3'))  # pest3 not in the table
    assert out.dup_maxreldiff.sel(dup_code='stride', dup_var='f') == 0


def test_q_never_blanked():
    out = blank_duplicates(_run())
    xr.testing.assert_identical(out.q, _run().q)


def test_differing_copy_kept_with_warning():
    ds = _run()
    ds['f'] = ds.f.where(ds.code != 'stride', ds.f * 1.01)
    with pytest.warns(UserWarning, match='f: stride differs'):
        out = blank_duplicates(ds)
    xr.testing.assert_identical(out.f, ds.f)
    assert out.dup_maxreldiff.sel(dup_code='stride', dup_var='f') == pytest.approx(0.01, rel=1e-6)


def test_exact_duplicates_dropped():
    out = drop_store_duplicates(_run())
    assert not {'r_prime_value', 'r_prime_occ', 'r_prime_unique', 'Delta_prime_diff_across_GPECvsPEST'} & set(out)
    # With a jGPEC code, GPECvsPEST differs from across_code: kept
    jg = drop_store_duplicates(_run(('rdcon', 'stride', 'jGPEC_galerkin', 'pest3')))
    assert 'Delta_prime_diff_across_GPECvsPEST' in jg


def test_collapse_to_primary_without_r_prime_vars():
    ds = _run()
    xr.testing.assert_identical(collapse_to_primary(drop_store_duplicates(ds)).drop_vars('r_prime_unique', errors='ignore'),
                                collapse_to_primary(ds).drop_vars(['r_prime_unique', 'Delta_prime_diff_across_GPECvsPEST']))


def test_to_float32_keeps_listed():
    ds = xr.Dataset({'a': ('x', [1.0, 2.0]), 'b': ('x', [1.0, 2.0])})
    out = to_float32(ds, keep={'b'})
    assert out.a.dtype == np.float32 and out.b.dtype == np.float64


# --- golden store (TPS_GOLDEN_DIR) ---------------------------------------------------------------
ZARR = 'work/multi_run_zarr/compiled_combined_xr.zarr'


@pytest.fixture(scope='module')
def golden_run(golden_dir):
    src = xr.open_zarr(golden_dir / ZARR).isel(run_idx=0).load()
    return src.assign(wd_static=False, force_lmfp=False, psi_pedestal_cutoff=0.9)


def test_golden_store_smaller_and_recompute_matches(golden_run, tmp_path):
    out = prepare_for_store(golden_run)
    a, b = tmp_path / 'old.zarr', tmp_path / 'new.zarr'
    golden_run.expand_dims('run_idx').to_zarr(a)
    out.expand_dims('run_idx').to_zarr(b)
    size = {p: sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in (a, b)}
    assert size[b] < 0.7 * size[a]
    # MRE outputs recomputed from the reduced store's (float64) inputs match its float32 outputs
    new = xr.open_zarr(recompute_mre_hyperparams(b, 0.8227, 1.7, 0.6, out_path=str(tmp_path / 'o.zarr'))).load()
    new = new.squeeze(['k0', 'k1', 'C0', 'run_idx'], drop=True)
    for v in K_OUTPUTS:
        stored = out[v].sel(code='rdcon') if v.startswith('X0_on_') else out[v]
        rec = new[v].sel(code='rdcon') if v.startswith('X0_on_') else new[v]
        stored = stored.squeeze([d for d in ('k0', 'k1', 'C0') if d in stored.dims], drop=True)
        np.testing.assert_allclose(rec.transpose(*stored.dims).values.astype(np.float32), stored.values,
                                   rtol=1e-6, err_msg=v)


def test_golden_stride_copies_blanked_only_where_equal(golden_run):
    out = prepare_for_store(golden_run)
    assert np.isnan(out.f.sel(code='stride')).all()
    assert np.isfinite(out.q.sel(code='stride')).any()
    assert np.nanmax(out.dup_maxreldiff.values) == 0


def test_dup_maxreldiff_same_coords_without_stride():
    a, b = blank_duplicates(_run()), blank_duplicates(_run(('rdcon', 'pest3')))
    xr.testing.assert_identical(a.dup_maxreldiff.dup_var, b.dup_maxreldiff.dup_var)
    assert np.isnan(b.dup_maxreldiff).all()


def test_golden_multi_compile_zarr(golden_dir, tmp_path):
    """multi_compile_zarr on the golden runs' result pickles: reduced store, options recorded."""
    import shutil

    from tearing_physics_suite.drivers.multi_run import multi_compile_zarr
    src = golden_dir / 'work' / 'multi_run_zarr'
    pkls = sorted(src.glob('result_*.pkl'))
    for p in pkls:
        shutil.copy(p, tmp_path)
    ds, _ = multi_compile_zarr([f'eq{i}' for i in range(len(pkls))], str(tmp_path), report_errs=False)
    assert ds.attrs['store_float32'] == 1 and 'dwdt_surf' not in ds and 'r_prime_value' not in ds
    assert ds.w_marg_surf.dtype == np.float32 and ds.Delta_prime_surf.dtype == np.float64
    old = xr.open_zarr(src / 'compiled_combined_xr.zarr')
    np.testing.assert_array_equal(ds.Delta_prime_surf.values, old.Delta_prime_surf.values)


def test_eigenvectors_of_two_least_stable_modes():
    mode, m = np.arange(1, 6), np.arange(-2, 3)
    vec = np.random.default_rng(0).random((2, 5, 5))
    ds = xr.Dataset({'W_t': (('i', 'mode', 'm'), vec), 'W_t_eigenvalue': (('i', 'mode'), vec[:, :, 0]),
                     'W_t_eigenvector': (('i', 'mode', 'm'), vec)}, coords={'mode': mode, 'm': m})
    out = keep_least_stable_eigenvectors(ds)
    np.testing.assert_array_equal(out.W_t_eigenvector.values, vec[:, :2])
    assert out.W_t_eigenvector.dims == ('i', 'eigvec_mode', 'm') and list(out.eigvec_mode.values) == [1, 2]
    xr.testing.assert_identical(out[['W_t', 'W_t_eigenvalue']], ds[['W_t', 'W_t_eigenvalue']])


def test_profiles_bit_rounded_in_float32_stores():
    x = np.array([1.0, np.pi, -1e-7, 6.02e23, 0.0, np.nan, np.inf], np.float32)
    r = bitround(x, PROFILE_KEEPBITS)
    fin = np.isfinite(x)
    assert np.all(np.abs(r[fin] - x[fin]) <= 2.0**-(PROFILE_KEEPBITS + 1)*np.abs(x[fin]))
    assert np.all(r.view(np.uint32)[fin] & ((1 << (23 - PROFILE_KEEPBITS)) - 1) == 0) and np.isnan(r[5]) and r[6] == np.inf
    ds = _run()
    ds = ds.assign(q=ds.q*np.pi, w_marg_surf=ds.w_marg_surf*np.pi)
    out = prepare_for_store(ds)
    np.testing.assert_array_equal(out.q.values, bitround(ds.q.values, PROFILE_KEEPBITS))      # (code, psi_n) profile
    assert not np.array_equal(out.q.values, np.float32(ds.q.values))
    np.testing.assert_array_equal(out.w_marg_surf.values, np.float32(ds.w_marg_surf.values))  # surface value: not rounded

