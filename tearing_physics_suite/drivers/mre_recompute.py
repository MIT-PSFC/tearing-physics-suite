"""Recompute the k0/k1/C0-dependent MRE outputs of a TPS zarr database on a new grid of k values.

The Fortran outputs and the RDCON surface terms are reused; only extract_critical_mre_factors_on_modes and
global_mre_quantities are rerun. Results go to a separate store (the main store is not rewritten).

    python -m tearing_physics_suite.drivers.mre_recompute DB.zarr --k1 1.2 1.7 2.2 --C0 0.4 0.6 0.8 --workers 16
"""
import argparse
import multiprocessing
import sys

import numpy as np
import xarray as xr

from tearing_physics_suite.physics.global_quantities import global_mre_quantities
from tearing_physics_suite.physics.mre_model import (
    extract_critical_mre_factors_on_modes,
    generate_wd_function,
    mre_combination_wrap,
)

# Inputs of extract_critical_mre_factors_on_modes / generate_wd_function / global_mre_quantities
SURFACE_INPUTS = ('psi_n_rational', 'Dr_surf', 'Di_surf', 'Dnc_surf', 'H_surf', 'eta_star_surf', 'X0_surf',
                  'chi_perp_surf', 'chi_para_smfp_surf', 'chi_para_lmfp_no_w_surf', 'Wc_prefac_m_surf')
SETTINGS = ('wd_static', 'force_lmfp', 'psi_pedestal_cutoff')
INPUTS = ('Delta_prime_surf',) + SURFACE_INPUTS + SETTINGS
# Outputs that depend on k1, C0 (and k0 for prefac_surf)
K_OUTPUTS = ('w_marg_surf', 'w_sat_surf', 'w_max_loc_surf', 'dwdtau_max_surf', 'wd_at_marg_surf',
             'X0_on_w_marg_surf', 'X0_on_wd_at_marg_surf', 'prefac_surf')
GLOBAL_OUTPUTS = ('min_w_marg_allsurf', 'max_dwdtau_allsurf', 'min_w_marg_rank', 'max_dwdtau_rank')


def check_recompute_inputs(ds):
    """Names in INPUTS missing from ds (empty if the MRE outputs can be recomputed)."""
    return [v for v in INPUTS if v not in ds.variables]


def _flags(ds):
    return dict(iterator=bool(ds['wd_static'].values), force_lmfp=bool(ds['force_lmfp'].values))


def recompute_run(run_ds, k0, k1, C0):
    """K_OUTPUTS and GLOBAL_OUTPUTS for one run (dims nn, code, ...; no run_idx) on the given k values."""
    # r is positional; the store may NaN-pad it, which extract_critical_mre_factors_on_modes's r check rejects
    run_ds = run_ds.assign_coords(r=np.arange(run_ds.sizes['r'], dtype=float))
    per_nn = []
    for n in run_ds.nn.values:
        ds = run_ds.sel(nn=n)
        rdcon = ds.sel(code=['rdcon'])
        out = []
        for code in ds.code.values:
            res = extract_critical_mre_factors_on_modes(ds.sel(code=[code])[['Delta_prime_surf']], rdcon,
                                                        k0=k0, k1=k1, C0=C0, **_flags(run_ds))
            out.append(res[list(K_OUTPUTS)])
        per_nn.append(xr.concat(out, dim='code', coords='minimal', compat='override').expand_dims(nn=[n]))
    res = xr.concat(per_nn, dim='nn', coords='minimal', compat='override')
    glob = global_mre_quantities(
        res.assign(psi_n_rational=run_ds['psi_n_rational'], Delta_prime_surf=run_ds['Delta_prime_surf']),
        psi_pedestal_cutoff=float(run_ds['psi_pedestal_cutoff'].values))
    return res.assign({v: glob[v] for v in GLOBAL_OUTPUTS})


def _recompute_chunk(args):
    """Worker: recompute runs [start, stop) of the input store and write them into the output store."""
    in_path, out_path, start, stop, k0, k1, C0 = args
    ds = xr.open_zarr(in_path)[list(INPUTS)].isel(run_idx=slice(start, stop)).load()
    runs = [recompute_run(ds.isel(run_idx=i), k0, k1, C0) for i in range(ds.sizes['run_idx'])]
    out = xr.concat(runs, dim='run_idx', coords='minimal', compat='override').transpose('run_idx', ...)
    out.drop_vars(list(out.coords)).to_zarr(out_path, region={'run_idx': slice(start, stop)})
    return stop - start


def _template(in_path, k0, k1, C0):
    """Output dataset of NaNs (dask) with the full output shape, from recomputing the first run."""
    import dask.array as da
    src = xr.open_zarr(in_path)
    first = recompute_run(src[list(INPUTS)].isel(run_idx=0).load(), k0, k1, C0)
    n = src.sizes['run_idx']
    tmpl = xr.Dataset(coords={'run_idx': np.arange(n), **{d: first[d] for d in first.dims if d in first.coords}})
    for v in first.data_vars:
        shape = (n,) + first[v].shape
        tmpl[v] = (('run_idx',) + first[v].dims, da.full(shape, np.nan, chunks=(1,) + first[v].shape))
    if 'run_idx' in src.coords:
        tmpl = tmpl.assign_coords(run_label=('run_idx', src['run_idx'].values))
    return tmpl, n


def recompute_mre_hyperparams(zarr_path, k0, k1, C0, out_path=None, n_workers=1, run_chunk=None):
    """Recompute K_OUTPUTS and GLOBAL_OUTPUTS for every run of a TPS zarr store on the (k0, k1, C0) grid.

    Writes <zarr>_mre_hyperparams.zarr (or out_path); returns its path. Runs are split into chunks of
    run_chunk runs (default: about 4 chunks per worker) and processed by n_workers spawn processes.
    """
    missing = check_recompute_inputs(xr.open_zarr(zarr_path))
    if missing:
        raise ValueError(f'{zarr_path} lacks {missing}; it was made before these were stored (item C1).')
    k0, k1, C0 = (list(np.atleast_1d(np.asarray(k, dtype=float))) for k in (k0, k1, C0))
    out_path = out_path or str(zarr_path).rstrip('/').removesuffix('.zarr') + '_mre_hyperparams.zarr'
    tmpl, n = _template(zarr_path, k0, k1, C0)
    run_chunk = run_chunk or max(1, int(np.ceil(n / (4 * n_workers))))
    tmpl = tmpl.chunk({'run_idx': run_chunk})
    tmpl.to_zarr(out_path, mode='w', compute=False)
    jobs = [(str(zarr_path), out_path, a, min(a + run_chunk, n), k0, k1, C0) for a in range(0, n, run_chunk)]
    if n_workers > 1:
        with multiprocessing.get_context('spawn').Pool(n_workers) as pool:
            done = sum(pool.imap_unordered(_recompute_chunk, jobs))
    else:
        done = sum(map(_recompute_chunk, jobs))
    assert done == n
    return out_path


def dwdt_curves(ds, k0=0.8227, k1=1.7, C0=0.6, w_bar=None):
    """dw/dt curves on a small, already-selected dataset, for any k0, k1, C0 grid.

    ds needs dims code and r (any others, e.g. run_idx, nn, Delta_prime_type, are kept) plus INPUTS.
    Returns a DataArray with dims (*other dims of Delta_prime_surf, w_bar, k0, k1, C0).
    """
    w_bar = np.logspace(-5, 0, num=200) if w_bar is None else np.asarray(w_bar, dtype=float)
    k0, k1, C0 = (np.atleast_1d(np.asarray(k, dtype=float)) for k in (k0, k1, C0))
    dp = ds['Delta_prime_surf']
    out = xr.DataArray(np.full(dp.shape + (len(w_bar), len(k0), len(k1), len(C0)), np.nan),
                       dims=dp.dims + ('w_bar', 'k0', 'k1', 'C0'),
                       coords={**dp.coords, 'w_bar': w_bar, 'k0': k0, 'k1': k1, 'C0': C0})
    other = [d for d in dp.dims if d not in ('code', 'r', 'Delta_prime_type')]
    for idx in np.ndindex(*[ds.sizes[d] for d in other]):
        sel = dict(zip(other, idx))
        sub = ds.isel(sel)
        flags = _flags(sub)
        rdcon = sub.sel(code=['rdcon'])
        for ri in range(sub.sizes['r']):
            s = rdcon.isel(r=ri)
            Dr, Di, Dnc, H = (s[v].values for v in ('Dr_surf', 'Di_surf', 'Dnc_surf', 'H_surf'))
            if np.isnan(Dr) or np.isnan(Di) or np.isnan(Dnc) or np.isnan(H) or Di > 0:
                continue
            wd_function = generate_wd_function(s, **flags)
            eta_star = s['eta_star_surf'].values
            for i0, kk0 in enumerate(k0):
                for i1, kk1 in enumerate(k1):
                    for iC, cc0 in enumerate(C0):
                        to_mre = mre_combination_wrap(wd_function, Dr, Di, Dnc, H, kk1, cc0, eta_star/kk0, w_bar, w_bar)
                        for ci in range(sub.sizes['code']):
                            for di in range(sub.sizes.get('Delta_prime_type', 1)):
                                at = {'r': ri, 'code': ci} | ({'Delta_prime_type': di} if 'Delta_prime_type' in dp.dims else {})
                                curve = to_mre(sub['Delta_prime_surf'].isel(at).values)[0]
                                out[{**sel, **at, 'k0': i0, 'k1': i1, 'C0': iC}] = curve
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('zarr')
    ap.add_argument('--k0', type=float, nargs='+', default=[0.8227])
    ap.add_argument('--k1', type=float, nargs='+', default=[1.7])
    ap.add_argument('--C0', type=float, nargs='+', default=[0.6])
    ap.add_argument('--out')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--run-chunk', type=int)
    a = ap.parse_args(argv)
    print(recompute_mre_hyperparams(a.zarr, a.k0, a.k1, a.C0, a.out, a.workers, a.run_chunk))
    return 0


if __name__ == '__main__':
    sys.exit(main())
