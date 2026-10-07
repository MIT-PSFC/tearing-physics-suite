"""What multi_compile_zarr drops, blanks and casts before a run goes into the compiled zarr (item F).

Per-run results (result_{idx}.pkl, combined_xr_{idx}.nc) keep everything; only the store is reduced.
"""
import warnings

import numpy as np
import xarray as xr

from tearing_physics_suite.drivers.mre_recompute import INPUTS as RECOMPUTE_INPUTS

# Recomputable on demand: drivers.mre_recompute.dwdt_curves (P1)
STORE_DROP = ('dwdt_surf',)
# k0/k1/C0 recompute inputs stay float64 when the store is float32 (P7)
KEEP_FLOAT64 = frozenset(RECOMPUTE_INPUTS)
# Never blanked as cross-code duplicates: every code keeps its own q (P9, user)
KEEP_ALWAYS = frozenset({'q', 'q0', 'q95', 'qmin', 'qmax', 'qlim', 'qa', 'q_pest3', 'q_rational'})
# GPEC equil outputs that STRIDE reports as well as RDCON (P9). Blanked on STRIDE where equal to RDCON.
GPEC_EQUIL_VARS = ('f', 'mu0p', 'dvdpsi', 'ca1', 'di', 'dr', 'amean', 'rmean', 'aratio', 'kappa', 'delta1',
                   'delta2', 'li1', 'li2', 'li3', 'ro', 'zo', 'psio', 'betap1', 'betap2', 'betap3', 'betat',
                   'betan', 'bt0', 'crnt', 'power_b', 'power_bp', 'power_r', 'mpsi', 'mtheta', 'mlow', 'psilow',
                   'psilim', 'shot', 'time')
# {var: {code: rtol}}: that code's slice is blanked (NaN) where it matches RDCON's within rtol.
# PEST3 and jGPEC pairs are added once their tolerances are measured (plans/F.md, F4).
DUPLICATE_SLICES = {v: {'stride': 1e-12} for v in GPEC_EQUIL_VARS}
# Comparisons that equal *_across_code when the only codes are RDCON, STRIDE and PEST3 (P6)
GPEC_PEST3_CODES = frozenset({'rdcon', 'stride', 'pest3'})


def blank_duplicates(ds, table=None):
    """Blank (NaN) code slices that duplicate RDCON's (see DUPLICATE_SLICES); RDCON's copy is kept.

    Adds dup_maxreldiff(dup_code, dup_var): the measured max relative difference to RDCON, NaN if not
    compared (same coords for every run, so runs stack). Slices outside their tolerance are kept, with a warning.
    """
    table = DUPLICATE_SLICES if table is None else table
    codes = sorted({c for per in table.values() for c in per})
    names = [v for v in table if v not in KEEP_ALWAYS]
    diff = np.full((len(codes), len(names)), np.nan, dtype=np.float32)
    out = {}
    have = list(ds['code'].values) if 'code' in ds.dims else []
    for j, v in enumerate(names):
        if 'rdcon' not in have or v not in ds or 'code' not in ds[v].dims or ds[v].dtype.kind != 'f':
            continue
        da = ds[v]
        ref = da.sel(code='rdcon')
        for i, c in enumerate(codes):
            if c not in table[v] or c not in have:
                continue
            other = da.sel(code=c)
            both = np.isfinite(ref.values) & np.isfinite(other.values)
            if not both.any():
                continue
            a, b = ref.values[both], other.values[both]
            diff[i, j] = np.max(np.abs(b - a) / np.maximum(np.abs(a), np.finfo(float).tiny))
            if np.all(np.isclose(b, a, rtol=table[v][c], atol=0)):
                da = da.where(da['code'] != c)
            else:
                warnings.warn(f'{v}: {c} differs from rdcon (max rel diff {diff[i, j]:.3g}); kept', stacklevel=2)
        out[v] = da
    ds = ds.assign(out)
    return ds.assign(dup_maxreldiff=xr.DataArray(diff, dims=('dup_code', 'dup_var'),
                                                 coords={'dup_code': codes, 'dup_var': names}))


def drop_store_duplicates(ds):
    """Drop variables that repeat others exactly (P6): r_prime_* (= r_*) and, when the only codes are
    RDCON/STRIDE/PEST3, the *_across_GPECvsPEST* comparisons (= *_across_code*)."""
    drop = [f'r_prime_{s}' for s in ('value', 'occ', 'unique')
            if f'r_prime_{s}' in ds and f'r_{s}' in ds
            and np.array_equal(ds[f'r_prime_{s}'].values, ds[f'r_{s}'].values, equal_nan=True)]
    if 'code' in ds.dims and set(ds['code'].values) <= GPEC_PEST3_CODES:
        drop += [v for v in ds.data_vars if '_across_GPECvsPEST' in v]
    return ds.drop_vars(drop)


def to_float32(ds, keep=KEEP_FLOAT64):
    """float64 data variables as float32, except keep (P7). Coordinates are unchanged."""
    return ds.assign({v: ds[v].astype(np.float32) for v in ds.data_vars
                      if ds[v].dtype == np.float64 and v not in keep})


def prepare_for_store(ds, store_float32=True, drop_duplicates=True):
    """One run's combined dataset reduced for the compiled zarr (plans/F.md, P1/P6/P7/P9)."""
    ds = ds.drop_vars([v for v in STORE_DROP if v in ds])
    if drop_duplicates:
        ds = blank_duplicates(drop_store_duplicates(ds))
    return to_float32(ds) if store_float32 else ds
