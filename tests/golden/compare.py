"""Diff two golden output dirs made by cases.py.

Usage: python tests/golden/compare.py REF_DIR NEW_DIR [--rtol 0] [--atol 0]
Exact match by default. Prints one line per difference, exits 1 if any.
"""
import argparse
import glob
import os
import pickle as pkl
import sys

import numpy as np
import pandas as pd
import xarray as xr

IGNORE = {'cpu_time', 'wall_time'}  # run timings, never reproducible


NORMALIZE = {}  # name -> function(Dataset) -> Dataset, applied to both sides before comparing


def _c2(ds):
    """Compare a post-C2 dataset with a pre-C2 reference: size-1 k dims squeezed, new setting
    variables dropped, and the X0_on_* ratios (now filled for every code) kept on rdcon only."""
    ds = ds.squeeze([d for d in ('k0', 'k1', 'C0') if d in ds.dims and ds.sizes[d] == 1], drop=True)
    ds = ds.drop_vars(['wd_static', 'force_lmfp', 'psi_pedestal_cutoff', 'dwdt_k0', 'dwdt_k1', 'dwdt_C0'],
                      errors='ignore')
    for v in ('X0_on_w_marg_surf', 'X0_on_wd_at_marg_surf'):
        if v in ds and 'code' in ds[v].dims:
            ds[v] = ds[v].where(ds.code == 'rdcon')
    return ds


NORMALIZE['c2'] = _c2


def diff_ds(a, b, where, rtol, atol, out, normalize=None):
    """Compare two Datasets variable by variable (NaNs equal)."""
    if normalize:
        a, b = NORMALIZE[normalize](a), NORMALIZE[normalize](b)
    va, vb = set(a.variables), set(b.variables)
    for v in sorted(va - vb):
        out.append(f'{where}: var only in ref: {v}')
    for v in sorted(vb - va):
        out.append(f'{where}: var only in new: {v}')
    for v in sorted((va & vb) - IGNORE):
        x, y = a[v], b[v]
        if x.dims != y.dims or x.shape != y.shape:
            out.append(f'{where}:{v}: dims/shape {x.dims}{x.shape} vs {y.dims}{y.shape}')
            continue
        xv, yv = np.asarray(x.values), np.asarray(y.values)
        if xv.dtype.kind in 'fc' and yv.dtype.kind in 'fc':
            if not np.allclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True):
                d = np.nanmax(np.abs(xv - yv)) if xv.size else 0
                out.append(f'{where}:{v}: max |diff| {d:.3e}')
        elif xv.dtype == object or yv.dtype == object:
            xn, yn = pd.isna(xv), pd.isna(yv)
            if not (np.array_equal(xn, yn) and all(p == q for p, q in zip(xv[~xn], yv[~yn]))):
                out.append(f'{where}:{v}: values differ (object)')
        elif not np.array_equal(xv, yv):
            out.append(f'{where}:{v}: values differ ({xv.dtype})')


def diff_obj(a, b, where, rtol, atol, out, normalize=None):
    if isinstance(a, xr.DataArray):
        a, b = a.to_dataset(name='_'), b.to_dataset(name='_')
    if isinstance(a, xr.Dataset):
        if not isinstance(b, xr.Dataset):
            out.append(f'{where}: type {type(a)} vs {type(b)}')
        else:
            diff_ds(a, b, where, rtol, atol, out, normalize)
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            out.append(f'{where}: length/type differs')
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff_obj(x, y, f'{where}[{i}]', rtol, atol, out, normalize)
    elif isinstance(a, dict):
        if not isinstance(b, dict):
            out.append(f'{where}: type {type(a)} vs {type(b)}')
            return
        for k in sorted(set(a) ^ set(b), key=str):
            out.append(f'{where}: key only in {"ref" if k in a else "new"}: {k}')
        for k in sorted(set(a) & set(b), key=str):
            diff_obj(a[k], b[k], f'{where}[{k!r}]', rtol, atol, out, normalize)
    elif isinstance(a, (np.ndarray, float, int, np.number)) and not isinstance(a, bool):
        try:
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                out.append(f'{where}: {a!r} vs {b!r}'[:300])
        except (TypeError, ValueError):
            out.append(f'{where}: not comparable')
    elif isinstance(a, str) and isinstance(b, str) and '/' in a and '/' in b:
        if os.path.basename(a.rstrip('/')) != os.path.basename(b.rstrip('/')):  # paths differ by checkout/out dir
            out.append(f'{where}: {a!r} vs {b!r}'[:300])
    elif isinstance(a, (str, bool, type(None))):
        if a != b:
            out.append(f'{where}: {a!r} vs {b!r}'[:300])
    # Splines and other objects are skipped.


def compare_dirs(ref, new, rtol=0.0, atol=0.0, names=None, normalize=None):
    """Return a list of differences between two golden dirs (empty if identical).

    names: limit to these case names (files named <case>*.pkl/.nc); None = all.
    """
    out = []
    md5 = [open(os.path.join(d, 'exe_md5.txt')).read() if os.path.exists(os.path.join(d, 'exe_md5.txt')) else None
           for d in (ref, new)]
    if None not in md5 and md5[0] != md5[1]:
        print('WARNING: rdcon/stride/pest3x executables differ between runs; exact match not expected')

    def wanted(f):
        return names is None or any(os.path.basename(f).startswith(n) for n in names)
    for f in sorted(glob.glob(os.path.join(ref, '*.pkl'))):
        if not wanted(f):
            continue
        g = os.path.join(new, os.path.basename(f))
        if not os.path.exists(g):
            out.append(f'missing in new: {os.path.basename(f)}')
            continue
        with open(f, 'rb') as fa, open(g, 'rb') as fb:
            diff_obj(pkl.load(fa), pkl.load(fb), os.path.basename(f), rtol, atol, out, normalize)
    for f in sorted(glob.glob(os.path.join(ref, '*.nc'))):
        if not wanted(f):
            continue
        g = os.path.join(new, os.path.basename(f))
        if not os.path.exists(g):
            out.append(f'missing in new: {os.path.basename(f)}')
            continue
        with xr.open_dataset(f) as x, xr.open_dataset(g) as y:
            diff_ds(x.load(), y.load(), os.path.basename(f), rtol, atol, out, normalize)
    z = 'work/multi_run_zarr/compiled_combined_xr.zarr'
    if os.path.exists(os.path.join(ref, z)) and (names is None or 'multi_run_zarr' in names):
        if not os.path.exists(os.path.join(new, z)):
            out.append('missing in new: ' + z)
        else:
            diff_ds(xr.open_zarr(os.path.join(ref, z)).load(),
                    xr.open_zarr(os.path.join(new, z)).load(), 'zarr', rtol, atol, out, normalize)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('new')
    ap.add_argument('--rtol', type=float, default=0.0)
    ap.add_argument('--atol', type=float, default=0.0)
    ap.add_argument('--normalize', choices=sorted(NORMALIZE))
    a = ap.parse_args()
    out = compare_dirs(a.ref, a.new, a.rtol, a.atol, normalize=a.normalize)
    print('\n'.join(out) if out else 'IDENTICAL')
    sys.exit(1 if out else 0)


if __name__ == '__main__':
    main()
