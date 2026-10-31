"""One shot's TPS results as one Dataset along run_idx, with t_ms and draw (baseline: -1) per run, and the ideal
pass's verdicts (<shot>_ideal.zarr, drivers/ideal_pass.py) where the shot has them."""
import glob
import json
import os

import numpy as np
import xarray as xr
import zarr


def open_shot(source, shot, variables=None):
    """source: a study dir (<study>/<shot>/tps/<shot>.zarr, or its published <shot>.zarr.zip), a zarr store
    (study or multi_compile_zarr, directory or .zip), or a glob of run dirs holding rdcon_xr.nc + meta.json
    (E7 layout). variables: keep only these. A study's shot gets its ideal-pass variables (nzero, ideal_*) by
    case_key."""
    study = os.path.join(source, str(shot), 'tps', f'{shot}.zarr')
    study = study + '.zip' if os.path.exists(study + '.zip') else study
    if os.path.exists(study) or source.rstrip('/').endswith(('.zarr', '.zarr.zip')):
        path = study if os.path.exists(study) else source.rstrip('/')
        ds = xr.open_zarr(zarr.storage.ZipStore(path, mode='r') if path.endswith('.zip') else path)
        if 't_ms' not in ds:
            ds = ds.assign(t_ms=ds['bq_scan_key'] if 'bq_scan_key' in ds else ds['time'])
        if 'draw' not in ds and 'bq_count' in ds:
            ds = ds.assign(draw=ds['bq_count'])
        if 'shot_id' in ds:
            ds = ds.isel(run_idx=np.flatnonzero(ds['shot_id'].values.astype(int) == shot))
        if path == study:
            ds = _with_ideal(ds, os.path.join(source, str(shot), 'tps', f'{shot}_ideal.zarr'))
        return _keep(ds, variables)
    return _open_runs(sorted(glob.glob(source)), shot, variables)


def _with_ideal(ds, path):
    """ds with the ideal store's variables at path (or its .zip), matched by case_key: NaN, -1 (int flags) or ''
    for runs without a verdict; ds unchanged without the store."""
    path = path + '.zip' if os.path.exists(path + '.zip') else path
    if not os.path.exists(path) or 'case_key' not in ds:
        return ds
    ideal = xr.open_zarr(zarr.storage.ZipStore(path, mode='r') if path.endswith('.zip') else path).load()
    pos = {k: i for i, k in enumerate(ideal['case_key'].values)}
    idx = np.array([pos.get(k, -1) for k in ds['case_key'].values])
    out = {}
    for v in [v for v in ideal.data_vars if v == 'nzero' or v.startswith('ideal_')]:
        a = ideal[v]
        text = a.dtype.kind in 'OTU'
        fill = '' if text else -1 if a.dtype.kind == 'i' else np.nan
        vals = np.take(a.values.astype(object) if text else a.values, np.maximum(idx, 0), axis=a.dims.index('run_idx'))
        mask = (idx < 0).reshape([-1 if d == 'run_idx' else 1 for d in a.dims])
        out[v] = xr.DataArray(np.where(mask, fill, vals), dims=a.dims,
                              coords={d: ideal[d] for d in a.dims if d in ideal.coords})
    return ds.assign(out)                                  # on ds's nn and code


def select(ds, t_ms=None, draws=None):
    """The runs at these t_ms and draws (None: all)."""
    keep = np.ones(ds.sizes['run_idx'], bool)
    if t_ms is not None:
        keep &= np.isin(ds['t_ms'].values, np.atleast_1d(t_ms))
    if draws is not None:
        keep &= np.isin(ds['draw'].values, np.atleast_1d(draws))
    return ds.isel(run_idx=np.flatnonzero(keep))


def _keep(ds, variables):
    return ds if variables is None else ds[[v for v in variables if v in ds] + ['t_ms', 'draw']]


def _open_runs(dirs, shot, variables):
    """Concatenate the ok runs of `shot`; dims without coords are padded by position, as in the zarr stores."""
    runs = []
    for d in dirs:
        with open(os.path.join(d, 'meta.json')) as fh:
            meta = json.load(fh)
        if int(meta['shot']) != shot or not meta.get('ok', True):
            continue
        ds = xr.open_dataset(os.path.join(d, 'rdcon_xr.nc')).drop_vars('run_idx', errors='ignore')
        ds = ds.assign(t_ms=float(meta['t_ms']), draw=-1 if meta['draw'] == 'base' else int(meta['draw']))
        ds = _keep(ds, variables)
        runs.append(ds.assign_coords({k: np.arange(n) for k, n in ds.sizes.items() if k not in ds.coords}))
    if not runs:
        raise FileNotFoundError(f'no ok runs of shot {shot} in {len(dirs)} dirs')
    return xr.concat(runs, 'run_idx', join='outer', data_vars='all', coords='minimal', compat='override')
