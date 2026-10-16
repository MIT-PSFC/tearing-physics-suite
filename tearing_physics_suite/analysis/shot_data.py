"""One shot's TPS results as one Dataset along run_idx, with t_ms and draw (baseline: -1) per run."""
import glob
import json
import os

import numpy as np
import xarray as xr


def open_shot(source, shot, variables=None):
    """source: a study dir (<study>/<shot>/tps/<shot>.zarr), a zarr store (study or multi_compile_zarr),
    or a glob of run dirs holding rdcon_xr.nc + meta.json (E7 layout). variables: keep only these."""
    study = os.path.join(source, str(shot), 'tps', f'{shot}.zarr')
    if os.path.isdir(study) or source.rstrip('/').endswith('.zarr'):
        ds = xr.open_zarr(study if os.path.isdir(study) else source)
        if 't_ms' not in ds:
            ds = ds.assign(t_ms=ds['bq_scan_key'] if 'bq_scan_key' in ds else ds['time'])
        if 'draw' not in ds and 'bq_count' in ds:
            ds = ds.assign(draw=ds['bq_count'])
        if 'shot_id' in ds:
            ds = ds.isel(run_idx=np.flatnonzero(ds['shot_id'].values.astype(int) == shot))
        return _keep(ds, variables)
    return _open_runs(sorted(glob.glob(source)), shot, variables)


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
