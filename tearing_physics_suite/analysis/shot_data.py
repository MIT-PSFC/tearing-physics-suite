"""One shot's TPS results as one Dataset along run_idx, with t_ms and draw (baseline: -1) per run, and the ideal
pass's verdicts (<shot>_ideal.zarr, drivers/ideal_pass.py) where the shot has them."""
import glob
import json
import os

import numpy as np
import xarray as xr
import zarr

from tearing_physics_suite.physics.global_quantities import global_mre_quantities


def open_shot(source, shot, variables=None, more_n=()):
    """source: a study dir (<study>/<shot>/tps/<shot>.zarr, or its published <shot>.zarr.zip), a zarr store
    (study or multi_compile_zarr, directory or .zip), or a glob of run dirs holding rdcon_xr.nc + meta.json
    (E7 layout). variables: keep only these. A study's shot gets its ideal-pass variables (nzero, ideal_*) by
    case_key, and eps_surf (see _with_eps_surf). more_n: sources of the same draws at further n (e.g. a later
    STRIDE-only study at nvec [4] on the same bouquets), joined by case_key (see _with_n)."""
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
        for other in more_n:
            ds = _with_n(ds, open_shot(other, shot))
        return _keep(_with_eps_surf(ds), variables)
    return _open_runs(sorted(glob.glob(source)), shot, variables)


def _with_ideal(ds, path):
    """ds with the ideal store's variables at path (or its .zip), matched by case_key: NaN, -1 (int flags) or ''
    for runs without a verdict, or ds's own value (the main pass's STRIDE nzero); ds unchanged without the store."""
    path = path + '.zip' if os.path.exists(path + '.zip') else path
    if not os.path.exists(path) or 'case_key' not in ds:
        return ds
    ideal = xr.open_zarr(zarr.storage.ZipStore(path, mode='r') if path.endswith('.zip') else path).load()
    keep = [v for v in ideal.data_vars if v == 'nzero' or v.startswith('ideal_')]
    text = [v for v in keep if ideal[v].dtype.kind in 'OTU']                    # zarr's StringDType: as objects
    fill = {v: _fill(ideal[v]) for v in keep}
    keys = ideal['case_key'].values.astype(object)
    ideal = ideal[keep].assign({v: ideal[v].astype(object) for v in text}).assign_coords(run_idx=keys)
    ideal = ideal.reindex(run_idx=ds['case_key'].values.astype(object), fill_value=fill)
    ideal = ideal.drop_vars('run_idx')
    return ds.assign({v: ideal[v].combine_first(ds[v]) if v in ds else ideal[v] for v in keep})   # on ds's nn, code


MRE_GLOBALS = ('min_w_marg_allsurf', 'max_dwdtau_allsurf', 'min_w_marg_rank', 'max_dwdtau_rank')   # global_mre_quantities


def _with_n(ds, other):
    """ds with other's n (other: the same draws at n ds has not), matched by case_key, on ds's variables with nn:
    codes by name, other dims padded by position as in the stores; NaN, -1 or '' where a run, code or variable has
    none. Variables without nn (inputs, provenance) stay ds's, but the least-stable-mode globals (min_w_marg_*,
    max_dwdtau_*) are redone over every n."""
    if set(other.nn.values) & set(ds.nn.values):
        raise ValueError(f'more_n: n {sorted(set(other.nn.values) & set(ds.nn.values))} already in the store')
    on_nn = [v for v in ds.data_vars if 'nn' in ds[v].dims]
    keys = other['case_key'].values.astype(object)
    other = other[[v for v in on_nn if v in other and 'nn' in other[v].dims]].load()
    ds, other = (x.assign({v: x[v].compute().astype(object) for v in x.data_vars if x[v].dtype.kind in 'OTU'})
                 for x in (ds, other))                                       # text as objects, as _with_ideal
    fill = {v: _fill(ds[v]) for v in on_nn}
    other = other.assign_coords(run_idx=keys).reindex(run_idx=ds['case_key'].values.astype(object), fill_value=fill)
    other = other.drop_vars('run_idx').reindex(code=ds.code.values, fill_value=fill) if 'code' in other.dims else other
    for d in (set(ds.dims) & set(other.dims)) - {'run_idx', 'nn', 'code'}:
        n = max(ds.sizes[d], other.sizes[d])
        coord = (ds if ds.sizes[d] == n else other)[d]
        ds, other = _pad(ds, d, coord), _pad(other, d, coord)
    joined = xr.concat([ds[on_nn], other], 'nn', join='override', data_vars='all', coords='minimal', compat='override',
                       fill_value=fill)
    ds = xr.merge([ds.drop_dims('nn'), joined], join='override', compat='override')
    if 'min_w_marg_allsurf' not in ds:
        return ds
    cut = np.unique(ds['psi_pedestal_cutoff'].values)
    if cut.size != 1:
        raise ValueError(f'more_n: runs with different psi_pedestal_cutoff {cut}')
    used = ['psi_n_rational', 'Delta_prime_surf', 'w_marg_surf', 'dwdtau_max_surf']
    redone = global_mre_quantities(ds[used].load(), psi_pedestal_cutoff=float(cut[0]))
    return ds.assign({v: redone[v].astype(ds[v].dtype) for v in MRE_GLOBALS if v in ds})


def _pad(ds, d, coord):
    """ds padded at the end of dim d to coord's length (each variable with its _fill), on coord."""
    w = coord.size - ds.sizes[d]
    if not w:
        return ds
    ds = ds.map(lambda da: da.pad({d: (0, w)}, constant_values=_fill(da)) if d in da.dims else da)
    return ds.assign_coords({d: coord})


def _fill(da):
    """The missing value of da's dtype: '' (text), -1 (ints, as the stores' flags) or NaN."""
    return '' if da.dtype.kind in 'OTU' else -1 if da.dtype.kind == 'i' else np.nan


def _with_eps_surf(ds):
    """ds with eps_surf: the inverse aspect ratio (the main pass's eps, else the ideal pass's ideal_eps, on their
    _psi_n) at each run's rational surfaces, cubic as physics.xr_utils.interp_to_surfaces; NaN for runs without it."""
    from scipy.interpolate import interp1d
    name = next((v for v in ('eps', 'ideal_eps') if v in ds), None)
    if name is None or 'psi_n_rational' not in ds:
        return ds
    psi = ds['psi_n_rational'].transpose('run_idx', ...)
    out = np.full(psi.shape, np.nan)
    for i, (x, y) in enumerate(zip(ds[f'{name}_psi_n'].values, ds[name].values)):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 3:
            out[i] = interp1d(x[ok], y[ok], kind='cubic', bounds_error=False)(psi.values[i])
    return ds.assign(eps_surf=psi.copy(data=out))


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
