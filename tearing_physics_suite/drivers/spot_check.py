# Random spot-checks of database runs: re-run a sample of cases through the key 1D input scans (quick values)
# with the database's own settings and n list, to measure how sensitive the database is to numerical settings.
# Used by drivers/multi_run.py (multi_run_(..., spot_check=...)); see README "Spot-checks".

import inspect
import json
import math
import os
import pickle as pkl
from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

from tearing_physics_suite.drivers.input_test_suite import SCANS, run_spec, spot_check_tests
from tearing_physics_suite.drivers.pipeline import analyse_with_mre, nonlinear_resistive_calculation
from tearing_physics_suite.wrappers.run_codes import run_resistive_calculation

SPOT_DIR = 'spot_checks'
WALL_KEYS = ('vac_flag', 'a_wall', 'ishape')
RUN_KEYS = ('run_rdcon', 'run_stride', 'run_pest3', 'run_jgpec')
# Known problems, named in the report so their flags aren't read as new (META_PLAN §5, D).
KNOWN_ISSUES = {
    'pest3_finite_element_scan': 'PEST3 near-edge Delta\' does not converge with nx (D, deferred): pest3 flags near the edge are expected.',
}


@dataclass(frozen=True)
class SpotCheck:
    """Spot-check options: fraction of cases (rounded up), selection seed, Delta' variability thresholds;
    scans replaces the default scan list (spot_check_tests for the run's settings)."""
    fraction: float = 0.01
    seed: int = 0
    abs_threshold: float = 0.05
    rel_threshold: float = 0.05
    scans: tuple | None = None


def as_spot_check(spot_check):
    """None/False -> None, True -> SpotCheck(), SpotCheck -> itself."""
    if spot_check is None or spot_check is False:
        return None
    return SpotCheck() if spot_check is True else spot_check


def _default(func, key):
    return inspect.signature(func).parameters[key].default


def _unquote(v):
    return str(v).strip('\'"')


def db_runs(db_kwargs, key):
    """Whether the database run uses the code switched on by key (run_rdcon, ..., run_jgpec)."""
    return bool(db_kwargs.get(key, _default(run_resistive_calculation, key)))


def db_scans(db_kwargs):
    """Scans spot-checked for this database's settings."""
    return spot_check_tests(db_runs(db_kwargs, 'run_pest3'), _unquote(db_kwargs.get('vac_flag', 't')))


def _not_inherited():
    """Database kwargs a Delta'-only scan doesn't take: profile/MRE settings and output locations."""
    keys = set()
    for func in (nonlinear_resistive_calculation, analyse_with_mre):
        keys |= {k for k, p in inspect.signature(func).parameters.items() if p.kind != p.VAR_KEYWORD}
    return keys | {'tau_e_label', 'working_dir', 'output_location', 'output_prefix'}


def resolve_spec(spec, db_kwargs, nn):
    """(spec, kwargs) for spot-checking a database run: the database's wall setting replaces the spec's;
    the spec's other fixed kwargs win; a spec run_X=True never turns on a code the database didn't run."""
    fixed = {k: v for k, v in spec.fixed.items() if k not in WALL_KEYS}
    for key in RUN_KEYS:
        if fixed.get(key) and not db_runs(db_kwargs, key):
            fixed[key] = False
    drop = _not_inherited() | set(fixed) | set(spec.defaults)
    kwargs = {k: v for k, v in db_kwargs.items() if k not in drop}
    return replace(spec, fixed=fixed, nn=nn), kwargs


def spot_dir(master_working_dir):
    return os.path.join(master_working_dir, SPOT_DIR)


def plan_spot_checks(n_runs, spot, db_kwargs, master_working_dir, warm_start=True):
    """Choose the checked cases (ceil(fraction*n_runs), seeded) and write spot_checks/plan.json.
    A warm start reuses an existing plan, so a restarted run checks the same cases."""
    path = os.path.join(spot_dir(master_working_dir), 'plan.json')
    if warm_start and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    n_spot = min(n_runs, math.ceil(spot.fraction * n_runs))
    idxs = np.random.default_rng(spot.seed).choice(n_runs, n_spot, replace=False)
    plan = dict(spot.__dict__, run_idx=sorted(int(i) for i in idxs), scans=list(spot.scans or db_scans(db_kwargs)),
                nvec=[int(n) for n in db_kwargs.get('nvec') or [1]], n_runs=n_runs)
    os.makedirs(spot_dir(master_working_dir), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(plan, f, indent=1)
    return plan


def spot_tasks(plan, eq_filenames, db_kwargs, spot, warm_start=True):
    """One task per (case, scan, n), for multi_run's pool."""
    return [('spot', idx, eq_filenames[idx], scan, nn, db_kwargs, spot, warm_start)
            for idx in plan['run_idx'] for scan in plan['scans'] for nn in plan['nvec']]


def estimate_spot_check_cost(plan):
    """Number of extra single-n Delta' runs the spot-checks add (prints it)."""
    per_case = sum(len(SCANS[s].quick_vals) for s in plan['scans']) * len(plan['nvec'])
    n = per_case * len(plan['run_idx'])
    print(f"[spot_check] {len(plan['run_idx'])}/{plan['n_runs']} cases checked: {n} extra single-n Delta' runs "
          f"({per_case} per case; a main case is {len(plan['nvec'])}).")
    return n


def _task_paths(master_working_dir, idx, scan, nn):
    base = os.path.join(spot_dir(master_working_dir), f'run_{idx}', f'{scan}_n{nn}')
    return base + '.pkl', base + '.done'


def run_spot_task(args, working_dir, master_working_dir):
    """Worker side of one spot task; returns (('spot', idx, scan, nn), success, error message)."""
    _, idx, eq_filename, scan, nn, db_kwargs, spot, warm_start = args
    key = ('spot', idx, scan, nn)
    pkl_path, done_path = _task_paths(master_working_dir, idx, scan, nn)
    if warm_start and os.path.exists(done_path):
        return key, True, None
    spec, kwargs = resolve_spec(SCANS[scan], db_kwargs, nn)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    result, _ = run_spec(spec, eq_filename, results_dir=os.path.join(os.path.dirname(pkl_path), f'n{nn}'),
                         quick_test=True, working_dir=working_dir, abs_threshold=spot.abs_threshold,
                         rel_threshold=spot.rel_threshold, **kwargs)
    with open(pkl_path, 'wb') as f:
        pkl.dump(dict(result, run_idx=idx, scan=scan, nn=nn, eq_filename=eq_filename), f)
    open(done_path, 'w').close()
    return key, True, None


def _flag(da, codes):
    """Per-code flag (1 exceeded, 0 not, -1 code absent) from a *_thresh_exceeded_psi95_anywhere DataArray."""
    have = dict(zip(np.atleast_1d(da.code.values).tolist(), np.atleast_1d(da.values).tolist()))
    return [int(bool(have[c])) if c in have else -1 for c in codes]


def _spot_dataset(res, codes):
    """One spot's Delta_prime_surf(scan_value, Delta_prime_type, code, surf) and flags; surf is positional."""
    name = res['input_name']
    dp = res['DP_surf_xarray']
    ds = dp[[v for v in ('Delta_prime_surf', 'psi_n_rational') if v in dp]]
    if 'nn' in ds.dims:
        ds = ds.squeeze('nn', drop=True)
    r_value = np.asarray(dp['r_value'].values if 'r_value' in dp else dp.r.values, float)
    ds = ds.reset_coords(drop=True).drop_vars([name, 'r']).rename_dims({name: 'scan_value', 'r': 'surf'})
    ds = ds.reindex(code=codes)
    ds['r_value'] = ('surf', r_value)
    ds['scan_value_str'] = ('scan_value', [str(v) for v in res['input_values']])
    ds['rel_exceeded_psi95'] = ('code', np.int8(_flag(res['rel_thresh_exceeded_psi95_anywhere'], codes)))
    ds['abs_exceeded_psi95'] = ('code', np.int8(_flag(res['abs_thresh_exceeded_psi95_anywhere'], codes)))
    return ds.assign(run_idx=res['run_idx'], scan=res['scan'], input_name=name, nn=res['nn'])


def load_spot_results(master_working_dir):
    """All finished spot pickles, sorted by (run_idx, scan, nn)."""
    root = spot_dir(master_working_dir)
    out = []
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if d.startswith('run_'):
            for f in sorted(os.listdir(os.path.join(root, d))):
                if f.endswith('.pkl'):
                    with open(os.path.join(root, d, f), 'rb') as fh:
                        out.append(pkl.load(fh))
    return sorted(out, key=lambda r: (r['run_idx'], r['scan'], r['nn']))


def compile_spot_checks(master_working_dir, verbose=True):
    """Write spot_checks/spot_checks.zarr and report.txt from the spot pickles; returns (dataset, report)."""
    results = load_spot_results(master_working_dir)
    if not results:
        return None, 'No spot-check results.'
    codes = list(dict.fromkeys(c for r in results for c in np.atleast_1d(r['DP_surf_xarray'].code.values).tolist()))
    ds = xr.concat([_spot_dataset(r, codes) for r in results], dim='spot', join='outer', fill_value={
        'rel_exceeded_psi95': -1, 'abs_exceeded_psi95': -1})
    ds = ds.assign_coords(code=codes)
    path = os.path.join(spot_dir(master_working_dir), 'spot_checks.zarr')
    ds.to_zarr(path, mode='w', consolidated=True)
    report = spot_report(ds, results, master_working_dir)
    with open(os.path.join(spot_dir(master_working_dir), 'report.txt'), 'w') as f:
        f.write(report)
    if verbose:
        print(report)
    return ds, report


def spot_report(ds, results, master_working_dir):
    """Text report: flagged runs (scan, n, codes), per-code Delta' spread, known issues, STRIDE/RDCON disagreements."""
    plan_path = os.path.join(spot_dir(master_working_dir), 'plan.json')
    rel = float(json.load(open(plan_path))['rel_threshold']) if os.path.exists(plan_path) else np.nan
    codes = [str(c) for c in ds.code.values]
    lines = [f'Spot-checks: {len(set(ds.run_idx.values.tolist()))} runs, {ds.sizes["spot"]} (run, scan, n) checks.']
    flagged = ds.rel_exceeded_psi95 == 1
    lines.append(f'Flagged (Delta\' relative change across core modes > {rel:g}):')
    for s in np.flatnonzero(flagged.any('code').values):
        sp = ds.isel(spot=s)
        fc = [c for c, f in zip(codes, sp.rel_exceeded_psi95.values) if f == 1]
        note = f'  [known: {KNOWN_ISSUES[str(sp.scan.values)]}]' if str(sp.scan.values) in KNOWN_ISSUES else ''
        lines.append(f'  run {int(sp.run_idx)}  {sp.scan.values}  n={int(sp.nn)}  codes {fc}{note}')
    if not flagged.any():
        lines.append('  none')
    lines.append('Largest relative Delta\' spread across a scan, core modes (psi_n < 0.95), single helicity:')
    for r in results:
        dp = r['DP_surf_xarray']
        var = f'Delta_prime_reldiff_across_{r["input_name"]}'
        if var not in dp:
            continue
        core = dp[var].sel(Delta_prime_type='single helicity') if 'Delta_prime_type' in dp[var].dims else dp[var]
        core = core.where(dp.psi_n_rational.mean(r['input_name']) < 0.95)
        spread = core.max([d for d in core.dims if d != 'code'])
        txt = ', '.join(f'{c} {float(v):.3g}' for c, v in zip(spread.code.values, spread.values) if np.isfinite(v))
        lines.append(f'  run {r["run_idx"]}  {r["scan"]}  n={r["nn"]}: {txt}')
    lines.append(f'STRIDE vs RDCON lowest-m Delta\' disagreement > {rel:g} (candidate examples for item G):')
    g = stride_rdcon_disagreements(results, rel)
    lines += [f'  run {i}  n={n}  rdcon {a:.4g}  stride {b:.4g}' for i, n, a, b in g] or ['  none']
    return '\n'.join(lines) + '\n'


def stride_rdcon_disagreements(results, rel_threshold):
    """(run_idx, nn, rdcon, stride) where the first-surface single-helicity Delta' of rdcon and stride differ by
    more than rel_threshold at the first value of a scan that ran both; one entry per (run, n)."""
    seen, out = set(), []
    for r in results:
        dp = r['DP_surf_xarray'].Delta_prime_surf
        if (r['run_idx'], r['nn']) in seen or not {'rdcon', 'stride'} <= set(dp.code.values.tolist()):
            continue
        dp = dp.isel({r['input_name']: 0, 'r': 0, **({'nn': 0} if 'nn' in dp.dims else {})})
        if 'Delta_prime_type' in dp.dims:
            dp = dp.sel(Delta_prime_type='single helicity')
        a, b = float(dp.sel(code='rdcon')), float(dp.sel(code='stride'))
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        seen.add((r['run_idx'], r['nn']))
        if 2 * abs(a - b) / (abs(a) + abs(b)) > rel_threshold:
            out.append((r['run_idx'], r['nn'], a, b))
    return out


def spot_check_flag(spot_ds, idx, nn_vals, codes):
    """spot_check_flag(nn, code) for main run idx: -1 not checked, 0 passed, 1 a scan exceeded rel_threshold."""
    out = np.full((len(nn_vals), len(codes)), -1, dtype=np.float32)
    if spot_ds is None:
        return out
    sel = spot_ds.where(spot_ds.run_idx == idx, drop=True)
    for i, nn in enumerate(nn_vals):
        flags = sel.rel_exceeded_psi95.where(sel.nn == nn, drop=True)
        for j, c in enumerate(codes):
            if c in flags.code.values and flags.sizes['spot']:
                f = flags.sel(code=c).values
                out[i, j] = f.max() if (f >= 0).any() else -1
    return out
