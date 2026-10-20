"""Study mode: TPS on bouquet archives in a shared study dir, on any number of nodes and jobs.

    python -m tearing_physics_suite.drivers.parallel init    --study S --config cfg.json
    python -m tearing_physics_suite.drivers.parallel feed    --study S
    python -m tearing_physics_suite.drivers.parallel work    --study S --workers W [--feed-every 300]
    python -m tearing_physics_suite.drivers.parallel compile --study S [--shots ...] [--finalize]
    python -m tearing_physics_suite.drivers.parallel status  --study S

Work items (``<unit_id>.c<k>``, a chunk of one bouquet unit's draws) are claimed from the fsqueue at
``<study>/_queue/tps`` and run in child processes. Each draw writes ``draws/<count>.nc`` and, last,
``draws/<count>.json`` (its status). See docs/parallel.md for the layout and the failure policy.

With ``--code-check``, feed/work/compile/status act on the code checks (config ``code_check``): the other codes on
the 'ok' draws of a seeded random fraction of shots or draws (STRIDE-flagged draws first, further draws when the
workers would idle), from ``<study>/_queue/tps_codes``, filled into ``draws/<count>.codes.nc`` (status
``.codes.json``) and compiled into ``<shot>_codes.zarr``. ``code-check-pending`` lists the draw files to keep for them.
"""
import argparse
import collections
import functools
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import zlib

import numpy as np

from tearing_physics_suite.drivers import fsqueue as fq
from tearing_physics_suite.utils import CODE_WALL_S, tps_home, work_root

DEFAULTS = dict(
    tps_kwargs={},              # nonlinear_resistive_calculation kwargs (nvec, run_stride, run_jgpec, vac_flag, ...)
    variants=['base'],          # bouquet variants to run
    selection='selected',       # 'selected' | 'all' draws of each archive
    eq_source='geqdsk',         # 'geqdsk' | 'ifile'
    draws_per_item=8,
    ida_extra_keys=[],
    meta_data=False,            # bq_* per-draw metadata in the profile dicts
    main_ion_mass_amu=2.0,      # species model masses (amu): main ions (thermal and fast) and the one impurity
    impurity_mass_amu=None,     # (None: 2 Z_imp)
    keep_pickle=False,          # also keep multi_run-style result pickles
    code_timeout=3600,          # s per Fortran code run (TPS_CODE_TIMEOUT)
    out_root=None,              # per-draw outputs under <out_root>/<shot>/tps (default: the study dir); $VARS expanded
    n_same_sig=6,               # this many draw failures in a row with one signature, over >= 2 units: worker broken
    tau_e=None,                 # {"path": I2 netCDF(s), ':'-separated globs, $VARS expanded, "variable": name}: the
                                # energy confinement time per draw (see tau_e_lookup)
    baseline=True,              # also run each archive's bouquet baseline, as draw -1 (item <unit>.base)
    spot_check=None,            # true or SpotCheck fields: re-run a seeded random fraction of the draws through the key
                                # input scans (drivers/spot_check.py); the store gets spot_check_flag(nn, code)
    code_check=None,            # {"by": "shot" | "draw", "fraction", "seed", "codes": [...], "nvec": [...], "flag":
                                # {"im_re_max", "psi_n": [lo, hi]}}: run these codes at these n (default: the study's)
                                # on the 'ok' draws of a seeded random fraction of shots or draws, filled in
                                # (--code-check); with flag, STRIDE-flagged draws first (see stride_flagged_n)
)
FINAL = ('ok', 'error', 'crashed')
MAX_DRAW_ATTEMPTS = 2           # a timed-out or crashed draw is retried once
INFRA = (ImportError, MemoryError)
INFRA_ERRNO = {28, 30, 122}     # ENOSPC, EROFS, EDQUOT


# --- paths and config -------------------------------------------------------------------------

def qroot(study, code_check=False):
    return os.path.join(study, '_queue', 'tps_codes' if code_check else 'tps')


def config_path(study):
    return os.path.join(qroot(study), 'study.json')


def load_config(study):
    cfg = fq.read_json(config_path(study))
    if cfg is None:
        raise FileNotFoundError(f'{config_path(study)}: run `study init` first')
    return cfg


def out_root(study, cfg):
    """Root of the per-draw outputs: cfg['out_root'] ($VARS expanded), else the study dir."""
    return os.path.expandvars(cfg.get('out_root') or study)


def unit_dir(study, cfg, shot, unit_id):
    return os.path.join(out_root(study, cfg), str(shot), 'tps', unit_id)


def shot_store(study, shot, code_check=False):
    return os.path.join(study, str(shot), 'tps', f'{shot}_codes.zarr' if code_check else f'{shot}.zarr')


def open_store(path):
    """A compiled store, from its published one-file copy `<path>.zip` when that exists, else the directory."""
    import xarray as xr
    import zarr
    zp = path + '.zip'
    return xr.open_zarr(zarr.storage.ZipStore(zp, mode='r') if os.path.exists(zp) else path, consolidated=True)


def _describe(path):
    try:
        return subprocess.run(['git', '-C', path, 'describe', '--always', '--dirty'], capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def versions():
    """git describes of the TPS package and its Fortran codes (None outside a checkout)."""
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        home = tps_home()
    except KeyError:
        home = pkg
    return dict(tps=_describe(pkg), gpec=_describe(os.path.join(home, 'submodules', 'GPEC')),
                jgpec=_describe(os.environ.get('JGPEC_HOME', '/fusion/projects/tmdb/src/GPEC')))


def cmd_init(a):
    cfg = dict(DEFAULTS)
    if a.config:
        with open(a.config) as fh:
            cfg.update(json.load(fh))
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f'unknown config keys {sorted(unknown)}')
    if cfg['tau_e']:
        tau_e_lookup(cfg['tau_e'])                       # fail now, not in every work item
    spot_config(cfg)
    code_check_config(cfg)
    cfg['hash'] = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    cfg['versions'] = versions()
    old = fq.read_json(config_path(a.study))
    if old and old['hash'] != cfg['hash'] and not a.force:
        raise SystemExit(f'{config_path(a.study)} exists with other settings (hash {old["hash"]}); --force to replace')
    fq.Queue(qroot(a.study))
    fq.write_json(config_path(a.study), cfg)
    print(f'init: {config_path(a.study)} (hash {cfg["hash"]})')


def check_build(cfg, allow_drift=False):
    """Refuse to run on a build other than the one the study was initialised with."""
    now, then = versions(), cfg.get('versions', {})
    drift = {k: (then.get(k), v) for k, v in now.items() if v and then.get(k) and v != then.get(k)}
    if drift and not allow_drift:
        raise fq.Broken(f'build differs from the study config: {drift} (--allow-build-drift to run anyway)')


def spot_config(cfg):
    """The study's SpotCheck, or None; unknown fields or scans raise."""
    from tearing_physics_suite.drivers import spot_check as sc
    s = cfg.get('spot_check')
    if not s:
        return None
    spot = sc.SpotCheck() if s is True else sc.SpotCheck(**dict(s, scans=tuple(s['scans']) if s.get('scans') else None))
    unknown = (set(spot.scans or ()) | set(spot.scan_vals or ())) - set(sc.SCANS)
    if unknown:
        raise ValueError(f'spot_check: unknown scans {sorted(unknown)}')
    return spot


def code_check_config(cfg):
    """The study's code check {by, fraction, seed, codes, nvec, flag} (defaults filled in), or None; bad values
    raise."""
    from tearing_physics_suite.physics.combine import CODE_ORDER
    c = cfg.get('code_check')
    if not c:
        return None
    defaults = dict(by='draw', fraction=0.01, seed=0, codes=[], nvec=None, flag=None)
    c = dict(defaults, **c)
    study_n = cfg['tps_kwargs'].get('nvec') or [1]
    flag_ok = c['flag'] is None or (set(c['flag']) == {'im_re_max', 'psi_n'}
                                    and cfg['tps_kwargs'].get('run_stride', True))     # the flag reads STRIDE's Delta'
    if (set(c) - set(defaults) or c['by'] not in ('shot', 'draw') or not c['codes'] or not flag_ok
            or set(c['codes']) - (set(CODE_ORDER) - {'stride'}) or not set(c['nvec'] or study_n) <= set(study_n)):
        raise ValueError(f"code_check: keys {sorted(defaults)}, by 'shot' or 'draw', codes from "
                         f"{[k for k in CODE_ORDER if k != 'stride']}, nvec from the study's {study_n}, flag "
                         f"{{im_re_max, psi_n: [lo, hi]}} on a STRIDE study: {c}")
    return c


def stride_flagged_n(ds, flag):
    """The n at which a draw's STRIDE single-helicity Delta' is suspect: |Im| > im_re_max |Re|, or no Re, on a
    rational surface with psi_n[0] < psi_N < psi_n[1] (RDCON's surface map; a draw without STRIDE has no Delta')."""
    sh = ds[['Delta_prime_surf', 'Im_Delta_prime_surf', 'psi_n_rational']].sel(Delta_prime_type='single helicity')
    sh = sh.reindex(code=['rdcon', 'stride'])
    re, im = sh.Delta_prime_surf.sel(code='stride'), sh.Im_Delta_prime_surf.sel(code='stride')
    psi = sh.psi_n_rational.sel(code='rdcon')
    bad = (re.isnull() | (abs(im) > flag['im_re_max'] * abs(re))) & (psi > flag['psi_n'][0]) & (psi < flag['psi_n'][1])
    return [int(n) for n in ds.nn.values[bad.any('r').values]]


def code_check_kwargs(cfg, check):
    """The study's tps_kwargs for a code check: only the check's codes, and RDCON (whose surfaces every code uses),
    at the check's n (every code there, Julia included)."""
    solvers = [c.removeprefix('jGPEC_') for c in check['codes'] if c.startswith('jGPEC_')]
    n = dict(nvec=check['nvec'], jgpec_nvec=check['nvec']) if check.get('nvec') else {}
    return dict(cfg['tps_kwargs'], run_rdcon=True, rdcon_delta_prime='rdcon' in check['codes'], run_stride=False,
                run_pest3='pest3' in check['codes'], run_jgpec=bool(solvers), jgpec_solvers=solvers, **n)


# --- feed --------------------------------------------------------------------------------------

def _archive_counts(path, selection):
    import h5py
    with h5py.File(path, 'r') as f:
        out = []
        for key in f['scan']:
            sc = f['scan'][key]
            out += [int(c) for c in sc if c.isdigit() and (selection == 'all' or sc[c].attrs.get('selected', False))]
    return sorted(out)


def feed(study, cfg=None):
    """Add work items for bouquet units recorded 'done' and not yet fed; returns how many were added.

    The baseline is its own item (<unit>.base), so it also reaches units fed before it was enabled.
    """
    cfg = cfg or load_config(study)
    q = fq.Queue(qroot(study))
    units = q.units(refresh=True)
    fed = {u.get('unit_id') for u in units.values()}
    items = []
    for rec_path in sorted(glob.glob(os.path.join(study, '*', 'bouquet', '*.json'))):
        r = fq.read_json(rec_path) or {}
        uid = r.get('unit_id')
        if r.get('status') != 'done' or r.get('variant') not in cfg['variants']:
            continue
        archive = os.path.join(os.path.dirname(rec_path), r['archive'])
        if cfg.get('baseline', True) and f'{uid}.base' not in units:
            items.append(dict(id=f'{uid}.base', unit_id=uid, shot=r['shot'], t_ms=r['t_ms'], variant=r['variant'],
                              archive=archive, draws=[-1], est=1))
        if uid in fed:
            continue
        counts = r.get('selected_counts' if cfg['selection'] == 'selected' else 'all_counts')
        if counts is None:
            counts = _archive_counts(archive, cfg['selection'])
        n = int(cfg['draws_per_item'])
        for k in range(0, len(counts), n):
            items.append(dict(id=f'{uid}.c{k // n}', unit_id=uid, shot=r['shot'], t_ms=r['t_ms'],
                              variant=r['variant'], archive=archive, draws=counts[k:k + n],
                              est=len(counts[k:k + n])))
        if not counts:
            print(f'feed: {uid} has no {cfg["selection"]} draws', flush=True)
    n_new = q.add_units(items)
    print(f'feed: {n_new} item(s) added', flush=True)
    return n_new


def _coin(seed, key):
    """A seeded uniform number per key (draw or shot), the same however the draws are chunked or retried; the key is
    picked when it is below the fraction."""
    return np.random.default_rng([seed, zlib.crc32(key.encode())]).random()


def _pick(check, r):
    """(coin, reason) of an 'ok' draw's code check, from its status record: 'flagged', 'random' (picked by the coin),
    'flagged+random', or 'extra'."""
    u = _coin(check['seed'], str(r['shot']) if check['by'] == 'shot' else r['case_key'])
    return u, '+'.join(k for k, on in (('flagged', r.get('stride_flagged_n')), ('random', u < check['fraction']))
                       if on) or 'extra'


def feed_codes(study, cfg=None, extra=0):
    """Add code-check items (<unit_id>.d<count>) for 'ok' draws not yet fed, by priority: the STRIDE-flagged draws
    (reason 'flagged'); once none of those waits for a worker, the picked shots' or draws' ('random', 'flagged+random'
    when both); once neither waits, ``extra`` further draws in the coin's order ('extra': the work feeder passes its
    worker count, so the check sessions keep going). Returns how many were added."""
    cfg = cfg or load_config(study)
    check = code_check_config(cfg)
    if check is None:
        raise SystemExit('feed --code-check: the study has no code_check')
    q = fq.Queue(qroot(study, code_check=True))
    units, done = q.units(refresh=True), q.done_ids()
    waiting = {(it.get('reason') or 'random').split('+')[0] for uid, it in units.items()
               if uid not in done and q.age('claims', uid) >= q.STALE_S}
    new = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(out_root(study, cfg), '*', 'tps', '*', 'draws', '*.json'))):
        if p.endswith('.codes.json'):
            continue
        r = fq.read_json(p) or {}
        uid = f"{r.get('unit_id')}.d{r.get('draw')}"
        if r.get('status') != 'ok' or uid in units or not os.path.exists(p[:-len('.json')] + '.nc'):   # cleaned
            continue
        u, reason = _pick(check, r)
        new[reason.split('+')[0]].append((u, uid, dict(
            id=uid, **{k: r[k] for k in ('unit_id', 'shot', 't_ms', 'variant', 'archive')}, draws=[r['draw']], est=1,
            reason=reason)))
    for tier in ('flagged', 'random', 'extra'):
        items = sorted(new[tier])[:extra] if tier == 'extra' else new[tier]
        if items or tier in waiting:
            break
    n_new = q.add_units([it for _, _, it in items])
    print(f'feed: {n_new} code-check item(s) added', flush=True)
    return n_new


class Feeder:
    """on_idle hook: feed, unless any worker fed in the last ``every`` seconds (code checks: up to ``workers`` extra
    draws)."""

    def __init__(self, study, every, code_check=False, workers=0):
        self.study, self.every, self.code_check, self.workers = study, every, code_check, workers

    def __call__(self):
        fn = functools.partial(feed_codes, extra=self.workers) if self.code_check else feed
        try:
            return bool(fq.Queue(qroot(self.study, self.code_check)).periodic('feeder', self.every,
                                                                              lambda: fn(self.study) > 0))
        except Exception as e:
            print(f'feeder failed: {fq.error_signature(e)}', flush=True)
            return False


# --- one item (child process) ------------------------------------------------------------------

def _run_spot_checks(spot, eq, kwargs, sdir, wd):
    """Spot-check one draw into sdir (its run_idx is 0); returns {'<scan>_n<n>': error} of the failed tasks."""
    from tearing_physics_suite.drivers import spot_check as sc
    errors = {}
    for task in sc.spot_tasks(sc.plan_spot_checks(1, spot, kwargs, sdir), [eq], kwargs, spot):
        try:
            sc.run_spot_task(task, wd, sdir)
        except Exception as e:
            errors[f'{task[3]}_n{task[4]}'] = fq.error_signature(e)
    return errors


def _draw_state(path):
    """(final, attempts) of a draw's status file."""
    r = fq.read_json(path)
    if r is None:
        return False, 0
    n = r.get('attempts', 1)
    return r.get('status') in FINAL or n >= MAX_DRAW_ATTEMPTS, n


def _scalars(d):
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in (d or {}).items()
            if isinstance(v, (bool, int, float, str, np.integer, np.floating))}


def tau_e_lookup(spec):
    """(shot, t [s]) -> spec['variable'] of the I2 file(s) holding that shot, at the sample nearest t, else None.

    I2 (bouquet_prep_omega.sh): dims (shot, <base>_idx). The time base is spec['time'], else the variable named
    '*time*' on that index ('time' for time_idx, 'eqdsk_time_efit01' for eqdsk_idx_efit01); 'ms' units are read from
    its attrs (else s). Samples further than spec['tol_ms'] (default 50) from t give None.
    """
    import xarray as xr
    paths = sorted(p for pat in os.path.expandvars(spec['path']).split(':') for p in glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"tau_e: no file matches {spec['path']}")
    var, tol = spec['variable'], spec.get('tol_ms', 50) / 1e3
    files = {}
    for p in paths:
        ds = xr.open_dataset(p)
        if var not in ds:
            raise KeyError(f"tau_e: no variable {var!r} in {p}")
        base = next(d for d in ds[var].dims if d != 'shot')
        tname = spec.get('time') or next(v for v in ds.variables if 'time' in v and v != var
                                         and ds[v].dims in ((base,), ('shot', base)))
        files[p] = (ds, tname, 1e-3 if ds[tname].attrs.get('units') == 'ms' else 1.0,
                    {int(x): i for i, x in enumerate(ds.shot.values)})

    def lookup(shot, t):
        for ds, tname, scale, row in files.values():
            if shot in row:
                tv = ds[tname]
                times = np.asarray((tv.isel(shot=row[shot]) if 'shot' in tv.dims else tv).values, float) * scale
                vals = np.asarray(ds[var].isel(shot=row[shot]).values, float)
                ok = np.isfinite(times) & np.isfinite(vals)
                if not ok.any():
                    return None
                i = int(np.argmin(np.abs(times[ok] - t)))
                return float(vals[ok][i]) if abs(times[ok][i] - t) <= tol else None
        return None
    return lookup


def run_item(study, item_id, code_check=False):
    """Run the draws of one work item (a code-check item: their code checks) that are not final yet.
    Returns a summary dict."""
    from tearing_physics_suite.drivers.multi_run import run_case
    from tearing_physics_suite.drivers.profile_read import read_bouquet_archive
    cfg = load_config(study)
    it = fq.Queue(qroot(study, code_check)).units()[item_id]
    if cfg.get('code_timeout'):
        os.environ['TPS_CODE_TIMEOUT'] = str(cfg['code_timeout'])
    ddir = os.path.join(unit_dir(study, cfg, it['shot'], it['unit_id']), 'draws')
    os.makedirs(ddir, exist_ok=True)
    sfx = '.codes' if code_check else ''
    states = {c: _draw_state(os.path.join(ddir, f'{c}{sfx}.json')) for c in it['draws']}
    for c, (final, _) in states.items():                 # a draw left 'running' crashed its process twice
        r = fq.read_json(os.path.join(ddir, f'{c}{sfx}.json')) or {}
        if final and r.get('status') == 'running':
            fq.write_json(os.path.join(ddir, f'{c}{sfx}.json'), dict(r, status='crashed'))
    todo = [c for c, (final, _) in states.items() if not final]
    out = dict(item=item_id, draws=[])
    if not todo:
        return out
    kwargs = code_check_kwargs(cfg, code_check_config(cfg)) if code_check else dict(cfg['tps_kwargs'])
    spot = None if code_check else spot_config(cfg)
    flag = None if code_check else (code_check_config(cfg) or {}).get('flag')
    if cfg.get('tau_e'):
        kwargs['energy_confinement_time'] = tau_e_lookup(cfg['tau_e'])(it['shot'], it['t_ms'] / 1e3)
    wd = tempfile.mkdtemp(prefix=f'tps_{item_id}_', dir=work_root())
    try:
        eqs, profs = read_bouquet_archive(it['archive'], selection='all', draw_counts=todo,
                                          eqdsk_out_dir=os.path.join(wd, 'eq'), eq_source=cfg['eq_source'],
                                          ida_extra_keys=cfg['ida_extra_keys'], meta_data=cfg['meta_data'],
                                          main_ion_mass_amu=cfg['main_ion_mass_amu'],
                                          impurity_mass_amu=cfg['impurity_mass_amu'])
        for eq, prof in zip(eqs, profs):
            out['draws'].append(_run_draw(it, cfg, kwargs, ddir, eq, prof, states[prof['bq_count']][1] + 1,
                                          os.path.join(wd, 'work'), _check_case if code_check else run_case,
                                          spot, sfx, flag))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    if any(d['status'] in ('timeout',) for d in out['draws']):
        raise RuntimeError('draw timeout(s), to retry: ' + ', '.join(str(d['draw']) for d in out['draws']
                                                                   if d['status'] == 'timeout'))
    return out


def _run_draw(it, cfg, kwargs, ddir, eq, prof, attempt, wd, run_case, spot=None, sfx='', flag=None):
    c = prof['bq_count']
    key = f"{it['unit_id']}_{c}"
    stat = os.path.join(ddir, f'{c}{sfx}.json')
    rec = dict(case_key=key, shot=it['shot'], t_ms=it['t_ms'], variant=it['variant'], unit_id=it['unit_id'],
               draw=c, archive=it['archive'], profile_coord=prof.get('bq_profile_coord'), attempts=attempt,
               config_hash=cfg['hash'], host=socket.gethostname(), slurm_job=os.environ.get('SLURM_JOB_ID', ''),
               started=time.time())
    if 'reason' in it:                                                  # a code check's
        rec['reason'] = it['reason']
    fq.write_json(stat, dict(rec, status='running'))
    os.makedirs(wd, exist_ok=True)
    CODE_WALL_S.clear()
    try:
        if cfg.get('tau_e') and kwargs.get('energy_confinement_time') is None:
            raise LookupError(f"no {cfg['tau_e']['variable']} for shot {it['shot']} at {it['t_ms']} ms")
        ds, inputs = run_case(eq, prof, kwargs, wd, nc_path=os.path.join(ddir, f'{c}{sfx}.nc'),
                              pkl_path=os.path.join(ddir, f'{c}{sfx}.pkl') if cfg['keep_pickle'] else None, label=key,
                              labels=dict(shot_id=it['shot'], t_ms=it['t_ms'], draw=c, variant=it['variant']))
        rec.update(status='ok' if ds is not None else 'error', inputs=_scalars(inputs), code_wall_s=_code_wall_s(),
                   error=None if ds is not None else dict(signature='no combined_xr'))
        if ds is not None and flag:                                         # never fails the draw
            try:
                rec['stride_flagged_n'] = stride_flagged_n(ds, flag)
            except Exception as e:                                          # unreadable: no Delta', every n
                rec.update(stride_flagged_n=ds.nn.values.tolist(), stride_flag_error=fq.error_signature(e))
        if ds is not None and spot and _coin(spot.seed, key) < spot.fraction:      # never fails the draw
            t = time.time()
            rec['spot_errors'] = _run_spot_checks(spot, eq, kwargs, os.path.join(ddir, f'{c}.spot'), wd)
            rec['spot_wall_s'] = round(time.time() - t, 1)
    except TimeoutError as e:
        rec.update(status='timeout', error=_err(e))
    except Exception as e:
        if isinstance(e, INFRA) or (isinstance(e, OSError) and e.errno in INFRA_ERRNO):
            os.remove(stat)
            raise
        rec.update(status='error', error=_err(e))
    finally:
        for f in os.listdir(wd):
            p = os.path.join(wd, f)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    rec.setdefault('code_wall_s', _code_wall_s())                  # a failed draw: the codes it ran
    rec['wall_s'] = round(time.time() - rec['started'], 1)
    rec['other_wall_s'] = round(rec['wall_s'] - sum(rec['code_wall_s'].values()) - rec.get('spot_wall_s', 0.0), 1)
    fq.write_json(stat, rec)
    return dict(draw=c, unit_id=it['unit_id'], status=rec['status'], signature=(rec['error'] or {}).get('signature'))


def _check_case(eq, prof, kwargs, wd, nc_path, pkl_path=None, label=None, labels=None):
    """run_case for a code check: the check's codes on the draw's inputs, filled into its dataset (<count>.nc), the
    result written to nc_path (<count>.codes.nc)."""
    import xarray as xr

    from tearing_physics_suite.drivers import multi_run
    from tearing_physics_suite.drivers.pipeline import fill_codes
    other, inputs = multi_run.run_case(eq, prof, kwargs, wd, os.path.join(wd, 'codes.nc'), pkl_path, label, labels)
    if other is None:
        return None, inputs
    with xr.open_dataset(nc_path.removesuffix('.codes.nc') + '.nc', engine='scipy') as ds:
        ds = fill_codes(ds.load().drop_encoding(), other)        # the file's int dtypes would not hold NaN padding
    tmp = f'{nc_path}.tmp.{os.getpid()}'
    ds.to_netcdf(tmp, engine='scipy')
    os.replace(tmp, nc_path)
    return ds, inputs


def _code_wall_s():
    """The draw's code wall times so far ({'rdcon_n1': s, ..., 'jGPEC_start': s})."""
    return {k: round(v, 1) for k, v in CODE_WALL_S.items()}


def _timing_vars(rec, nn, codes):
    """A draw's wall times for the store: code_wall_s(nn, code), and its total, Julia start-up, spot checks and the
    rest (TPS input/output and MRE); NaN for a record written before timing."""
    w = rec.get('code_wall_s')
    code_wall = np.array([[(w or {}).get(f'{c}_n{n}', np.nan) for c in codes] for n in nn], dtype=np.float32)
    zero = 0.0 if w is not None else np.nan
    return dict(code_wall_s=(('nn', 'code'), code_wall), draw_wall_s=np.float32(rec.get('wall_s', np.nan)),
                other_wall_s=np.float32(rec.get('other_wall_s', np.nan)),
                spot_wall_s=np.float32(rec.get('spot_wall_s', zero)),
                jgpec_start_wall_s=np.float32((w or {}).get('jGPEC_start', zero)))


def _err(e):
    return dict(type=type(e).__name__, signature=fq.error_signature(e), message=str(e)[:2000],
                traceback_tail=traceback.format_exc()[-4000:])


def cmd_run_item(a):
    res = a.result or os.path.join(qroot(a.study, a.code_check), 'results', f'{a.item}.json')
    os.makedirs(os.path.dirname(res), exist_ok=True)
    try:
        fq.write_json(res, run_item(a.study, a.item, a.code_check))
    except BaseException as e:
        traceback.print_exc()
        infra = isinstance(e, INFRA) or (isinstance(e, OSError) and e.errno in INFRA_ERRNO)
        fq.write_json(res, dict(error=dict(type=type(e).__name__, signature=fq.error_signature(e), infra=infra)))
        sys.exit(1)


# --- worker (parent) ---------------------------------------------------------------------------

class Runner:
    """Picklable ``make_run`` for fsqueue.launch: each item runs as a run-item child process."""

    def __init__(self, a):
        self.study, self.timeout, self.min_left = a.study, a.item_timeout, a.min_left
        self.allow_drift, self.deadline = a.allow_build_drift, fq.job_deadline(0.0)
        self.compile = not getattr(a, 'no_compile', False)
        self.code_check = getattr(a, 'code_check', False)

    def __call__(self, q, worker):
        cfg = load_config(self.study)
        recent = collections.deque(maxlen=cfg['n_same_sig'])

        checked = []

        def run(uid, item):
            if not checked:
                check_build(cfg, self.allow_drift)
                checked.append(True)
            if self.deadline and self.deadline - time.time() < self.min_left:
                raise fq.Requeue('too little walltime left')
            log = os.path.join(unit_dir(self.study, cfg, item['shot'], item['unit_id']), 'logs', f'{uid}.log')
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, 'a') as fh:
                fh.write(f'\n=== {time.ctime()} {worker} attempt {q.n_attempts(uid)}\n')
            res_path = os.path.join(q.root, 'results', f'{uid}.{worker}.{q.n_attempts(uid)}.json')
            argv = [sys.executable, '-m', 'tearing_physics_suite.drivers.parallel', 'run-item',
                    '--study', self.study, '--item', uid, '--result', res_path]
            if self.code_check:
                argv.append('--code-check')
            try:
                res = fq.run_child(argv, log, timeout=self.timeout, result_path=res_path)
            except RuntimeError:
                if (fq.read_json(res_path) or {}).get('error', {}).get('infra'):
                    raise fq.Broken((fq.read_json(res_path) or {})['error']['signature']) from None
                raise
            for d in res.get('draws', []):
                if d['status'] == 'ok':
                    recent.clear()
                else:
                    recent.append((d['signature'], d['unit_id']))
            if (len(recent) == recent.maxlen and len({s for s, _ in recent}) == 1
                    and len({u for _, u in recent}) >= 2):
                raise fq.Broken(f'{recent.maxlen} draws in a row failed with: {recent[0][0]}')
            n = collections.Counter(d['status'] for d in res.get('draws', []))
            if self.compile and _shot_finished(q, item['shot'], uid):
                try:
                    compile_shot(self.study, item['shot'], cfg, code_check=self.code_check)
                except Exception as e:
                    print(f'[{worker}] compile of shot {item["shot"]} failed: {fq.error_signature(e)}', flush=True)
            return dict(n_draws=len(res.get('draws', [])), **n)
        return run


def _shot_finished(q, shot, current):
    """True if every queued item of the shot except ``current`` is terminal."""
    done = q.done_ids()
    return all(uid in done for uid, it in q.units().items() if it.get('shot') == shot and uid != current)


def cmd_work(a):
    d = fq.job_deadline(a.margin)
    print(f'work: {a.workers} worker(s) on {socket.gethostname()}, deadline {time.ctime(d) if d else "none"}',
          flush=True)
    out = fq.launch(qroot(a.study, a.code_check), a.workers, Runner(a), deadline=d, idle_exit_s=a.idle_exit,
                    on_idle=Feeder(a.study, a.feed_every, a.code_check, a.workers) if a.feed_every else None)
    for w, why in sorted(out.items()):
        print(f'  {w}: {why}', flush=True)
    if not a.no_compile:          # catch draws that finished while their shot was being compiled
        fq.Queue(qroot(a.study, a.code_check)).periodic('compile_all', 600, lambda: [
            compile_shot(a.study, s, code_check=a.code_check) for s in study_shots(a.study)])


# --- compile -----------------------------------------------------------------------------------

def _shot_draws(study, cfg, shot, code_check=False):
    """{case_key: (nc path, status record)} of the shot's draws (code checks) with status 'ok'."""
    out = {}
    root = os.path.join(out_root(study, cfg), str(shot), 'tps')
    for p in glob.glob(os.path.join(root, '*', 'draws', '*.json')):
        if p.endswith('.codes.json') != code_check:
            continue
        r = fq.read_json(p) or {}
        if r.get('status') == 'ok':
            out[r['case_key']] = (p[:-len('.json')] + '.nc', r)
    return out


def code_check_pending(study, shot, cfg=None):
    """The <count>.nc of the shot's 'ok' draws whose code check is still to come (queued and not finished, or not yet
    queued and flagged or picked), which must stay until it is done."""
    cfg = cfg or load_config(study)
    check = code_check_config(cfg)
    if check is None:
        return []
    q = fq.Queue(qroot(study, code_check=True))
    units, done = q.units(refresh=True), q.done_ids()
    out = []
    for nc, r in _shot_draws(study, cfg, shot).values():
        uid = f"{r['unit_id']}.d{r['draw']}"
        if (uid in units and uid not in done) or (uid not in units and _pick(check, r)[1] != 'extra'):
            out.append(nc)
    return sorted(out)


def cmd_code_check_pending(a):
    for shot in a.shots or study_shots(a.study):
        for nc in code_check_pending(a.study, shot):
            print(nc)


def _spot_flag(nc_path, rec, nn, codes):
    """spot_check_flag(nn, code) of one draw (-1 where it was not checked)."""
    from tearing_physics_suite.drivers import spot_check as sc
    sdir = nc_path[:-len('.nc')] + '.spot'
    ds = sc.spot_dataset(sc.load_spot_results(sdir)) if 'spot_errors' in rec else None
    return sc.spot_check_flag(ds, 0, nn, codes)


def _stride_flag(rec, nn):
    """stride_flag(nn) of one draw: 1 where STRIDE was flagged, 0 where not, -1 for a record without the flag."""
    f = rec.get('stride_flagged_n')
    return np.array([-1 if f is None else int(n in f) for n in nn], dtype=np.int8)


def compile_shot(study, shot, cfg=None, finalize=False, mb_per_chunk=10, batch=32, code_check=False):
    """Append the shot's new 'ok' draws to <study>/<shot>/tps/<shot>.zarr (code checks: to <shot>_codes.zarr) (single
    writer, under a lease).

    Draws go in ``batch`` at a time, one store write each: a write costs a few file operations per variable,
    whatever the number of draws. A store left longer than compile_state.json by a killed compile is cut back first. With
    ``finalize``, a lease held elsewhere is waited for (a dead one is stolen once stale) instead of skipped."""
    import xarray as xr

    from tearing_physics_suite.drivers.multi_run import (
        _zarr_signature,
        align_codes,
        bare_episode,
        rechunk_store,
        store_ready,
    )
    from tearing_physics_suite.drivers.zarr_store import add_to_zarr_store, stack_episodes, truncate_zarr_store
    cfg = cfg or load_config(study)
    q = fq.Queue(qroot(study, code_check))
    lease = f'compile_{shot}'
    while not q.acquire('leases', lease, socket.gethostname()):
        if not finalize:
            print(f'compile: shot {shot} is being compiled elsewhere', flush=True)
            return None
        time.sleep(q.HEARTBEAT_S)
    zpath = shot_store(study, shot, code_check)
    spath = os.path.join(os.path.dirname(zpath), 'compile_state_codes.json' if code_check else 'compile_state.json')
    try:
        with fq.Heartbeat(q, lease, kind='leases'):
            state = fq.read_json(spath) or dict(compiled=[], broken=[], signature=None)
            if state['compiled'] and not os.path.exists(zpath):
                state = dict(compiled=[], broken=[], signature=None)
            elif os.path.exists(zpath):
                truncate_zarr_store(zpath, 'run_idx', len(state['compiled']))
            seen = set(state['compiled']) | set(state['broken'])
            draws = _shot_draws(study, cfg, shot, code_check)
            check = code_check_config(cfg)
            codes = study_codes(cfg['tps_kwargs'], check['codes'] if code_check else ())
            spot = None if code_check else spot_config(cfg)
            flag = not code_check and (check or {}).get('flag')
            new = sorted(k for k in draws if k not in seen)
            for i in range(0, len(new), batch):
                keys, dsets = [], []
                for key in new[i:i + batch]:
                    with xr.open_dataset(draws[key][0], engine='scipy') as ds:
                        ds = ds.load().drop_encoding()
                    ds = store_ready(_vlen_strings(align_codes(ds, codes)))
                    if spot:
                        ds['spot_check_flag'] = (('nn', 'code'), _spot_flag(*draws[key], ds.nn.values, codes))
                    if flag:
                        ds['stride_flag'] = ('nn', _stride_flag(draws[key][1], ds.nn.values))
                    if code_check:
                        ds['code_check_reason'] = np.array(draws[key][1].get('reason', 'random'), dtype=object)
                    ds = ds.assign(_timing_vars(draws[key][1], ds.nn.values, codes))
                    sig = [sorted(map(str, x)) for x in _zarr_signature(ds)]
                    if state['signature'] is None:
                        state['signature'] = sig
                    if sig != state['signature']:
                        state['broken'].append(key)
                    else:
                        keys.append(key)
                        dsets.append(bare_episode(ds.assign(case_key=np.array(key, dtype=object))))
                if dsets:
                    add_to_zarr_store(stack_episodes(dsets, 'run_idx'), zpath, episode_dim='run_idx')
                    state['compiled'] += keys
                fq.write_json(spath, state)
            if state['compiled']:
                _write_inputs(zpath, state['compiled'], draws)
            if finalize and new and os.path.exists(zpath):
                rechunk_store(zpath, 'run_idx', mb_per_chunk)
    finally:
        q.drop('leases', lease)
    print(f'compile: shot {shot}: {len(new)} new draw(s), {len(state["compiled"])} in the store, '
          f'{len(state["broken"])} broken', flush=True)
    return state


def study_codes(tps_kwargs, extra=()):
    """The codes a study runs, and extra ones (a code check's), in CODE_ORDER (run_resistive_calculation's defaults
    for unset flags)."""
    from tearing_physics_suite.physics.combine import CODE_ORDER
    k = tps_kwargs
    on = {'rdcon': k.get('run_rdcon', True), 'stride': k.get('run_stride', True), 'pest3': k.get('run_pest3', True)}
    if k.get('run_jgpec', False):
        on.update({f'jGPEC_{s}': True for s in k.get('jgpec_solvers', ('galerkin',))})
    return [c for c in CODE_ORDER if on.get(c) or c in extra]


def _vlen_strings(ds):
    """String variables and coords (scipy netCDF gives bytes, '<U' or objects) as object str arrays.

    zarr appends need one dtype across runs: variable-length strings, not fixed-width ones. Call it on a
    dataset without its netCDF encoding (which would write the strings back as chars).
    """
    def is_str(v):
        return v.dtype.kind in 'SU' or (v.dtype.kind == 'O' and any(isinstance(x, (bytes, str))
                                                                    for x in np.asarray(v.values).ravel()))

    def conv(v):
        vals = [x.decode() if isinstance(x, bytes) else x if isinstance(x, str) else ''
                for x in np.asarray(v.values).ravel()]
        return v.copy(data=np.array(vals, dtype=object).reshape(v.shape))
    return ds.assign_coords({k: conv(v) for k, v in ds.coords.items() if is_str(v)}).assign(
        {k: conv(v) for k, v in ds.data_vars.items() if is_str(v)})


def _write_inputs(zpath, keys, draws):
    """<shot>_inputs.nc: the scalar inputs of every compiled draw, along run_idx (store order)."""
    import xarray as xr
    rows = [draws[k][1].get('inputs', {}) if k in draws else {} for k in keys]
    names = sorted({n for r in rows for n in r})
    data = {}
    for n in names:
        vals = [r.get(n) for r in rows]
        kinds = {type(v) for v in vals if v is not None}
        if kinds <= {str}:
            data[n] = ('run_idx', np.array(['' if v is None else v for v in vals], dtype=object))
        elif kinds <= {bool, int, float}:
            data[n] = ('run_idx', np.array([np.nan if v is None else float(v) for v in vals]))
    ds = xr.Dataset(data, coords={'case_key': ('run_idx', np.array(keys, dtype=object))})
    out = zpath.removesuffix('.zarr') + '_inputs.nc'
    tmp = f'{out}.tmp.{os.getpid()}'
    ds.to_netcdf(tmp)
    os.replace(tmp, out)


def study_shots(study):
    return sorted(int(d) for d in os.listdir(study) if d.isdigit() and os.path.isdir(os.path.join(study, d)))


def cmd_compile(a):
    cfg = load_config(a.study)
    for shot in (a.shots or study_shots(a.study)):
        compile_shot(a.study, shot, cfg, finalize=a.finalize, code_check=a.code_check)


def open_study(study, shots=None, code_check=False):
    """Lazy dataset of every compiled shot store (code-check store), concatenated along run_idx.

    Like the stores themselves (add_to_zarr_store), ragged dims are NaN-padded at the end and aligned by
    position, not by coordinate value.
    """
    import xarray as xr
    paths = [p for p in (shot_store(study, s, code_check) for s in (shots or study_shots(study)))
             if os.path.exists(p) or os.path.exists(p + '.zip')]
    dss = [open_store(p) for p in paths]
    size = {}
    for ds in dss:
        for d, n in ds.sizes.items():
            size[d] = max(size.get(d, 0), n)
    dss = [ds.pad({d: (0, size[d] - n) for d, n in ds.sizes.items() if d != 'run_idx' and n < size[d]})
           for ds in dss]
    return xr.concat(dss, dim='run_idx', data_vars='minimal', coords='minimal', compat='override', join='override')


# --- status ------------------------------------------------------------------------------------

def draw_status(study, cfg=None, code_check=False):
    """Counts of draw (code-check) statuses per shot, and failure signatures with counts and an example."""
    cfg = cfg or load_config(study)
    per_shot, sigs = {}, {}
    for p in glob.glob(os.path.join(out_root(study, cfg), '*', 'tps', '*', 'draws', '*.json')):
        if p.endswith('.codes.json') != code_check:
            continue
        r = fq.read_json(p) or {}
        c = per_shot.setdefault(r.get('shot'), collections.Counter())
        c[r.get('status')] += 1
        if r.get('status') not in ('ok', 'running'):
            s = sigs.setdefault((r.get('error') or {}).get('signature', r.get('status')), [0, p])
            s[0] += 1
    return per_shot, sigs


def cmd_status(a):
    q = fq.Queue(qroot(a.study, a.code_check))
    s = q.status()
    print(f"items {s['total']}: done {s['done']}, failed {s['failed']}, running {s['running']}, "
          f"stale {s['stale']}, open {s['open']}" + (' [STOP]' if s['stopped'] else '')
          + (' [DRAIN]' if s['draining'] else ''))
    for sig, v in sorted(s['failures'].items(), key=lambda kv: -kv[1]['n']):
        print(f"  item failures {v['n']:4d} x {sig}  (e.g. {v['units'][0]})")
    per_shot, sigs = draw_status(a.study, code_check=a.code_check)
    tot = sum(per_shot.values(), collections.Counter())
    print('draws: ' + ', '.join(f'{k} {v}' for k, v in sorted(tot.items(), key=str)))
    for sig, (n, ex) in sorted(sigs.items(), key=lambda kv: -kv[1][0]):
        print(f'  draw failures {n:4d} x {sig}  (e.g. {ex})')
    if a.verbose:
        for shot, c in sorted(per_shot.items(), key=str):
            print(f'  shot {shot}: ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items(), key=str)))
        for w in sorted(s['workers'], key=lambda w: w['worker']):
            print(f"  {w['worker']}: {w.get('state')} {w.get('unit', '')} ok={w.get('n_ok', 0)}")
    if a.json:
        fq.write_json(a.json, dict(queue=s, draws={str(k): dict(v) for k, v in per_shot.items()},
                                   draw_failures={k: v[0] for k, v in sigs.items()}))


def cmd_flag(a):
    q = fq.Queue(qroot(a.study, a.code_check))
    if a.cmd == 'undrain':
        os.remove(q.path('DRAIN'))
    else:
        q.set_flag(a.cmd.upper(), a.reason)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def add(name, fn, code_check=True):
        p = sub.add_parser(name)
        p.add_argument('--study', required=True)
        if code_check:
            p.add_argument('--code-check', action='store_true', help='act on the code checks (config code_check)')
        p.set_defaults(fn=fn)
        return p

    p = add('init', cmd_init, code_check=False)
    p.add_argument('--config', default=None, help='JSON file of settings (see DEFAULTS)')
    p.add_argument('--force', action='store_true')
    add('feed', lambda a: feed_codes(a.study) if a.code_check else feed(a.study))
    p = add('work', cmd_work)
    p.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    p.add_argument('--item-timeout', type=float, default=6 * 3600)
    p.add_argument('--min-left', type=float, default=1800, help='s of walltime needed to start an item')
    p.add_argument('--margin', type=float, default=300, help='s before the job end to stop claiming')
    p.add_argument('--idle-exit', type=float, default=600)
    p.add_argument('--feed-every', type=float, default=300, help='s; 0: no feeding from workers')
    p.add_argument('--allow-build-drift', action='store_true')
    p.add_argument('--no-compile', action='store_true', help="don't compile a shot's store when its items finish")
    p = add('run-item', cmd_run_item)
    p.add_argument('--item', required=True)
    p.add_argument('--result', default=None, help='result JSON for the parent worker')
    p = add('compile', cmd_compile)
    p.add_argument('--shots', type=int, nargs='*')
    p.add_argument('--finalize', action='store_true', help='rechunk the stores')
    p = add('code-check-pending', cmd_code_check_pending, code_check=False)
    p.add_argument('--shots', type=int, nargs='*')
    p = add('status', cmd_status)
    p.add_argument('--json', default=None)
    p.add_argument('-v', '--verbose', action='store_true')
    for name in ('stop', 'drain', 'undrain'):
        add(name, cmd_flag).add_argument('--reason', default='user')
    a = ap.parse_args(argv)
    a.study = os.path.abspath(a.study)
    a.fn(a)


if __name__ == '__main__':
    main()
