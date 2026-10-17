"""Study mode: TPS on bouquet archives in a shared study dir, on any number of nodes and jobs.

    python -m tearing_physics_suite.drivers.parallel init    --study S --config cfg.json
    python -m tearing_physics_suite.drivers.parallel feed    --study S
    python -m tearing_physics_suite.drivers.parallel work    --study S --workers W [--feed-every 300]
    python -m tearing_physics_suite.drivers.parallel compile --study S [--shots ...] [--finalize]
    python -m tearing_physics_suite.drivers.parallel status  --study S

Work items (``<unit_id>.c<k>``, a chunk of one bouquet unit's draws) are claimed from the fsqueue at
``<study>/_queue/tps`` and run in child processes. Each draw writes ``draws/<count>.nc`` and, last,
``draws/<count>.json`` (its status). See docs/parallel.md for the layout and the failure policy.
"""
import argparse
import collections
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
from tearing_physics_suite.utils import tps_home, work_root

DEFAULTS = dict(
    tps_kwargs={},              # nonlinear_resistive_calculation kwargs (nvec, run_stride, run_jgpec, vac_flag, ...)
    variants=['base'],          # bouquet variants to run
    selection='selected',       # 'selected' | 'all' draws of each archive
    eq_source='geqdsk',         # 'geqdsk' | 'ifile'
    draws_per_item=8,
    ida_extra_keys=[],
    meta_data=False,            # bq_* per-draw metadata in the profile dicts
    keep_pickle=False,          # also keep multi_run-style result pickles
    code_timeout=3600,          # s per Fortran code run (TPS_CODE_TIMEOUT)
    out_root=None,              # per-draw outputs under <out_root>/<shot>/tps (default: the study dir); $VARS expanded
    n_same_sig=6,               # this many draw failures in a row with one signature, over >= 2 units: worker broken
    tau_e=None,                 # {"path": I2 netCDF(s), ':'-separated globs, $VARS expanded, "variable": name}: the
                                # energy confinement time per draw (see tau_e_lookup)
    baseline=True,              # also run each archive's bouquet baseline, as draw BASELINE_DRAW (item <unit>.base)
    spot_check=None,            # true or SpotCheck fields: re-run a seeded random fraction of the draws through the key
                                # input scans (drivers/spot_check.py); the store gets spot_check_flag(nn, code)
)
BASELINE_DRAW = -1
FINAL = ('ok', 'error', 'crashed')
MAX_DRAW_ATTEMPTS = 2           # a timed-out or crashed draw is retried once
INFRA = (ImportError, MemoryError)
INFRA_ERRNO = {28, 30, 122}     # ENOSPC, EROFS, EDQUOT


# --- paths and config -------------------------------------------------------------------------

def qroot(study):
    return os.path.join(study, '_queue', 'tps')


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


def shot_store(study, shot):
    return os.path.join(study, str(shot), 'tps', f'{shot}.zarr')


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
    unknown = set(spot.scans or ()) - set(sc.SCANS)
    if unknown:
        raise ValueError(f'spot_check: unknown scans {sorted(unknown)}')
    return spot


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

    The baseline is its own item (<unit>.base), so it is added to units fed before it was enabled.
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
                              archive=archive, draws=[BASELINE_DRAW], est=1))
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


class Feeder:
    """on_idle hook: feed, unless any worker fed in the last ``every`` seconds."""

    def __init__(self, study, every):
        self.study, self.every = study, every

    def __call__(self):
        try:
            return bool(fq.Queue(qroot(self.study)).periodic('feeder', self.every, lambda: feed(self.study) > 0))
        except Exception as e:
            print(f'feeder failed: {fq.error_signature(e)}', flush=True)
            return False


# --- one item (child process) ------------------------------------------------------------------

def _spot_picked(spot, case_key):
    """A seeded coin per draw, so the same draws are checked however they are chunked or retried."""
    return np.random.default_rng([spot.seed, zlib.crc32(case_key.encode())]).random() < spot.fraction


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


def run_item(study, item_id):
    """Run the draws of one work item that are not final yet. Returns a summary dict."""
    from tearing_physics_suite.drivers.multi_run import run_case
    from tearing_physics_suite.drivers.profile_read import read_bouquet_archive
    cfg = load_config(study)
    it = fq.Queue(qroot(study)).units()[item_id]
    if cfg.get('code_timeout'):
        os.environ['TPS_CODE_TIMEOUT'] = str(cfg['code_timeout'])
    ddir = os.path.join(unit_dir(study, cfg, it['shot'], it['unit_id']), 'draws')
    os.makedirs(ddir, exist_ok=True)
    states = {c: _draw_state(os.path.join(ddir, f'{c}.json')) for c in it['draws']}
    for c, (final, _) in states.items():                 # a draw left 'running' crashed its process twice
        r = fq.read_json(os.path.join(ddir, f'{c}.json')) or {}
        if final and r.get('status') == 'running':
            fq.write_json(os.path.join(ddir, f'{c}.json'), dict(r, status='crashed'))
    todo = [c for c, (final, _) in states.items() if not final]
    out = dict(item=item_id, draws=[])
    if not todo:
        return out
    kwargs = dict(cfg['tps_kwargs'])
    spot = spot_config(cfg)
    if cfg.get('tau_e'):
        kwargs['energy_confinement_time'] = tau_e_lookup(cfg['tau_e'])(it['shot'], it['t_ms'] / 1e3)
    wd = tempfile.mkdtemp(prefix=f'tps_{item_id}_', dir=work_root())
    try:
        eqs, profs = read_bouquet_archive(it['archive'], selection='all', draw_counts=todo,
                                          eqdsk_out_dir=os.path.join(wd, 'eq'), eq_source=cfg['eq_source'],
                                          ida_extra_keys=cfg['ida_extra_keys'], meta_data=cfg['meta_data'])
        for eq, prof in zip(eqs, profs):
            out['draws'].append(_run_draw(it, cfg, kwargs, ddir, eq, prof, states[prof['bq_count']][1] + 1,
                                          os.path.join(wd, 'work'), run_case, spot))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    if any(d['status'] in ('timeout',) for d in out['draws']):
        raise RuntimeError('draw timeout(s), to retry: ' + ', '.join(str(d['draw']) for d in out['draws']
                                                                   if d['status'] == 'timeout'))
    return out


def _run_draw(it, cfg, kwargs, ddir, eq, prof, attempt, wd, run_case, spot=None):
    c = prof['bq_count']
    key = f"{it['unit_id']}_{c}"
    stat = os.path.join(ddir, f'{c}.json')
    rec = dict(case_key=key, shot=it['shot'], t_ms=it['t_ms'], variant=it['variant'], unit_id=it['unit_id'],
               draw=c, archive=it['archive'], profile_coord=prof.get('bq_profile_coord'), attempts=attempt,
               config_hash=cfg['hash'], host=socket.gethostname(), slurm_job=os.environ.get('SLURM_JOB_ID', ''),
               started=time.time())
    fq.write_json(stat, dict(rec, status='running'))
    os.makedirs(wd, exist_ok=True)
    try:
        if cfg.get('tau_e') and kwargs.get('energy_confinement_time') is None:
            raise LookupError(f"no {cfg['tau_e']['variable']} for shot {it['shot']} at {it['t_ms']} ms")
        ds, inputs = run_case(eq, prof, kwargs, wd, nc_path=os.path.join(ddir, f'{c}.nc'),
                              pkl_path=os.path.join(ddir, f'{c}.pkl') if cfg['keep_pickle'] else None, label=key,
                              labels=dict(shot_id=it['shot'], t_ms=it['t_ms'], draw=c, variant=it['variant']))
        rec.update(status='ok' if ds is not None else 'error', inputs=_scalars(inputs),
                   error=None if ds is not None else dict(signature='no combined_xr'))
        if ds is not None and spot and _spot_picked(spot, key):     # never fails the draw
            rec['spot_errors'] = _run_spot_checks(spot, eq, kwargs, os.path.join(ddir, f'{c}.spot'), wd)
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
    rec['wall_s'] = round(time.time() - rec['started'], 1)
    fq.write_json(stat, rec)
    return dict(draw=c, unit_id=it['unit_id'], status=rec['status'], signature=(rec['error'] or {}).get('signature'))


def _err(e):
    return dict(type=type(e).__name__, signature=fq.error_signature(e), message=str(e)[:2000],
                traceback_tail=traceback.format_exc()[-4000:])


def cmd_run_item(a):
    res = a.result or os.path.join(qroot(a.study), 'results', f'{a.item}.json')
    os.makedirs(os.path.dirname(res), exist_ok=True)
    try:
        fq.write_json(res, run_item(a.study, a.item))
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
                    compile_shot(self.study, item['shot'], cfg)
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
    out = fq.launch(qroot(a.study), a.workers, Runner(a), deadline=d, idle_exit_s=a.idle_exit,
                    on_idle=Feeder(a.study, a.feed_every) if a.feed_every else None)
    for w, why in sorted(out.items()):
        print(f'  {w}: {why}', flush=True)
    if not a.no_compile:          # catch draws that finished while their shot was being compiled
        fq.Queue(qroot(a.study)).periodic('compile_all', 600, lambda: [compile_shot(a.study, s)
                                                                       for s in study_shots(a.study)])


# --- compile -----------------------------------------------------------------------------------

def _shot_draws(study, cfg, shot):
    """{case_key: (nc path, status record)} of the shot's draws with status 'ok'."""
    out = {}
    root = os.path.join(out_root(study, cfg), str(shot), 'tps')
    for p in glob.glob(os.path.join(root, '*', 'draws', '*.json')):
        r = fq.read_json(p) or {}
        if r.get('status') == 'ok':
            out[r['case_key']] = (p[:-len('.json')] + '.nc', r)
    return out


def _spot_flag(nc_path, rec, nn, codes):
    """spot_check_flag(nn, code) of one draw (-1 where it was not checked)."""
    from tearing_physics_suite.drivers import spot_check as sc
    sdir = nc_path[:-len('.nc')] + '.spot'
    ds = sc.spot_dataset(sc.load_spot_results(sdir)) if 'spot_errors' in rec else None
    return sc.spot_check_flag(ds, 0, nn, codes)


def compile_shot(study, shot, cfg=None, finalize=False, mb_per_chunk=10, batch=32):
    """Append the shot's new 'ok' draws to <study>/<shot>/tps/<shot>.zarr (single writer, under a lease).

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
    q = fq.Queue(qroot(study))
    lease = f'compile_{shot}'
    while not q.acquire('leases', lease, socket.gethostname()):
        if not finalize:
            print(f'compile: shot {shot} is being compiled elsewhere', flush=True)
            return None
        time.sleep(q.HEARTBEAT_S)
    zpath = shot_store(study, shot)
    spath = os.path.join(os.path.dirname(zpath), 'compile_state.json')
    try:
        with fq.Heartbeat(q, lease, kind='leases'):
            state = fq.read_json(spath) or dict(compiled=[], broken=[], signature=None)
            if state['compiled'] and not os.path.exists(zpath):
                state = dict(compiled=[], broken=[], signature=None)
            elif os.path.exists(zpath):
                truncate_zarr_store(zpath, 'run_idx', len(state['compiled']))
            seen = set(state['compiled']) | set(state['broken'])
            draws = _shot_draws(study, cfg, shot)
            codes = study_codes(cfg['tps_kwargs'])
            spot = spot_config(cfg)
            new = sorted(k for k in draws if k not in seen)
            for i in range(0, len(new), batch):
                keys, dsets = [], []
                for key in new[i:i + batch]:
                    with xr.open_dataset(draws[key][0], engine='scipy') as ds:
                        ds = ds.load().drop_encoding()
                    ds = store_ready(_vlen_strings(align_codes(ds, codes)))
                    if spot:
                        ds['spot_check_flag'] = (('nn', 'code'), _spot_flag(*draws[key], ds.nn.values, codes))
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


def study_codes(tps_kwargs):
    """The codes a study runs, in CODE_ORDER (run_resistive_calculation's defaults for unset flags)."""
    from tearing_physics_suite.physics.combine import CODE_ORDER
    k = tps_kwargs
    on = {'rdcon': k.get('run_rdcon', True), 'stride': k.get('run_stride', True), 'pest3': k.get('run_pest3', True)}
    if k.get('run_jgpec', False):
        on.update({f'jGPEC_{s}': True for s in k.get('jgpec_solvers', ('galerkin',))})
    return [c for c in CODE_ORDER if on.get(c)]


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
        compile_shot(a.study, shot, cfg, finalize=a.finalize)


def open_study(study, shots=None):
    """Lazy dataset of every compiled shot store, concatenated along run_idx.

    Like the stores themselves (add_to_zarr_store), ragged dims are NaN-padded at the end and aligned by
    position, not by coordinate value.
    """
    import xarray as xr
    paths = [p for p in (shot_store(study, s) for s in (shots or study_shots(study)))
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

def draw_status(study, cfg=None):
    """Counts of draw statuses per shot, and draw-failure signatures with counts and an example."""
    cfg = cfg or load_config(study)
    per_shot, sigs = {}, {}
    for p in glob.glob(os.path.join(out_root(study, cfg), '*', 'tps', '*', 'draws', '*.json')):
        r = fq.read_json(p) or {}
        c = per_shot.setdefault(r.get('shot'), collections.Counter())
        c[r.get('status')] += 1
        if r.get('status') not in ('ok', 'running'):
            s = sigs.setdefault((r.get('error') or {}).get('signature', r.get('status')), [0, p])
            s[0] += 1
    return per_shot, sigs


def cmd_status(a):
    q = fq.Queue(qroot(a.study))
    s = q.status()
    print(f"items {s['total']}: done {s['done']}, failed {s['failed']}, running {s['running']}, "
          f"stale {s['stale']}, open {s['open']}" + (' [STOP]' if s['stopped'] else '')
          + (' [DRAIN]' if s['draining'] else ''))
    for sig, v in sorted(s['failures'].items(), key=lambda kv: -kv[1]['n']):
        print(f"  item failures {v['n']:4d} x {sig}  (e.g. {v['units'][0]})")
    per_shot, sigs = draw_status(a.study)
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
    q = fq.Queue(qroot(a.study))
    if a.cmd == 'undrain':
        os.remove(q.path('DRAIN'))
    else:
        q.set_flag(a.cmd.upper(), a.reason)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def add(name, fn):
        p = sub.add_parser(name)
        p.add_argument('--study', required=True)
        p.set_defaults(fn=fn)
        return p

    p = add('init', cmd_init)
    p.add_argument('--config', default=None, help='JSON file of settings (see DEFAULTS)')
    p.add_argument('--force', action='store_true')
    add('feed', lambda a: feed(a.study))
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
