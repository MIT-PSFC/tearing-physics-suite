"""Ideal-stability pass of a study: per 'ok' draw of the main pass, RDCON's ideal ODE (fixed-boundary zero crossings,
nzero) at each n in turn, stopping at the first n found unstable. RDCON's count decides unless it is pathological
(nzero >= max_nzero, or a crossing on a rational surface: |n q - m| below q_tol, or below n times the resolution of
the q crit.out prints, 4 significant digits); STRIDE's count decides then. STRIDE also
runs on a seeded random fraction (sample) of the runs RDCON alone finds unstable, to measure the codes' agreement.

    python -m tearing_physics_suite.drivers.ideal_pass init    --study S [--config ideal.json]
    python -m tearing_physics_suite.drivers.ideal_pass feed    --study S [--shots ...]
    python -m tearing_physics_suite.drivers.ideal_pass work    --study S --workers W
    python -m tearing_physics_suite.drivers.ideal_pass compile --study S [--shots ...]
    python -m tearing_physics_suite.drivers.ideal_pass status  --study S

Items (<unit_id>.i<first draw>) come from <study>/_queue/tps_ideal (settings in its ideal.json). Each draw writes
draws/<count>.ideal.nc (status .ideal.json), compiled into <shot>_ideal.zarr: nzero(nn, code) (NaN where a code did
not run), ideal_unstable(nn) (1, 0; -1 undecided or not run), ideal_rational_crossing(nn), ideal_stride_reason(nn)
('pathological', 'sample' or '') and RDCON's crossings (ideal_crossing_psi, ideal_crossing_q).
"""
import argparse
import collections
import functools
import hashlib
import inspect
import json
import os
import re
import shutil
import socket

import numpy as np

from tearing_physics_suite.drivers import fsqueue as fq
from tearing_physics_suite.drivers import parallel

DEFAULTS = dict(nvec=None, max_nzero=3, q_tol=1e-3, sample=0.01, seed=0,   # nvec None: the study's
                ifile_dir=None)     # i-files of draws whose archive dropped them (read_bouquet_archive)
IDEAL_RDCON = dict(ode_flag='t', vac_flag='f', gal_flag='f', MRE_flag='f',
                   rdcon_delta_prime=True)                 # (False would switch the ODE off too)
IDEAL_STRIDE = dict(calc_nzero='t')                       # STRIDE's zero-crossing count
CROSSING = re.compile(r'Zero crossing at psi =\s*(\S+), q =\s*(\S+)')
CODES = ['rdcon', 'stride']


def qroot(study):
    return os.path.join(study, '_queue', 'tps_ideal')


def load_config(study):
    cfg = fq.read_json(os.path.join(qroot(study), 'ideal.json'))
    if cfg is None:
        raise FileNotFoundError(f'{qroot(study)}/ideal.json: run `ideal_pass init` first')
    return cfg


def ideal_config(study, cfg):
    """The pass's settings (DEFAULTS filled in, nvec the study's when None); bad values raise."""
    c = dict(DEFAULTS, **cfg)
    study_n = parallel.load_config(study)['tps_kwargs'].get('nvec') or [1]
    c['nvec'] = c['nvec'] or study_n
    if (set(c) - set(DEFAULTS) or not set(c['nvec']) <= set(study_n) or int(c['max_nzero']) < 1
            or not 0 <= c['sample'] <= 1 or c['q_tol'] <= 0):
        raise ValueError(f"ideal pass: keys {sorted(DEFAULTS)}, nvec from the study's {study_n}, max_nzero >= 1, "
                         f"0 <= sample <= 1, q_tol > 0: {c}")
    return c


def gpec_kwargs(tps_kwargs):
    """The study's tps_kwargs that GPEC's input files take (write_rdcon_stride_inputs, write_equil_in)."""
    from tearing_physics_suite.wrappers.gpec_inputs import SHARED_KEYS, write_equil_in, write_rdcon_stride_inputs
    keys = {*inspect.signature(write_rdcon_stride_inputs).parameters, *inspect.signature(write_equil_in).parameters}
    keys -= {'working_dir', 'eq_filename', 'run_rdcon', 'run_stride', 'nn', 'kwargs'}
    return {k: v for k, v in tps_kwargs.items() if k in keys or k.removesuffix('_stride') in SHARED_KEYS}


# --- one draw ----------------------------------------------------------------------------------

def _crossings(path):
    """RDCON's zero crossings [(psi, q)] from its crit.out."""
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [(float(m[1]), float(m[2])) for m in CROSSING.finditer(fh.read())]


def _half_ulp(q):
    """Half the last printed digit of q in crit.out (Fortran e10.3: 4 significant digits)."""
    return 5e-4 * 10.0 ** np.floor(np.log10(abs(q))) if q else 0.0


def _run_code(code, eq, n, kwargs, wd):
    """nzero of one RDCON (ideal ODE) or STRIDE run at n, and RDCON's crossings; (NaN, []) when it failed."""
    from tearing_physics_suite.wrappers.gpec import GPEC_resistive_calculation
    d = os.path.join(wd, f'{code}_n{n}')
    kw = {**kwargs, **(IDEAL_RDCON if code == 'rdcon' else IDEAL_STRIDE)}
    rd, sd, *_ = GPEC_resistive_calculation(eq, n, run_rdcon=code == 'rdcon', run_stride=code == 'stride',
                                            working_dir=d, **kw)
    ds = rd if code == 'rdcon' else sd
    if ds is None:
        return np.nan, []
    if 'nzero' not in ds.attrs:
        raise RuntimeError(f'{code} wrote no nzero (a STRIDE build without calc_nzero?)')
    return float(ds.attrs['nzero']), _crossings(os.path.join(d, 'crit.out')) if code == 'rdcon' else []


def _ideal_case(eq, prof, kwargs, wd, nc_path, pkl_path=None, label=None, labels=None, icfg=None):
    """run_case for the ideal pass (see the module docstring): the draw's verdicts per n, written to nc_path."""
    import xarray as xr
    nvec, rows = icfg['nvec'], []
    for n in nvec:
        z, cross = _run_code('rdcon', eq, n, kwargs, wd)
        rational = any(abs(n * q - round(n * q)) < max(icfg['q_tol'], n * _half_ulp(q)) for _, q in cross)
        path = z >= icfg['max_nzero'] or rational
        sample = z > 0 and not path and parallel._coin(icfg['seed'], f'{label}_n{n}') < icfg['sample']
        reason = 'pathological' if path else 'sample' if sample else ''
        s = _run_code('stride', eq, n, kwargs, wd)[0] if reason else np.nan
        verdict = s if path else z
        rows.append(dict(z=z, s=s, cross=cross, rational=-1 if np.isnan(z) else int(rational), reason=reason,
                         unstable=-1 if np.isnan(verdict) else int(verdict > 0)))
        if rows[-1]['unstable'] == 1:
            break
    rows += [dict(z=np.nan, s=np.nan, cross=[], rational=-1, reason='', unstable=-1)] * (len(nvec) - len(rows))
    m = max([1] + [len(r['cross']) for r in rows])
    cross = np.full((len(nvec), m, 2), np.nan)
    for i, r in enumerate(rows):
        cross[i, :len(r['cross'])] = np.reshape(r['cross'], (-1, 2))
    ds = xr.Dataset(dict(nzero=(('nn', 'code'), np.array([[r['z'], r['s']] for r in rows], np.float32)),
                         ideal_unstable=('nn', np.array([r['unstable'] for r in rows], np.int8)),
                         ideal_rational_crossing=('nn', np.array([r['rational'] for r in rows], np.int8)),
                         ideal_stride_reason=('nn', np.array([r['reason'] for r in rows], object)),
                         ideal_crossing_psi=(('nn', 'crossing'), cross[..., 0]),
                         ideal_crossing_q=(('nn', 'crossing'), cross[..., 1]),
                         **{k: np.array(v, object if isinstance(v, str) else None) for k, v in (labels or {}).items()}),
                    coords=dict(nn=nvec, code=CODES))
    parallel._write_nc(ds, nc_path)
    return ds, {}


def run_item(study, item_id):
    """Run the ideal pass on one work item's draws that are not final yet. Returns a summary dict."""
    cfg, icfg = parallel.load_config(study), load_config(study)
    it = fq.Queue(qroot(study)).units()[item_id]
    ddir, states, todo = parallel.draws_to_run(study, cfg, it, '.ideal')
    out = dict(item=item_id, draws=[])
    if todo:
        case = functools.partial(_ideal_case, icfg=icfg)
        out['draws'] = parallel.run_draws(dict(cfg, tau_e=None), it, ddir, states, todo,    # (no MRE: no tau_E)
                                          gpec_kwargs(cfg['tps_kwargs']), case, sfx='.ideal',
                                          ifile_dir=icfg['ifile_dir'])
    return out


# --- feed, compile, pending --------------------------------------------------------------------

def feed(study, shots=None):
    """Add items for the main pass's 'ok' draws (of shots) not yet fed, draws_per_item per item. Returns how many."""
    cfg = parallel.load_config(study)
    q = fq.Queue(qroot(study))
    fed = {(it['unit_id'], c) for it in q.units(refresh=True).values() for c in it['draws']}
    new = collections.defaultdict(list)
    for shot in shots or ['*']:
        for p in parallel._records(study, cfg, '', shot):
            r = fq.read_json(p) or {}
            if r.get('status') == 'ok' and (r['unit_id'], r['draw']) not in fed:
                new[r['unit_id']].append(r)
    items, n = [], int(cfg['draws_per_item'])
    for uid, recs in sorted(new.items()):
        recs.sort(key=lambda r: r['draw'])
        for k in range(0, len(recs), n):
            chunk = recs[k:k + n]
            unit = {x: chunk[0][x] for x in ('unit_id', 'shot', 't_ms', 'variant', 'archive')}
            items.append(dict(id=f"{uid}.i{chunk[0]['draw']}", **unit, draws=[r['draw'] for r in chunk],
                              est=len(chunk)))
    added = q.add_units(items)
    print(f'feed: {added} ideal-pass item(s) added', flush=True)
    return added


def ideal_pending(study, shot):
    """case_keys of the shot's 'ok' draws whose ideal pass is not final (none before the pass is set up), whose
    bouquet i-files must stay until it is."""
    if not os.path.exists(os.path.join(qroot(study), 'ideal.json')):
        return []
    cfg = parallel.load_config(study)
    out = []
    for p in parallel._records(study, cfg, '', shot):
        r = fq.read_json(p) or {}
        if r.get('status') == 'ok' and not parallel._draw_state(p.removesuffix('.json') + '.ideal.json')[0]:
            out.append(r['case_key'])
    return sorted(out)


def shot_store(study, shot):
    return os.path.join(study, str(shot), 'tps', f'{shot}_ideal.zarr')


def compile_shot(study, shot, cfg=None):
    """Write <shot>_ideal.zarr from the shot's 'ok' ideal-pass draws, whole (a new store replaces the old one), under
    a lease. Returns the number of draws, None if compiled elsewhere."""
    import xarray as xr

    from tearing_physics_suite.drivers.multi_run import bare_episode
    from tearing_physics_suite.drivers.zarr_store import stack_episodes
    cfg = cfg or parallel.load_config(study)
    q, lease = fq.Queue(qroot(study)), f'compile_{shot}'
    if not q.acquire('leases', lease, socket.gethostname()):
        print(f'compile: shot {shot} is being compiled elsewhere', flush=True)
        return None
    try:
        with fq.Heartbeat(q, lease, kind='leases'):
            dsets = []
            for p in parallel._records(study, cfg, '.ideal', shot):
                r = fq.read_json(p) or {}
                if r.get('status') == 'ok':
                    with xr.open_dataset(p.removesuffix('.json') + '.nc', engine='scipy') as ds:
                        ds = parallel._vlen_strings(ds.load().drop_encoding())
                    dsets.append(bare_episode(ds.assign(case_key=np.array(r['case_key'], dtype=object))))
            if dsets:
                path = shot_store(study, shot)
                tmp = f'{path}.tmp.{os.getpid()}'
                stack_episodes(dsets, 'run_idx').to_zarr(tmp, mode='w', consolidated=True)
                shutil.rmtree(path, ignore_errors=True)
                os.replace(tmp, path)
    finally:
        q.drop('leases', lease)
    print(f'compile: shot {shot}: {len(dsets)} draw(s) in {shot}_ideal.zarr', flush=True)
    return len(dsets)


# --- CLI ---------------------------------------------------------------------------------------

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


def cmd_init(a):
    cfg = {}
    if a.config:
        with open(a.config) as fh:
            cfg = json.load(fh)
    cfg = ideal_config(a.study, cfg)
    cfg['hash'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    path, old = os.path.join(qroot(a.study), 'ideal.json'), fq.read_json(os.path.join(qroot(a.study), 'ideal.json'))
    if old and old['hash'] != cfg['hash'] and not a.force:
        raise SystemExit(f'{path} exists with other settings (hash {old["hash"]}); --force to replace')
    fq.Queue(qroot(a.study))
    fq.write_json(path, cfg)
    print(f'init: {path} (hash {cfg["hash"]})')


def cmd_work(a):
    d = fq.JobDeadline(a.margin)
    runner = parallel.Runner(a, module='tearing_physics_suite.drivers.ideal_pass', compile_fn=compile_shot)
    out = fq.launch(qroot(a.study), a.workers, runner, deadline=d, idle_exit_s=a.idle_exit,
                    on_idle=Feeder(a.study, a.feed_every) if a.feed_every else None)
    for w, why in sorted(out.items()):
        print(f'  {w}: {why}', flush=True)


def cmd_run_item(a):
    res = a.result or os.path.join(qroot(a.study), 'results', f'{a.item}.json')
    os.makedirs(os.path.dirname(res), exist_ok=True)
    try:
        fq.write_json(res, run_item(a.study, a.item))
    except BaseException as e:
        infra = isinstance(e, parallel.INFRA) or (isinstance(e, OSError) and e.errno in parallel.INFRA_ERRNO)
        fq.write_json(res, dict(error=dict(type=type(e).__name__, signature=fq.error_signature(e), infra=infra)))
        raise


def cmd_status(a):
    s = fq.Queue(qroot(a.study)).status()
    print(f"items {s['total']}: done {s['done']}, failed {s['failed']}, running {s['running']}, open {s['open']}"
          + (' [STOP]' if s['stopped'] else '') + (' [DRAIN]' if s['draining'] else ''))
    cfg = parallel.load_config(a.study)
    c = collections.Counter((fq.read_json(p) or {}).get('status') for p in parallel._records(a.study, cfg, '.ideal'))
    print('draws: ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items(), key=str)))


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
    add('feed', lambda a: feed(a.study, a.shots)).add_argument('--shots', type=int, nargs='*')
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
    p = add('compile', lambda a: [compile_shot(a.study, s) for s in a.shots or parallel.study_shots(a.study)])
    p.add_argument('--shots', type=int, nargs='*')
    add('status', cmd_status)
    for name in ('stop', 'drain', 'undrain'):
        add(name, cmd_flag).add_argument('--reason', default='user')
    a = ap.parse_args(argv)
    a.study = os.path.abspath(a.study)
    a.fn(a)


if __name__ == '__main__':
    main()
