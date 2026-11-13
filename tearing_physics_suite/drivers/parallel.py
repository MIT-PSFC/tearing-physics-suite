"""Study mode: TPS on bouquet archives in a shared study dir, on any number of nodes and jobs.

    python -m tearing_physics_suite.drivers.parallel init    --study S --config cfg.json
    python -m tearing_physics_suite.drivers.parallel feed    --study S
    python -m tearing_physics_suite.drivers.parallel work    --study S --workers W [--feed-every 300] [--feed-extra N]
    python -m tearing_physics_suite.drivers.parallel compile --study S [--shots ...] [--finalize]
    python -m tearing_physics_suite.drivers.parallel status  --study S
    python -m tearing_physics_suite.drivers.parallel join-n  --study S --more-n S2 [...] --out S12 [--shots ...]

Work items (``<unit_id>.c<k>``, a chunk of one bouquet unit's draws) are claimed from the fsqueue at
``<study>/_queue/tps`` and run in child processes. Each draw writes ``draws/<count>.nc`` and, last,
``draws/<count>.json`` (its status). See docs/parallel.md for the layout and the failure policy.

With ``--code-check``, feed/work/compile/status act on the code checks (config ``code_check``): the other codes on
the 'ok' draws of a seeded random fraction of shots or draws (STRIDE-flagged draws first; further draws when the
workers would idle), from ``<study>/_queue/tps_codes``, filled into ``draws/<count>.codes.nc`` (status
``.codes.json``) and compiled into ``<shot>_codes.zarr``. A draw that lost a STRIDE n (killed, or skipped as too big
for its session) first gets a fill from the same queue: STRIDE alone at those n, into the draw itself (status
``.fill.json``). A session given its memory (``--mem-gib``) takes only the items predicted to fit it.
``code-check-pending`` lists the draw files to keep for them.
"""
import argparse
import collections
import concurrent.futures
import functools
import glob
import hashlib
import itertools
import json
import os
import re
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
from tearing_physics_suite.utils import (
    CODE_KILLED,
    CODE_SKIPPED,
    CODE_WALL_S,
    fits_memory,
    gpec_home,
    session_mem_gib,
    work_root,
)

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
    eps=True,                   # with eq_source 'ifile': each draw stores its surfaces' inverse aspect ratio (eps on
                                # eps_psi_n; open_shot: eps_surf); off for studies initialised without the key
    spot_check=None,            # true or SpotCheck fields: re-run a seeded random fraction of the draws through the key
                                # input scans (drivers/spot_check.py); the store gets spot_check_flag(nn, code)
    code_check=None,            # {"by": "shot" | "draw", "fraction", "seed", "codes": [...], "nvec": [...], "flag":
                                # {"im_re_max", "psi_n": [lo, hi], optional "dp_abs_max", "dp_range": [lo, hi],
                                # "im_re_m_offset"}, "flagged_n_only", "mem_classes": [GiB, ...]}: run these codes at
                                # these n (default: the study's) on the 'ok' draws of a seeded random fraction of shots
                                # or draws, filled in (--code-check); with flag, STRIDE-flagged draws first (see
                                # stride_flagged_n; with flagged_n_only, at their flagged n, see _check_n); fills of
                                # lost STRIDE n before all; feeding per memory class (session sizes, see feed_codes)
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


def _records(study, cfg, kind='', shot='*'):
    """Status files of the draws (kind ''), their code checks ('.codes') or their fills ('.fill')."""
    name = re.compile(rf'-?\d+{re.escape(kind)}\.json')
    paths = glob.glob(os.path.join(out_root(study, cfg), str(shot), 'tps', '*', 'draws', f'*{kind}.json'))
    return sorted(p for p in paths if name.fullmatch(os.path.basename(p)))


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


def code_dirs():
    """The TPS package, the GPEC executables it runs (gpec_home) and jGPEC."""
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        gpec = gpec_home()
    except KeyError:
        gpec = os.path.join(pkg, 'submodules', 'GPEC')
    return dict(tps=pkg, gpec=gpec, jgpec=os.environ.get('JGPEC_HOME', '/fusion/projects/tmdb/src/GPEC'))


def versions():
    """git describes of code_dirs (None outside a checkout)."""
    return {k: _describe(p) for k, p in code_dirs().items()}


def builds_dir(study, cfg):
    return os.path.join(out_root(study, cfg), '_builds')


def record_build(study, cfg, now):
    """<out_root>/_builds/<job>.json: the code versions and dirs of this Slurm job (else 'local'), whose draws'
    provenance (run_provenance) the compile reads by their slurm_job."""
    os.makedirs(builds_dir(study, cfg), exist_ok=True)
    fq.write_json(os.path.join(builds_dir(study, cfg), f"{os.environ.get('SLURM_JOB_ID') or 'local'}.json"),
                  dict(versions=now, dirs=code_dirs(), host=socket.gethostname(),
                       time=time.strftime('%Y-%m-%dT%H:%M:%S')))


def load_builds(study, cfg):
    return {os.path.basename(p).removesuffix('.json'): fq.read_json(p) or {}
            for p in glob.glob(os.path.join(builds_dir(study, cfg), '*.json'))}


def run_provenance(rec, builds):
    """A draw's slurm_job and the {tps,gpec,jgpec}_version of its job's build (the describe, else the dir; '' without
    a build record)."""
    b = builds.get(rec.get('slurm_job') or 'local', {})
    return dict(slurm_job=rec.get('slurm_job', ''),
                **{f'{c}_version': b.get('versions', {}).get(c) or b.get('dirs', {}).get(c, '')
                 for c in ('tps', 'gpec', 'jgpec')})


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
    """Refuse to run on a build other than the one the study was initialised with; returns the build's versions."""
    now, then = versions(), cfg.get('versions', {})
    drift = {k: (then.get(k), v) for k, v in now.items() if v and then.get(k) and v != then.get(k)}
    if drift and not allow_drift:
        raise fq.Broken(f'build differs from the study config: {drift} (--allow-build-drift to run anyway)')
    return now


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
    """The study's code check {by, fraction, seed, codes, nvec, flag, flagged_n_only, mem_classes} (defaults filled
    in), or None; bad values raise."""
    from tearing_physics_suite.physics.combine import CODE_ORDER
    c = cfg.get('code_check')
    if not c:
        return None
    defaults = dict(by='draw', fraction=0.01, seed=0, codes=[], nvec=None, flag=None, flagged_n_only=False,
                    mem_classes=None)
    c = dict(defaults, **c)
    study_n = cfg['tps_kwargs'].get('nvec') or [1]
    flag_ok = c['flag'] is None or ({'im_re_max', 'psi_n'} <= set(c['flag']) <= set(FLAG_KEYS)
                                    and cfg['tps_kwargs'].get('run_stride', True))     # the flag reads STRIDE's Delta'
    classes = c['mem_classes']
    classes_ok = classes is None or (bool(classes) and classes == sorted(classes) and classes[0] > 0)
    if (set(c) - set(defaults) or c['by'] not in ('shot', 'draw') or not c['codes'] or not flag_ok or not classes_ok
            or set(c['codes']) - (set(CODE_ORDER) - {'stride'}) or not set(c['nvec'] or study_n) <= set(study_n)):
        raise ValueError(f"code_check: keys {sorted(defaults)}, by 'shot' or 'draw', codes from "
                         f"{[k for k in CODE_ORDER if k != 'stride']}, nvec from the study's {study_n}, flag "
                         f"{{im_re_max, psi_n: [lo, hi], optional {FLAG_KEYS[2:]}}} on a STRIDE study, mem_classes "
                         f"ascending GiB: {c}")
    return c


FLAG_KEYS = ('im_re_max', 'psi_n', 'dp_abs_max', 'dp_range', 'im_re_m_offset')


def stride_flagged_n(ds, flag):
    """The n at which a draw's STRIDE single-helicity Delta' is suspect, on a rational surface with psi_n[0] < psi_N <
    psi_n[1] (RDCON's surface map; a draw without STRIDE has no Delta'): no Re; |Im| > im_re_max |Re| (with
    im_re_m_offset only where m <= n + im_re_m_offset); and, when given, |Re| > dp_abs_max or Re outside dp_range."""
    sh = ds[['Delta_prime_surf', 'Im_Delta_prime_surf', 'psi_n_rational']].sel(Delta_prime_type='single helicity')
    sh = sh.reindex(code=['rdcon', 'stride'])
    re, im = sh.Delta_prime_surf.sel(code='stride'), sh.Im_Delta_prime_surf.sel(code='stride')
    psi = sh.psi_n_rational.sel(code='rdcon')
    im_bad = abs(im) > flag['im_re_max'] * abs(re)
    if flag.get('im_re_m_offset') is not None:
        im_bad = im_bad & (ds['m_rational'] <= ds['nn'] + flag['im_re_m_offset'])
    suspect = re.isnull() | im_bad
    if flag.get('dp_abs_max') is not None:
        suspect = suspect | (abs(re) > flag['dp_abs_max'])
    if flag.get('dp_range') is not None:
        suspect = suspect | (re < flag['dp_range'][0]) | (re > flag['dp_range'][1])
    bad = suspect & (psi > flag['psi_n'][0]) & (psi < flag['psi_n'][1])
    return [int(n) for n in ds.nn.values[bad.any('r').values]]


def _check_n(it):
    """The n a code-check item needs with flagged_n_only: its flagged n when it is checked for its flags alone (not
    for the seeded random sample); None (every n) otherwise."""
    reasons = set((it.get('reason') or 'random').split('+'))
    return it.get('flagged_n') if 'flagged' in reasons and 'random' not in reasons else None


def code_check_kwargs(cfg, check, only_n=None):
    """The study's tps_kwargs for a code check: only the check's codes, and RDCON (whose surfaces every code uses),
    at the check's n (every code there, Julia included), or only at only_n among them (with flagged_n_only: a flagged
    draw's flagged n; Julia then at those of its n)."""
    solvers = [c.removeprefix('jGPEC_') for c in check['codes'] if c.startswith('jGPEC_')]
    n = dict(nvec=check['nvec'], jgpec_nvec=check['nvec']) if check.get('nvec') else {}
    if only_n:
        check_n = check.get('nvec') or cfg['tps_kwargs'].get('nvec') or [1]
        jgpec_n = check.get('nvec') or cfg['tps_kwargs'].get('jgpec_nvec') or check_n
        n = dict(nvec=[k for k in check_n if k in only_n], jgpec_nvec=[k for k in jgpec_n if k in only_n])
        solvers = solvers if n['jgpec_nvec'] else []
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

    The baseline is its own item (<unit>.base), so it also reaches units fed before it was enabled. Draws added to a fed
    unit's archive later (a bouquet fill-up) get items <unit>.d<first count>. Every item has the same est, so workers
    claim in manifest order: a unit's items run together and its shot can be compiled and cleaned during the run.
    """
    cfg = cfg or load_config(study)
    n = int(cfg['draws_per_item'])
    q = fq.Queue(qroot(study))
    units = q.units(refresh=True)
    fed = {}
    for u in units.values():
        fed.setdefault(u.get('unit_id'), set()).update(u['draws'])
    items = []
    for rec_path in sorted(glob.glob(os.path.join(study, '*', 'bouquet', '*.json'))):
        r = fq.read_json(rec_path) or {}
        uid = r.get('unit_id')
        if r.get('status') != 'done' or r.get('variant') not in cfg['variants']:
            continue
        archive = os.path.join(os.path.dirname(rec_path), r['archive'])
        if cfg.get('baseline', True) and f'{uid}.base' not in units:
            items.append(dict(id=f'{uid}.base', unit_id=uid, shot=r['shot'], t_ms=r['t_ms'], variant=r['variant'],
                              archive=archive, draws=[-1], est=n))
        counts = r.get('selected_counts' if cfg['selection'] == 'selected' else 'all_counts')
        if counts is None:
            counts = _archive_counts(archive, cfg['selection'])
        new = [c for c in counts if c not in fed.get(uid, ())]
        for k in range(0, len(new), n):
            items.append(dict(id=f'{uid}.d{new[k]}' if uid in fed else f'{uid}.c{k // n}', unit_id=uid,
                              shot=r['shot'], t_ms=r['t_ms'], variant=r['variant'], archive=archive,
                              draws=new[k:k + n], est=n))
        if not counts and uid not in fed:
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


_SIZES = {}  # draw .nc -> {n: (RDCON's mpert, msing)}: the feeder's cache


def _need_gib(nc_path, inputs, codes, n=None):
    """Predicted peak (GiB, gpec_common.peak_gib) of running codes ('stride'; 'rdcon': its Delta') on a draw at n
    (None: all its n), from the draw's RDCON surface pass (mpert, msing) and inputs (nx, delta_mhigh)."""
    import xarray as xr

    from tearing_physics_suite.wrappers.gpec_common import peak_gib
    if not codes:
        return 0.0
    if nc_path not in _SIZES:
        with xr.open_dataset(nc_path, engine='scipy') as ds:
            rd = ds[['mpert', 'psi_n_rational']].sel(code='rdcon').load()
        _SIZES[nc_path] = {int(k): (int(rd.mpert.sel(nn=k)), int(rd.psi_n_rational.sel(nn=k).count()))
                           for k in rd.nn.values}
    dm, nx = inputs['delta_mhigh'], inputs['nx']
    sizes = [s for k, s in _SIZES[nc_path].items() if n is None or k in n]
    return max([peak_gib('stride', mp + dm, ms) for mp, ms in sizes if 'stride' in codes]
               + [peak_gib('rdcon', mp, ms, nx) for mp, ms in sizes if 'rdcon' in codes], default=0.0)


def _item_need_gib(study, cfg, it):
    """Predicted peak (GiB) of a code-check item: a fill's STRIDE at its fill_n, a check's RDCON Delta' (when a check
    code) at its n, with the draw's inputs as the study's tps_kwargs set them now (the item runs with those)."""
    check = code_check_config(cfg)
    path = os.path.join(unit_dir(study, cfg, it['shot'], it['unit_id']), 'draws', str(it['draws'][0]))
    if 'fill_n' in it:
        codes, n = ['stride'], it['fill_n']
    else:
        codes = ['rdcon'] if 'rdcon' in check['codes'] else []
        n = (check['flagged_n_only'] and _check_n(it)) or check['nvec']
    inputs = {**(fq.read_json(path + '.json') or {}).get('inputs', {}), **cfg['tps_kwargs']}
    return _need_gib(path + '.nc', inputs, codes, n)


def _run_need_gib(study, cfg, it):
    """The predicted peak a code-check item runs under: the need_gib it was classed and accepted by (feed stores it
    rounded), so a session never refuses an item it took; recomputed only for items fed without one."""
    return it.get('need_gib') or _item_need_gib(study, cfg, it)


def _mem_class(need_gib, classes):
    """The smallest memory class (GiB) need_gib fits (utils.fits_memory), else the largest; None without classes."""
    return next((c for c in classes if fits_memory(need_gib, c)), classes[-1]) if classes else None


def _dir_facts(d, cached):
    """[mtime_ns, unit, {count: [fill_n, flagged_n]}] of the 'ok' draws with their .nc in draws dir d (unit: the item
    keys of their records; fill_n: the lost STRIDE n of a draw still awaiting its fill); cached if d's mtime is the
    same (no status, .nc or fill written since); None without d."""
    try:
        m = os.stat(d).st_mtime_ns                                  # before its records are read
    except OSError:
        return None
    if cached and cached[0] == m:
        return cached
    names, unit, facts = set(os.listdir(d)), {}, {}
    for n in sorted(names):                                         # _records' order
        c = n[:-len('.json')]
        if not re.fullmatch(r'-?\d+\.json', n) or c + '.nc' not in names:                           # cleaned
            continue
        r = fq.read_json(os.path.join(d, n)) or {}
        if r.get('status') != 'ok':
            continue
        unit = {k: r[k] for k in ('unit_id', 'shot', 't_ms', 'variant', 'archive')}
        fill = ('code_killed' in r and 'code_filled' not in r
                and not _draw_state(os.path.join(d, c + '.fill.json'))[0])
        facts[c] = [sorted({k for ns in r['code_killed'].values() for k in ns}) if fill else None,
                    r.get('stride_flagged_n')]
    return [m, unit, facts]


def _feed_facts(study, cfg, q):
    """{draws dir: _dir_facts} of the study's units, kept in the code-check queue (feed_facts.json) so a feed reads only
    the dirs changed since the last; 16 dirs at a time (NFS metadata calls are slow)."""
    path = q.path('feed_facts.json')
    old = fq.read_json(path) or {}
    dirs = sorted(os.path.join(u, 'draws') for u in glob.glob(os.path.join(out_root(study, cfg), '*', 'tps', '*')))
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        out = {d: f for d, f in zip(dirs, ex.map(lambda d: _dir_facts(d, old.get(d)), dirs)) if f is not None}
    if out != old:
        fq.write_json(path, out)
    return out


def _prefetch_need(study, cfg, q, items, need):
    """need(it) of items, 16 at a time; each draw's .nc is read once: its RDCON sizes (_SIZES) are kept in the
    code-check queue (feed_sizes.json, the items' draws only), saved every 4096 so a feed that is killed keeps them."""
    path = q.path('feed_sizes.json')
    for p, s in (fq.read_json(path) or {}).items():
        _SIZES.setdefault(p, {int(k): tuple(v) for k, v in s.items()})
    ncs = [os.path.join(unit_dir(study, cfg, it['shot'], it['unit_id']), 'draws', f"{it['draws'][0]}.nc")
           for it in items]
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        for i in range(0, len(items), 4096):
            n = len(_SIZES)
            list(ex.map(need, items[i:i + 4096]))
            if len(_SIZES) > n:
                fq.write_json(path, {p: _SIZES[p] for p in ncs if p in _SIZES})


def feed_codes(study, cfg=None, extra=0, mem_gib=None):
    """Add code-check items for 'ok' draws not yet fed, each with its predicted peak (need_gib, _item_need_gib), by
    priority within each memory class (code_check.mem_classes; the smallest that fits need_gib):
    - a draw that lost a STRIDE n (code_killed, not yet code_filled) first gets a fill (<unit_id>.f<count>, reason
      'killed', fill_n: STRIDE alone at those n, see _fill_case); its check waits for it;
    - once none of those waits for a worker, the STRIDE-flagged draws' checks (<unit_id>.d<count>, 'flagged');
    - then the picked shots' or draws' ('random', 'flagged+random');
    - once nothing a session of mem_gib could take waits, ``extra`` further draws (that fit it) in the coin's order
      ('extra': the work feeder passes its worker count and memory, so the check sessions keep going).
    Only the draws dirs changed since the last feed are read (_feed_facts), and each draw's .nc once (_prefetch_need).
    Returns how many were added."""
    cfg = cfg or load_config(study)
    check = code_check_config(cfg)
    if check is None:
        raise SystemExit('feed --code-check: the study has no code_check')
    q = fq.Queue(qroot(study, code_check=True))
    units, done = q.units(refresh=True), q.done_ids()
    open_ = [it for uid, it in units.items() if uid not in done and q.age('claims', uid) >= q.STALE_S]

    def tier(it):
        return (it.get('reason') or 'random').split('+')[0]

    def need(it):                                    # predicted only with classes or a session memory
        if 'need_gib' not in it:
            it['need_gib'] = round(_item_need_gib(study, cfg, it), 1)
        return it['need_gib']
    classes = check['mem_classes']
    waiting = {(tier(it), _mem_class(it.get('need_gib', 0.0), classes)) for it in open_}
    new, extras = collections.defaultdict(list), []
    for _, unit, facts in _feed_facts(study, cfg, q).values():
        for c, (fill_n, flagged_n) in facts.items():
            item = dict(unit, draws=[int(c)], est=1)
            if fill_n is not None:
                uid = f"{unit['unit_id']}.f{c}"
                if uid not in units:
                    item.update(id=uid, reason='killed', fill_n=fill_n)
                    new['killed'].append((0.0, uid, item))
                continue
            uid = f"{unit['unit_id']}.d{c}"
            if uid in units:
                continue
            u, reason = _pick(check, dict(shot=unit['shot'], case_key=f"{unit['unit_id']}_{c}",
                                          stride_flagged_n=flagged_n))
            item.update(id=uid, reason=reason)
            if flagged_n:
                item['flagged_n'] = flagged_n
            (extras if reason == 'extra' else new[reason.split('+')[0]]).append((u, uid, item))
    if classes:
        _prefetch_need(study, cfg, q, [it for t in new.values() for _, _, it in t], need)
    items = []
    for cls in classes or [None]:
        for t in ('killed', 'flagged', 'random'):
            add = [it for _, _, it in new[t] if not classes or _mem_class(need(it), classes) == cls]
            if add or (t, cls) in waiting:
                items += add
                break
    if extra and not any(fits_memory(it.get('need_gib', 0.0), mem_gib) for it in open_ + items):
        items += itertools.islice((it for _, _, it in sorted(extras) if not mem_gib or fits_memory(need(it), mem_gib)),
                                  extra)
    n_new = q.add_units(items)
    print(f'feed: {n_new} code-check item(s) added', flush=True)
    return n_new


class Feeder:
    """on_idle hook: feed, unless any worker fed in the last ``every`` seconds (code checks: up to ``workers`` extra
    draws that fit the session's mem_gib; work --feed-extra, else its --workers)."""

    def __init__(self, study, every, code_check=False, workers=0, mem_gib=None):
        self.study, self.every, self.code_check, self.workers, self.mem_gib = study, every, code_check, workers, mem_gib

    def __call__(self):
        fn = functools.partial(feed_codes, extra=self.workers, mem_gib=self.mem_gib) if self.code_check else feed
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
    """(final, attempts) of a draw's status file; a code check's or fill's timeout is final (not retried)."""
    r = fq.read_json(path)
    if r is None:
        return False, 0
    n = r.get('attempts', 1)
    timed_out_check = r.get('status') == 'timeout' and path.endswith(('.codes.json', '.fill.json'))
    return r.get('status') in FINAL or n >= MAX_DRAW_ATTEMPTS or timed_out_check, n


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
    """Run the draws of one work item (a code-check item: their code checks, or fills) that are not final yet.
    Returns a summary dict."""
    from tearing_physics_suite.drivers.multi_run import run_case
    cfg = load_config(study)
    it = fq.Queue(qroot(study, code_check)).units()[item_id]
    sfx = ('.fill' if 'fill_n' in it else '.codes') if code_check else ''
    ddir, states, todo = draws_to_run(study, cfg, it, sfx)
    out = dict(item=item_id, draws=[])
    if not todo:
        return out
    check = code_check_config(cfg)
    flag = (check or {}).get('flag')
    need = _run_need_gib(study, cfg, it) if code_check and session_mem_gib() else 0.0
    if not code_check:
        kwargs = dict(cfg['tps_kwargs'])
        case = functools.partial(run_case, eps=True) if cfg.get('eps') and cfg['eq_source'] == 'ifile' else run_case
    elif 'fill_n' in it:
        kwargs = dict(cfg['tps_kwargs'], nvec=it['fill_n'])
        case = functools.partial(_fill_case, need_gib=need, flag=flag)
    else:
        kwargs = code_check_kwargs(cfg, check, _check_n(it) if check['flagged_n_only'] else None)
        case = functools.partial(_check_case, need_gib=need)
    spot = None if code_check else spot_config(cfg)
    if cfg.get('tau_e'):
        kwargs['energy_confinement_time'] = tau_e_lookup(cfg['tau_e'])(it['shot'], it['t_ms'] / 1e3)
    out['draws'] = run_draws(cfg, it, ddir, states, todo, kwargs, case, spot, sfx, None if code_check else flag)
    if not code_check and any(d['status'] in ('timeout',) for d in out['draws']):
        raise RuntimeError('draw timeout(s), to retry: ' + ', '.join(str(d['draw']) for d in out['draws']
                                                                   if d['status'] == 'timeout'))
    return out


def draws_to_run(study, cfg, it, sfx):
    """(draws dir, {count: (final, attempts)}, counts not final) of a work item's draws, by their status records
    <count><sfx>.json (code timeout set from the config); a draw left 'running' when final is recorded 'crashed'."""
    if cfg.get('code_timeout'):
        os.environ['TPS_CODE_TIMEOUT'] = str(cfg['code_timeout'])
    ddir = os.path.join(unit_dir(study, cfg, it['shot'], it['unit_id']), 'draws')
    os.makedirs(ddir, exist_ok=True)
    states = {c: _draw_state(os.path.join(ddir, f'{c}{sfx}.json')) for c in it['draws']}
    for c, (final, _) in states.items():                 # a draw left 'running' crashed its process twice
        r = fq.read_json(os.path.join(ddir, f'{c}{sfx}.json')) or {}
        if final and r.get('status') == 'running':
            fq.write_json(os.path.join(ddir, f'{c}{sfx}.json'), dict(r, status='crashed'))
    return ddir, states, [c for c, (final, _) in states.items() if not final]


def run_draws(cfg, it, ddir, states, todo, kwargs, case, spot=None, sfx='', flag=None, ifile_dir=None):
    """Read the todo draws of item it from its bouquet archive (dropped i-files from ifile_dir, see
    read_bouquet_archive) and run each through case (run_case's signature, see _run_draw) in a scratch dir; returns
    their summaries."""
    from tearing_physics_suite.drivers.profile_read import read_bouquet_archive
    wd = tempfile.mkdtemp(prefix=f"tps_{it['id']}_", dir=work_root())
    try:
        eqs, profs = read_bouquet_archive(it['archive'], selection='all', draw_counts=todo,
                                          eqdsk_out_dir=os.path.join(wd, 'eq'), eq_source=cfg['eq_source'],
                                          ida_extra_keys=cfg['ida_extra_keys'], meta_data=cfg['meta_data'],
                                          main_ion_mass_amu=cfg['main_ion_mass_amu'],
                                          impurity_mass_amu=cfg['impurity_mass_amu'], ifile_dir=ifile_dir)
        return [_run_draw(it, cfg, kwargs, ddir, eq, prof, states[prof['bq_count']][1] + 1, os.path.join(wd, 'work'),
                          case, spot, sfx, flag) for eq, prof in zip(eqs, profs)]
    finally:
        shutil.rmtree(wd, ignore_errors=True)


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
    from tearing_physics_suite.drivers.profile_read import IFILE_REBUILT
    if eq in IFILE_REBUILT:                                             # its i-file rebuilt (read_bouquet_archive)
        rec.update(ifile_rebuild_s=IFILE_REBUILT[eq]['s'], ifile_rebuild_oft=IFILE_REBUILT[eq]['oft'])
    fq.write_json(stat, dict(rec, status='running'))
    os.makedirs(wd, exist_ok=True)
    CODE_WALL_S.clear()
    CODE_KILLED.clear()
    CODE_SKIPPED.clear()
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
    if CODE_KILLED:
        rec['code_killed'] = dict(CODE_KILLED)
    if CODE_SKIPPED:
        rec['code_skipped'] = dict(CODE_SKIPPED)
    rec['wall_s'] = round(time.time() - rec['started'], 1)
    rec['other_wall_s'] = round(rec['wall_s'] - sum(rec['code_wall_s'].values()) - rec.get('spot_wall_s', 0.0), 1)
    fq.write_json(stat, rec)
    return dict(draw=c, unit_id=it['unit_id'], status=rec['status'], signature=(rec['error'] or {}).get('signature'))


def _check_memory(need_gib):
    """Refuse to run a code-check item predicted to peak above the session's memory."""
    if not fits_memory(need_gib):
        raise RuntimeError(f'predicted peak {need_gib:.0f} GiB does not fit the session ({session_mem_gib():g} GiB)')


def _check_case(eq, prof, kwargs, wd, nc_path, pkl_path=None, label=None, labels=None, need_gib=0.0):
    """run_case for a code check: the check's codes on the draw's inputs, filled into its dataset (<count>.nc), the
    result written to nc_path (<count>.codes.nc)."""
    import xarray as xr

    from tearing_physics_suite.drivers import multi_run
    from tearing_physics_suite.drivers.pipeline import fill_codes
    _check_memory(need_gib)
    other, inputs = multi_run.run_case(eq, prof, kwargs, wd, os.path.join(wd, 'codes.nc'), pkl_path, label, labels)
    if other is None:
        return None, inputs
    with xr.open_dataset(nc_path.removesuffix('.codes.nc') + '.nc', engine='scipy') as ds:
        ds = fill_codes(ds.load().drop_encoding(), other)        # the file's int dtypes would not hold NaN padding
    _write_nc(ds, nc_path)
    return ds, inputs


def _fill_case(eq, prof, kwargs, wd, nc_path, pkl_path=None, label=None, labels=None, need_gib=0.0, flag=None):
    """run_case for a fill: the study's codes at the n a draw lost (kwargs' nvec), its lost codes (code_killed) then
    filled into the draw itself at those n (<count>.nc; in its record code_filled, and stride_flagged_n redone with
    flag); a code lost again fails it. Nothing is written at nc_path (<count>.fill.nc)."""
    import xarray as xr

    from tearing_physics_suite.drivers import multi_run
    from tearing_physics_suite.drivers.pipeline import fill_codes
    _check_memory(need_gib)
    other, inputs = multi_run.run_case(eq, prof, kwargs, wd, os.path.join(wd, 'fill.nc'), pkl_path, label, labels)
    if other is None:
        return None, inputs
    base = nc_path.removesuffix('.fill.nc')
    rec = fq.read_json(base + '.json')
    killed = sorted(rec['code_killed'])
    if CODE_KILLED or not set(killed) <= set(other.code.values):         # lost again here, or no result
        raise RuntimeError(f'lost code(s) not re-run: {dict(CODE_KILLED) or killed}')
    with xr.open_dataset(base + '.nc', engine='scipy') as ds:
        ds = ds.load().drop_encoding()                          # the file's int dtypes would not hold NaN padding
    own = list(ds.variables)                                    # the draw keeps its own variables
    ds = fill_codes(ds, other.sel(code=killed))[own]
    rec['code_filled'] = killed
    if flag:
        rec['stride_flagged_n'] = stride_flagged_n(ds, flag)
    _write_nc(ds, base + '.nc')
    fq.write_json(base + '.json', rec)
    return ds, inputs


def _write_nc(ds, path):
    """ds to path (netCDF, scipy engine) by a temporary file and a rename."""
    tmp = f'{path}.tmp.{os.getpid()}'
    ds.to_netcdf(tmp, engine='scipy')
    os.replace(tmp, path)


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
    """Picklable ``make_run`` for fsqueue.launch: each item runs as a run-item child process (of module, compiled by
    compile_fn(study, shot, cfg); default: this driver's, per --code-check)."""

    def __init__(self, a, module='tearing_physics_suite.drivers.parallel', compile_fn=None):
        self.study, self.timeout, self.min_left = a.study, a.item_timeout, a.min_left
        self.allow_drift, self.deadline = a.allow_build_drift, fq.JobDeadline(0.0)   # live: --time-min limits rise
        self.compile = not getattr(a, 'no_compile', False)
        self.code_check = getattr(a, 'code_check', False)
        self.module = module
        self.compile_fn = compile_fn or functools.partial(compile_shot, code_check=self.code_check)

    def __call__(self, q, worker):
        cfg = load_config(self.study)
        recent = collections.deque(maxlen=cfg['n_same_sig'])

        checked = []

        def run(uid, item):
            if not checked:
                record_build(self.study, cfg, check_build(cfg, self.allow_drift))
                checked.append(True)
            dl = self.deadline()
            if dl and dl - time.time() < self.min_left:
                raise fq.Requeue('too little walltime left')
            log = os.path.join(unit_dir(self.study, cfg, item['shot'], item['unit_id']), 'logs', f'{uid}.log')
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, 'a') as fh:
                fh.write(f'\n=== {time.ctime()} {worker} attempt {q.n_attempts(uid)}\n')
            res_path = os.path.join(q.root, 'results', f'{uid}.{worker}.{q.n_attempts(uid)}.json')
            argv = [sys.executable, '-m', self.module, 'run-item',
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
                    self.compile_fn(self.study, item['shot'], cfg)
                except Exception as e:
                    print(f'[{worker}] compile of shot {item["shot"]} failed: {fq.error_signature(e)}', flush=True)
            return dict(n_draws=len(res.get('draws', [])), **n)
        return run


def _shot_finished(q, shot, current):
    """True if every queued item of the shot except ``current`` is terminal."""
    done = q.done_ids()
    return all(uid in done for uid, it in q.units().items() if it.get('shot') == shot and uid != current)


def _fits_session(study, mem_gib, it):
    """fsqueue accept for a code-check session of mem_gib: the item's memory class is at most mem_gib (one that fits
    no class goes to the largest, whose sessions refuse to run it, _check_memory); without classes, its predicted peak
    fits mem_gib."""
    cfg = load_config(study)
    need, classes = it.get('need_gib'), code_check_config(cfg)['mem_classes']
    need = _item_need_gib(study, cfg, it) if need is None else need
    return _mem_class(need, classes) <= mem_gib if classes else fits_memory(need, mem_gib)


def cmd_work(a):
    d = fq.JobDeadline(a.margin)              # re-read: Slurm raises a --time-min job's limit after it starts
    print(f'work: {a.workers} worker(s) on {socket.gethostname()}, deadline now {time.ctime(d()) if d() else "none"}',
          flush=True)
    if a.mem_gib:                             # the codes' memory check (utils.fits_memory), in the workers' children
        os.environ['TPS_MEM_GIB'] = str(a.mem_gib)
    accept = functools.partial(_fits_session, a.study, a.mem_gib) if a.code_check and a.mem_gib else None
    out = fq.launch(qroot(a.study, a.code_check), a.workers, Runner(a), deadline=d, idle_exit_s=a.idle_exit,
                    accept=accept,
                    on_idle=Feeder(a.study, a.feed_every, a.code_check, a.feed_extra or a.workers, a.mem_gib)
                    if a.feed_every else None)
    for w, why in sorted(out.items()):
        print(f'  {w}: {why}', flush=True)
    if not a.no_compile:          # catch draws that finished while their shot was being compiled
        fq.Queue(qroot(a.study, a.code_check)).periodic('compile_all', 600, lambda: [
            compile_shot(a.study, s, code_check=a.code_check) for s in study_shots(a.study)])


# --- compile -----------------------------------------------------------------------------------

def _shot_draws(study, cfg, shot, code_check=False):
    """{case_key: (nc path, status record)} of the shot's draws (code checks) with status 'ok'."""
    out = {}
    for p in _records(study, cfg, '.codes' if code_check else '', shot):
        r = fq.read_json(p) or {}
        if r.get('status') == 'ok':
            out[r['case_key']] = (p[:-len('.json')] + '.nc', r)
    return out


def code_check_pending(study, shot, cfg=None):
    """The <count>.nc of the shot's 'ok' draws whose fill or code check is still to come (a lost STRIDE n not yet
    filled; a check queued and not finished, or not yet queued and flagged or picked), which must stay until it is
    done."""
    cfg = cfg or load_config(study)
    check = code_check_config(cfg)
    if check is None:
        return []
    q = fq.Queue(qroot(study, code_check=True))
    units, done = q.units(refresh=True), q.done_ids()
    out = []
    for nc, r in _shot_draws(study, cfg, shot).values():
        uid = f"{r['unit_id']}.d{r['draw']}"
        if (_awaits_fill(nc, r) or (uid in units and uid not in done)
                or (uid not in units and _pick(check, r)[1] != 'extra')):
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


def _awaits_fill(nc_path, rec):
    """True while a draw's lost codes (code_killed) wait for its fill (code_filled, or the fill final)."""
    return ('code_killed' in rec and 'code_filled' not in rec
            and not _draw_state(nc_path[:-len('.nc')] + '.fill.json')[0])


# Variables only some draws carry, NaN in the others: STRIDE's zero-crossing count (calc_nzero 't', ledger C-12)
OPTIONAL_VARS = {'nzero': ('nn', 'code')}


def _fit_optional(dsets, zpath):
    """The draws of a batch with every OPTIONAL_VARS variable that one of them or the store has (NaN where a draw lacks
    it); one the store lacks goes into it first, NaN for the runs already there."""
    import xarray as xr
    old = xr.open_zarr(zpath, consolidated=True) if os.path.exists(zpath) else None
    have = set() if old is None else set(old.data_vars)
    want = {v: ('run_idx',) + d for v, d in OPTIONAL_VARS.items() if v in have or any(v in ds for ds in dsets)}
    nan = lambda ds, d: (d, np.full([ds.sizes[k] for k in d], np.nan, np.float32))   # noqa: E731
    if old is not None and set(want) - have:
        xr.Dataset({v: nan(old, d) for v, d in want.items() if v not in have}).to_zarr(zpath, mode='a',
                                                                                         consolidated=True)
    return [ds.assign({v: ds[v].transpose(*d) if v in ds else nan(ds, d) for v, d in want.items()}) for ds in dsets]


def _fingerprint(study, cfg, shot, jobs):
    """The mtimes of the shot's draws dirs and of its jobs' build records: a draw's status, .nc or fill written, or one
    of those build records, changes one."""
    paths = sorted(glob.glob(os.path.join(out_root(study, cfg), str(shot), 'tps', '*', 'draws')))
    paths += [os.path.join(builds_dir(study, cfg), f'{j}.json') for j in jobs]
    stamps = [(p, os.stat(p).st_mtime_ns) for p in paths if os.path.exists(p)]
    return hashlib.sha1(repr(stamps).encode()).hexdigest()


def _still_holds(q, lease, owner):
    """Raise before a store write if the lease was taken over (its owner stalled past STALE_S, e.g. on NFS)."""
    if not q.owns('leases', lease, owner):
        raise RuntimeError(f'{lease}: lease taken over by another compile; stopped before writing')


def compile_shot(study, shot, cfg=None, finalize=False, mb_per_chunk=10, batch=32, code_check=False):
    """Append the shot's new 'ok' draws to <study>/<shot>/tps/<shot>.zarr (code checks: to <shot>_codes.zarr) (single
    writer, under a lease checked before each write).

    Draws go in ``batch`` at a time, one store write each: a write costs a few file operations per variable,
    whatever the number of draws. A store left longer than compile_state.json by a killed compile is cut back first,
    and a rechunk killed in its swap completed (finish_rechunk). A shot leased elsewhere is skipped (None), and one
    unchanged since its last compile (_fingerprint) is not read again.
    With ``finalize``, new draws are rechunked. With a code check, a draw with a killed code waits until its check has
    filled it or is final (see _check_case)."""
    import xarray as xr

    from tearing_physics_suite.drivers.multi_run import (
        _zarr_signature,
        align_codes,
        bare_episode,
        finish_rechunk,
        rechunk_store,
        store_ready,
    )
    from tearing_physics_suite.drivers.zarr_store import add_to_zarr_store, stack_episodes, truncate_zarr_store
    from tearing_physics_suite.physics.global_quantities import delta_prime_variability
    cfg = cfg or load_config(study)
    q = fq.Queue(qroot(study, code_check))
    lease, owner = f'compile_{shot}', f'{socket.gethostname()}:{os.getpid()}'
    if not q.acquire('leases', lease, owner):
        print(f'compile: shot {shot} is being compiled elsewhere', flush=True)
        return None
    zpath = shot_store(study, shot, code_check)
    spath = os.path.join(os.path.dirname(zpath), 'compile_state_codes.json' if code_check else 'compile_state.json')
    fpath = spath.replace('compile_state', 'compile_fingerprint')    # apart: a pass that adds nothing leaves the state
    try:
        with fq.Heartbeat(q, lease, kind='leases'):
            finish_rechunk(zpath)
            state = fq.read_json(spath) or dict(compiled=[], broken=[], signature=None)
            last = fq.read_json(fpath) or {}
            fingerprint = _fingerprint(study, cfg, shot, last.get('jobs', []))      # before any record is read
            if last.get('fingerprint') == fingerprint and os.path.exists(zpath):
                return state
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
            held = [] if code_check or not check else [k for k in draws if k not in seen and _awaits_fill(*draws[k])]
            new = sorted(k for k in draws if k not in seen and k not in held)
            for i in range(0, len(new), batch):
                keys, dsets = [], []
                for key in new[i:i + batch]:
                    with xr.open_dataset(draws[key][0], engine='scipy') as ds:
                        ds = ds.load().drop_encoding()
                    ds = delta_prime_variability(align_codes(ds, codes))   # the same comparisons in every run
                    ds = store_ready(_vlen_strings(ds))
                    if spot:
                        ds['spot_check_flag'] = (('nn', 'code'), _spot_flag(*draws[key], ds.nn.values, codes))
                    if flag:
                        ds['stride_flag'] = ('nn', _stride_flag(draws[key][1], ds.nn.values))
                    if code_check:
                        ds['code_check_reason'] = np.array(draws[key][1].get('reason', 'random'), dtype=object)
                    ds = ds.assign(_timing_vars(draws[key][1], ds.nn.values, codes))
                    bare = ds.drop_vars(OPTIONAL_VARS, errors='ignore')
                    sig = [sorted(map(str, x)) for x in _zarr_signature(bare)]
                    if state['signature'] is None:
                        state['signature'] = sig
                    if sig != state['signature']:
                        state['broken'].append(key)
                    else:
                        keys.append(key)
                        dsets.append(bare_episode(ds.assign(case_key=np.array(key, dtype=object))))
                if dsets:
                    _still_holds(q, lease, owner)
                    dsets = _fit_optional(dsets, zpath)
                    add_to_zarr_store(stack_episodes(dsets, 'run_idx'), zpath, episode_dim='run_idx')
                    state['compiled'] += keys
                _still_holds(q, lease, owner)
                fq.write_json(spath, state)
            if finalize and new and os.path.exists(zpath):
                _still_holds(q, lease, owner)
                rechunk_store(zpath, 'run_idx', mb_per_chunk)
            if state['compiled']:
                _still_holds(q, lease, owner)
                _write_inputs(zpath, state['compiled'], draws, load_builds(study, cfg))
                jobs = sorted({draws[k][1].get('slurm_job') or 'local' for k in state['compiled'] if k in draws})
                fq.write_json(fpath, dict(jobs=jobs, fingerprint=fingerprint))
    finally:
        if q.owns('leases', lease, owner):
            q.drop('leases', lease)
    print(f'compile: shot {shot}: {len(new)} new draw(s), {len(state["compiled"])} in the store, '
          f'{len(state["broken"])} broken, {len(held)} awaiting a code-check fill', flush=True)
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
    return ds.assign_coords({k: conv(v) for k, v in ds.coords.variables.items() if is_str(v)}).assign(
        {k: conv(v) for k, v in ds.data_vars.variables.items() if is_str(v)})


def _write_inputs(zpath, keys, draws, builds):
    """<shot>_inputs.nc: the scalar inputs and run_provenance of every compiled draw, along run_idx (store order)."""
    import xarray as xr
    rows = [dict(draws[k][1].get('inputs', {}), **run_provenance(draws[k][1], builds)) if k in draws else {}
            for k in keys]
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
    """Every shot, those leased elsewhere (another worker's) last: with --finalize they are retried until compiled
    (a dead lease is stolen once stale)."""
    cfg = load_config(a.study)
    left = list(a.shots or study_shots(a.study))
    while True:
        left = [s for s in left if compile_shot(a.study, s, cfg, finalize=a.finalize, code_check=a.code_check) is None]
        if not (a.finalize and left):
            return
        time.sleep(fq.Queue(qroot(a.study, a.code_check)).HEARTBEAT_S)


def cmd_join_n(a):
    """Each shot stored in the study and in every --more-n study, written with all their n to --out
    (shot_data.write_shot) unless already there, so a rerun resumes; the shots missing from any study are listed."""
    from tearing_physics_suite.analysis.shot_data import write_shot
    shots = a.shots or study_shots(a.study)
    stored = [shot for shot in shots if all(os.path.exists(p) or os.path.exists(p + '.zip')
                                            for p in (shot_store(s, shot) for s in [a.study, *a.more_n]))]
    for shot in stored:
        if not os.path.exists(shot_store(a.out, shot)):
            write_shot(a.study, shot, a.out, a.more_n)
    if len(stored) < len(shots):
        print(f'join-n: not in every study, skipped: {sorted(set(shots) - set(stored))}')


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
    """Counts of draw (code-check and 'fill ...') statuses per shot (and of records with code_killed, code_skipped,
    code_filled, ifile_rebuilt, with the rebuilds' summed ifile_rebuild_s), and failure signatures with counts and an
    example."""
    cfg = cfg or load_config(study)
    per_shot, sigs = {}, {}
    for kind in ('.codes', '.fill') if code_check else ('',):
        for p in _records(study, cfg, kind):
            r = fq.read_json(p) or {}
            c = per_shot.setdefault(r.get('shot'), collections.Counter())
            c[f"{'fill ' if kind == '.fill' else ''}{r.get('status')}"] += 1
            c.update(k for k in ('code_killed', 'code_skipped', 'code_filled') if k in r)
            if 'ifile_rebuild_s' in r:                                  # summed: status gives the mean
                c.update(ifile_rebuilt=1, ifile_rebuild_s=r['ifile_rebuild_s'])
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
    rebuild_s = tot.pop('ifile_rebuild_s', 0.0)
    print('draws: ' + ', '.join(f'{k} {v}' for k, v in sorted(tot.items(), key=str)))
    if tot['ifile_rebuilt']:
        print(f"  i-files rebuilt {tot['ifile_rebuilt']}: mean {rebuild_s / tot['ifile_rebuilt']:.2f} s")
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
    p.add_argument('--feed-extra', type=int, help="code checks: extra draws a feed adds (default --workers); one feed "
                   "reads every draw's status, so with many 1-worker sessions a feed should serve them all")
    p.add_argument('--allow-build-drift', action='store_true')
    p.add_argument('--no-compile', action='store_true', help="don't compile a shot's store when its items finish")
    p.add_argument('--mem-gib', type=float, default=float(os.environ.get('TPS_MEM_GIB') or 0) or None,
                   help="the session's memory (Slurm --mem G): codes predicted not to fit are not started; with "
                        "--code-check, only the items that fit are taken (default $TPS_MEM_GIB)")
    p = add('run-item', cmd_run_item)
    p.add_argument('--item', required=True)
    p.add_argument('--result', default=None, help='result JSON for the parent worker')
    p = add('compile', cmd_compile)
    p.add_argument('--shots', type=int, nargs='*')
    p.add_argument('--finalize', action='store_true', help='rechunk the stores')
    p = add('join-n', cmd_join_n, code_check=False)
    p.add_argument('--more-n', nargs='+', required=True, help='studies of the same draws at further n')
    p.add_argument('--out', required=True, help='study dir for the joined shot stores')
    p.add_argument('--shots', type=int, nargs='*')
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
