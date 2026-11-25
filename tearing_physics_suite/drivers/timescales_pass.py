"""Timescales pass of a study: <shot>_timescales.zarr beside each shot's store, from the store's variables and the
geometric inverse aspect ratio eps_surf (physics/timescales.py; no code reruns), both read through
analysis.shot_data.open_shot, which joins the result back by case_key.

    python -m tearing_physics_suite.drivers.timescales_pass --study S [--shots ...] [--e-beam TABLE] [--force]
                                                            [--workers W]

TABLE: one JSON line per shot {shot, t_s, E_b_keV} (eq_stab claude_findings/fast_ion_times/scripts/e_beam_table.py:
the beam injection energy per FUSE slice). Each run takes the value at its t_ms, linear between slices; NaN outside
them or without a table. A store is written whole (a new one replaces the old) and skipped when present, unless
--force.
"""
import argparse
import functools
import json
import multiprocessing
import os
import shutil

import numpy as np
import xarray as xr

from tearing_physics_suite.analysis.shot_data import open_shot
from tearing_physics_suite.drivers import parallel
from tearing_physics_suite.physics.timescales import INPUTS, OPTIONAL_INPUTS, surface_timescales


def shot_store(study, shot):
    return os.path.join(study, str(shot), 'tps', f'{shot}_timescales.zarr')


def read_e_beam(path):
    """{shot: (t_s, E_b_keV)} of a beam-energy table."""
    out = {}
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:                     # a line cut by a kill
                continue
            out[int(row['shot'])] = tuple(np.array(row[k], dtype=float) for k in ('t_s', 'E_b_keV'))
    return out


def e_beam_at(table, shot, t_ms):
    """The beam energy [keV] at each t_ms of shot; NaN outside the table's slices or without the shot."""
    if shot not in table:
        return np.full(len(t_ms), np.nan)
    t, e = table[shot]
    return np.interp(np.asarray(t_ms, float) / 1e3, t, e, left=np.nan, right=np.nan)


def write_shot(study, shot, table=None, force=False):
    """Write <shot>_timescales.zarr; returns the number of runs, None if skipped (present, or no main store)."""
    path, main = shot_store(study, shot), parallel.shot_store(study, shot)
    if not (os.path.exists(main) or os.path.exists(main + '.zip')) or (os.path.exists(path) and not force):
        return None
    ds = open_shot(study, shot, variables=list(INPUTS + OPTIONAL_INPUTS) + ['case_key']).load()
    if 'eps_surf' not in ds:                        # no i-file eps in the main or ideal store: NaN nu* and mu
        ds = ds.assign(eps_surf=xr.full_like(ds['avg_R_surf'], np.nan))
    e_b = ds['t_ms'].copy(data=e_beam_at(table or {}, shot, ds['t_ms'].values))
    out = surface_timescales(ds, e_b).assign(case_key=ds['case_key'])
    tmp = f'{path}.tmp.{os.getpid()}'
    parallel._vlen_strings(out).to_zarr(tmp, mode='w', consolidated=True)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)
    return out.sizes['run_idx']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study', required=True)
    ap.add_argument('--shots', type=int, nargs='*')
    ap.add_argument('--e-beam', default=None, help='beam-energy table (JSON lines)')
    ap.add_argument('--force', action='store_true', help='rewrite stores that exist')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
                    help='shots written at once (~1.5 GB each for a 5700-run shot)')
    a = ap.parse_args(argv)
    table = read_e_beam(a.e_beam) if a.e_beam else None
    shots = a.shots or parallel.study_shots(a.study)
    with multiprocessing.Pool(a.workers) as pool:
        work = functools.partial(_write, a.study, table=table, force=a.force)
        for shot, n in pool.imap_unordered(work, shots):
            print(f'timescales: shot {shot}: ' + ('skipped' if n is None else f'{n} run(s)'), flush=True)


def _write(study, shot, table=None, force=False):
    return shot, write_shot(study, shot, table, force)


if __name__ == '__main__':
    main()
