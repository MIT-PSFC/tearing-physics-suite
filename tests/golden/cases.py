"""Golden-reference cases for the Group 1 refactor (A0).

Runs the same TPS cases on any layout (pre- or post-A1) and saves the outputs.
Usage: python tests/golden/cases.py OUT_DIR [--cases name ...]
TPSHOME and PYTHONPATH must point at the TPS checkout under test.
"""
import argparse
import os
import pickle as pkl
import shutil
import sys
import tempfile
import time
import traceback

import xarray as xr


def _imports():
    """Return the TPS callables, from the new layout if present, else the old one."""
    try:
        from tearing_physics_suite.drivers.multi_run import multi_compile_zarr, multi_run_
        from tearing_physics_suite.drivers.pipeline import linear_resistive_calculation, nonlinear_resistive_calculation
        from tearing_physics_suite.drivers.profile_read import read_kin_file
        from tearing_physics_suite.physics.global_quantities import global_mre_quantities
    except ImportError:
        from tearing_physics_suite.multi_run import multi_compile_zarr, multi_run_
        from tearing_physics_suite.profile_read import read_kin_file
        from tearing_physics_suite.tearing_physics_suite import (
            global_mre_quantities,
            linear_resistive_calculation,
            nonlinear_resistive_calculation,
        )
    return dict(linear=linear_resistive_calculation, nonlinear=nonlinear_resistive_calculation,
                gmq=global_mre_quantities, read_kin=read_kin_file,
                multi_run_=multi_run_, multi_compile_zarr=multi_compile_zarr)


EQ = os.path.join(os.environ['TPSHOME'], 'tests/data/g147131.02300_DIIID_KEFIT')
REDUCED = dict(etol=1e-7, nx=64, mpsi=128, mtheta=129)
MRE = dict(Zeff=1.5, average_ion_mass=2.5, energy_confinement_time=0.12)


def _profiles(f):
    p = f['read_kin'](EQ + '.kin')
    return p, (p['ni_spline'], p['ne_spline'], p['te_keV_spline'], p['ti_keV_spline'])


def case_nonlinear_wall_rotation(f, wd):
    p, spl = _profiles(f)
    return f['nonlinear'](EQ, *spl, omega_splines=p['omega_splines'], Er_spline=None,
                          q_surfs_of_interest=[1.5, 2.0], psi_surfs_of_interest=[0.3, 0.95],
                          nvec=[1, 2], run_stride=True, run_pest3=True, vac_flag='f', ode_flag='f',
                          wd_static=True, pest_match_truncation=False, working_dir=wd,
                          **MRE, **REDUCED)


def case_nonlinear_nowall(f, wd):
    _, spl = _profiles(f)
    return f['nonlinear'](EQ, *spl, nvec=[1, 2], run_stride=True, run_pest3=True,
                          vac_flag='t', gal_flag='t', ode_flag='t', working_dir=wd,
                          **MRE, **REDUCED)


def case_linear_wall(f, wd):
    return f['linear'](EQ, nvec=[1, 2], run_stride=True, run_pest3=True, vac_flag='f',
                       ode_flag='f', working_dir=wd, **REDUCED)


def case_multi_run_zarr(f, wd):
    # Worker dirs on node-local disk: on NFS, _clean_working_dir hits EBUSY on .nfs* files
    # (claude_findings/omega_nfs_clean_working_dir.md). Results are copied back to wd.
    profs = [f['read_kin'](EQ + '.kin') for _ in range(3)]
    eqs = [EQ] * 3
    local = tempfile.mkdtemp(prefix='tps_golden_', dir=os.environ.get('TMPDIR', '/tmp'))
    out = f['multi_run_'](eqs, profs, local, q_surfs_of_interest=[1.5, 2.0],
                          psi_surfs_of_interest=[0.3, 0.95], nvec=[1, 2], run_stride=True,
                          run_pest3=True, vac_flag='f', ode_flag='f', fail_fast=True,
                          **MRE, **REDUCED)
    f['multi_compile_zarr'](eqs, local, meta_data_dicts=profs)
    shutil.copytree(local, wd, dirs_exist_ok=True, ignore=shutil.ignore_patterns('worker_*', '.nfs*'))
    shutil.rmtree(local, ignore_errors=True)
    return out


CASES = dict(nonlinear_wall_rotation=case_nonlinear_wall_rotation,
             nonlinear_nowall=case_nonlinear_nowall,
             linear_wall=case_linear_wall,
             multi_run_zarr=case_multi_run_zarr)


def _save(out_dir, name, result, f):
    with open(os.path.join(out_dir, f'{name}.pkl'), 'wb') as fh:
        pkl.dump(result, fh)
    if isinstance(result, tuple) and isinstance(result[0], xr.Dataset):
        result[0].to_netcdf(os.path.join(out_dir, f'{name}_combined.nc'))
        if name == 'nonlinear_nowall':
            for cut in (0.8, 1.0):
                f['gmq'](result[0], psi_pedestal_cutoff=cut).to_netcdf(
                    os.path.join(out_dir, f'{name}_gmq{cut}.nc'))


def run_case(name, out_dir, f=None):
    """Run one golden case into out_dir (outputs <name>.pkl etc.; work dir out_dir/work/<name>)."""
    f = f or _imports()
    wd = os.path.join(out_dir, 'work', name)
    os.makedirs(wd, exist_ok=True)
    _save(out_dir, name, CASES[name](f, wd), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--cases', nargs='*', default=list(CASES))
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    f = _imports()
    failed = []
    for name in a.cases:
        t0 = time.time()
        print(f'=== {name} ===', flush=True)
        try:
            run_case(name, out_dir, f)
            print(f'=== {name} done in {time.time() - t0:.0f} s ===', flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            print(f'=== {name} FAILED ===', flush=True)
    print('failed:', failed)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
