"""Helpers shared by the RDCON and STRIDE wrappers (stage, run, read, save)."""
import os
import shutil

import xarray as xr

from tearing_physics_suite.utils import CODE_KILLED, run_code

STRIDE_STACK = 1 << 30  # bytes: main-thread stack limit of a STRIDE run (OMP_STACKSIZE 1G for its threads)


def peak_gib(code, mpert, msing, nx=None, banded=False):
    """Predicted peak memory (GiB) of a STRIDE run or an RDCON Delta' run ('rdcon') at one n, from that code's mpert
    and the msing rational surfaces (complex arrays, 16 B). STRIDE (CG, W4 peaks): 2 GiB plus the dense BVP matrix and
    its copy in ode_calc_delta_prime, nMat^2 each, nMat = (2 + 4 msing) mpert; banded (dprime_banded): 0.3 GiB plus
    2.6e-4 mpert^2 (CCG: 109 peaks, mpert 34-77, within -0.2-0.26 GiB of it). RDCON (CCG, gal.f; within 0.1-1.05 GiB
    of 124 measured peaks): 1.1 GiB plus gal_alloc's arrays for solver 'LU' (also the bound for 'cholesky', which can
    fall back to it): the element matrices, 16 mpert^2 per cell over nx (msing + 1) cells, the band, 12 mpert + 1 rows,
    and rhs and sol, 4 msing columns, of ndim = 2 mpert (nx (msing + 1) + 1) + 2 msing unknowns."""
    if code == 'stride':
        return 0.3 + 2.6e-4 * mpert ** 2 if banded else 2 + 32 * ((2 + 4 * msing) * mpert) ** 2 / 2 ** 30
    ndim = 2 * mpert * (nx * (msing + 1) + 1) + 2 * msing
    return 1.1 + 16 * (16 * mpert ** 2 * nx * (msing + 1) + (12 * mpert + 1 + 4 * msing) * ndim) / 2 ** 30


def executable(gpec_dir, code):
    """Path of the built GPEC executable for code ('rdcon' or 'stride'); raises if missing."""
    path = os.path.join(gpec_dir, f'{code}/{code}')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{code} executable not found at {path}")
    return path


def stage_executable(gpec_dir, code, working_dir):
    """Copy the code's executable into working_dir, replacing any old copy."""
    if os.path.exists(os.path.join(working_dir, code)):
        os.remove(os.path.join(working_dir, code))
    shutil.copy(executable(gpec_dir, code), working_dir)
    assert os.path.isfile(os.path.join(working_dir, code))


def run_gpec_code(code, working_dir, nn, fresh_start=True, save_terminal_output=True, verbose=False):
    """Run working_dir/<code> (inputs already written) and read <code>_output_n{nn}.nc.

    Returns (xr.Dataset or None, ran). Must be called with the cwd set to working_dir. A run killed by a signal (e.g.
    an out-of-memory kill) is also recorded in utils.CODE_KILLED.
    """
    out_nc = os.path.join(working_dir, f'{code}_output_n{nn}.nc')
    if fresh_start and os.path.exists(out_nc):
        os.remove(out_nc)
        if verbose: print(f"Removed existing {code} output file from working directory before running {code}")
    if verbose: print(f"Running {code}...")
    # STRIDE's OpenMP threads need a set stack size: under `ulimit -s unlimited` libgomp's default is 2 MB, which
    # crashed it at mpert 105; RDCON is serial (serial LibSci): one OpenMP thread, whatever the worker's setting
    run_env = dict(env={'OMP_STACKSIZE': '1G', **os.environ}, stack_bytes=STRIDE_STACK) if code == 'stride' else \
        dict(env={**os.environ, 'OMP_NUM_THREADS': '1'})
    if save_terminal_output:
        status = run_code(f'{working_dir}/{code} > {code}_terminal_output_n{nn}.txt', **run_env)
        if verbose: print(f"{code} terminal output saved to {code}_terminal_output_n{nn}.txt")
    else:
        status = run_code(f'{working_dir}/{code}', shell=False, **run_env)
    if status < 0 or (save_terminal_output and status > 128):  # the shell reports 128 + signal
        CODE_KILLED.setdefault(code, []).append(nn)
    if status != 0 or not os.path.exists(out_nc):
        return None, False
    ds = xr.open_dataset(out_nc)
    if 'Delta_prime' in ds:  # RDCON and STRIDE label it 'PEST3 Delta Prime Matrix'
        ds['Delta_prime'].attrs['long_name'] = 'Delta prime matrix'
    return ds, True


def save_output(code, ds, path, override_save=False, verbose=False):
    """Save a code's output Dataset to path (netCDF, scipy engine)."""
    if override_save and os.path.isfile(path):
        os.remove(path)
    ds.to_netcdf(path, engine="scipy")
    if verbose: print(f"Saved {code} output to {path}")


def copy_terminal_output(code, working_dir, nn, dest, verbose=False):
    """Copy working_dir/<code>_terminal_output_n{nn}.txt to dest; raises if it is missing."""
    src = os.path.join(working_dir, f'{code}_terminal_output_n{nn}.txt')
    if not os.path.exists(src):
        raise FileNotFoundError(f"{code.capitalize()} terminal output file {code}_terminal_output_n{nn}.txt not found in the working directory.")
    shutil.copy(src, dest)
    if verbose: print(f"Saved {code} terminal output to {dest}")
