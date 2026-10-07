"""Helpers shared by the RDCON and STRIDE wrappers (stage, run, read, save)."""
import os
import shutil
import subprocess

import xarray as xr


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

    Returns (xr.Dataset or None, ran). Must be called with the cwd set to working_dir.
    """
    out_nc = os.path.join(working_dir, f'{code}_output_n{nn}.nc')
    if fresh_start and os.path.exists(out_nc):
        os.remove(out_nc)
        if verbose: print(f"Removed existing {code} output file from working directory before running {code}")
    if verbose: print(f"Running {code}...")
    if save_terminal_output:
        status = subprocess.call(f'{working_dir}/{code} > {code}_terminal_output_n{nn}.txt', shell=True)
        if verbose: print(f"{code} terminal output saved to {code}_terminal_output_n{nn}.txt")
    else:
        status = subprocess.call(f'{working_dir}/{code}')
    if status != 0 or not os.path.exists(out_nc):
        return None, False
    return xr.open_dataset(out_nc), True


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
