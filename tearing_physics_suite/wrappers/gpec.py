import os
import pickle as pkl
import shutil

from tearing_physics_suite.utils import eq_stem, tps_home
from tearing_physics_suite.wrappers import rdcon as rdcon_wrapper
from tearing_physics_suite.wrappers import stride as stride_wrapper
from tearing_physics_suite.wrappers.gpec_common import copy_terminal_output, executable, save_output
from tearing_physics_suite.wrappers.gpec_inputs import write_rdcon_stride_inputs


def _default_eq_type(eq_filename, kwargs):
    """An OFT i-file (*.ifile) is read by GPEC as eq_type 'ldp_i' unless eq_type is given."""
    if 'eq_type' not in kwargs and str(eq_filename).endswith('.ifile'):
        kwargs['eq_type'] = """'ldp_i'"""


def GPEC_resistive_calculation(eq_filename, nn, run_rdcon=False, run_stride=False,
        make_working_dir=True,
        make_results_dir=True,
        working_dir=None,
        gpec_dir=None,
        verbose=False,
        fresh_start=True,
        output_location=None,
        output_prefix='',
        save_input=True,
        save_terminal_output=False,
        override_save=True,
        **kwargs):
    """
    Run GPEC resistive calculation by calling the rdcon and stride Fortran executables
    in a working directory. GPEC-only subset of run_resistive_calculation.

    Parameters
    ----------
    eq_filename : str
        Path to the equilibrium file.
    nn : int
        Toroidal mode number.
    run_rdcon, run_stride : bool
        Whether to run each GPEC executable.
    working_dir : str
        Path to the working directory.
    verbose : bool
        Print verbose output.
    fresh_start : bool
        Remove existing calculations from working_dir before running.
    output_location : str or None
        If specified, save output files to this location.
    output_prefix : str
        Prefix for output filenames.
    **kwargs
        Forwarded to write_rdcon_stride_inputs.

    Returns
    -------
    rdcon_xr, stride_xr : xr.Dataset or None
        Output xarrays from each executable.
    rdcon_ran, stride_ran : bool
        Whether each executable ran successfully.
    rdcon_stride_input_dict : dict or None
        Input parameters used for the calculations.
    """
    if gpec_dir is None:
        gpec_dir = os.path.join(tps_home(), 'submodules/GPEC')
    if working_dir is None:
        working_dir = os.path.join(tps_home(), 'working_dir')

    if not (run_rdcon or run_stride): # set warning if both are False
        print("Warning: Neither rdcon nor stride will be run. No calculations will be performed.")
    codes = [code for code, run in (('rdcon', run_rdcon), ('stride', run_stride)) if run]
    for code in codes:  # check the executables that will run exist
        executable(gpec_dir, code)

    #########################################################################################################
    # Set up the working directory:
    #########################################################################################################

    if make_working_dir:
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
            if verbose: print(f"Created working directory: {working_dir}")
        else:
            if verbose: print(f"Working directory {working_dir} already exists. Using existing directory.")
    else:
        if (len(working_dir) == 0) or (working_dir is None):
            raise ValueError("Working directory must be specified if make_working_dir is False.")
        if not os.path.exists(working_dir):
            raise FileNotFoundError(f"Working directory {working_dir} does not exist.")
    if make_results_dir:
        if output_location is not None:
            if not os.path.exists(output_location):
                os.makedirs(output_location)
                if verbose: print(f"Created output directory: {output_location}")
            else:
                if verbose: print(f"Output directory {output_location} already exists. Using existing directory.")
        else:
            output_location = working_dir

    # Check if equilibrium file exists
    if not os.path.exists(eq_filename):
        raise FileNotFoundError(f"Equilibrium file {eq_filename} does not exist.")

    # Move the equilibrium file to the working directory
    shutil.copy(eq_filename, working_dir)
    eq_filename = os.path.basename(eq_filename)  # Get the base name of the equilibrium

    #########################################################################################################
    # Write the input files for rdcon and stride & create output names:
    #########################################################################################################

    _default_eq_type(eq_filename, kwargs)
    rdcon_stride_input_dict = write_rdcon_stride_inputs(working_dir, eq_filename, nn=nn, run_stride=run_stride, run_rdcon=run_rdcon, fresh_start=fresh_start, **kwargs)
    out_stem = output_prefix + eq_stem(eq_filename)

    #########################################################################################################
    # Run the codes and read their outputs:
    #########################################################################################################

    os.chdir(working_dir)  # Change to the working directory
    run_opts = dict(fresh_start=fresh_start, save_terminal_output=save_terminal_output, verbose=verbose)
    rdcon_xr, rdcon_ran = rdcon_wrapper.run_rdcon(gpec_dir, working_dir, nn, **run_opts) if run_rdcon else (None, False)
    stride_xr, stride_ran = stride_wrapper.run_stride(gpec_dir, working_dir, nn, **run_opts) if run_stride else (None, False)

    #########################################################################################################
    # Save results to output locations:
    #########################################################################################################

    if output_location is not None:
        for code, ds in (('rdcon', rdcon_xr), ('stride', stride_xr)):
            if ds is not None:
                save_output(code, ds, os.path.join(output_location, f'{out_stem}_{code}_n{nn}.nc'), override_save, verbose)
        if save_input:
            input_pkl = os.path.join(output_location, out_stem + '_rdcon_stride_input_n'+str(nn)+'.pkl')
            if override_save and os.path.isfile(input_pkl):
                os.remove(input_pkl)
            with open(input_pkl, "wb") as fpkl:
                pkl.dump(rdcon_stride_input_dict, fpkl)
            if verbose: print(f"Saved rdcon and stride input to {input_pkl}")
        if save_terminal_output:
            for code in codes:
                copy_terminal_output(code, working_dir, nn, os.path.join(output_location, f'{out_stem}_{code}_terminal_output_n{nn}.txt'), verbose)

    return rdcon_xr, stride_xr, rdcon_ran, stride_ran, rdcon_stride_input_dict
