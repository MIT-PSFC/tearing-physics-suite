import os
import shutil
import subprocess
import xarray as xr
import pickle as pkl
home_dir = os.environ['TPSHOME']
from tearing_physics_suite.wrappers.gpec_inputs import write_rdcon_stride_inputs
from tearing_physics_suite.utils import eq_stem


def _default_eq_type(eq_filename, kwargs):
    """An OFT i-file (*.ifile) is read by GPEC as eq_type 'ldp_i' unless eq_type is given."""
    if 'eq_type' not in kwargs and str(eq_filename).endswith('.ifile'):
        kwargs['eq_type'] = """'ldp_i'"""


def GPEC_resistive_calculation(eq_filename, nn, run_rdcon=False, run_stride=False, 
        make_working_dir=True, 
        make_results_dir=True,
        working_dir=os.path.join(home_dir, 'working_dir'),
        gpec_dir=os.path.join(home_dir, 'submodules/GPEC'), 
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

    if not (run_rdcon or run_stride): # set warning if both are False
        print("Warning: Neither rdcon nor stride will be run. No calculations will be performed.")

    #########################################################################################################
    # Set up the working directory and executables:
    #########################################################################################################

    # Set up the environment
    rdcon_executable = os.path.join(gpec_dir, 'rdcon/rdcon')
    stride_executable = os.path.join(gpec_dir, 'stride/stride')
    # Check if file exists
    if not os.path.exists(rdcon_executable):
        raise FileNotFoundError(f"rdcon executable not found at {rdcon_executable}")
    if not os.path.exists(stride_executable):
        raise FileNotFoundError(f"stride executable not found at {stride_executable}")

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

    # Clean executables in working directory
    if os.path.exists(os.path.join(working_dir, 'rdcon')):
        os.remove(os.path.join(working_dir, 'rdcon'))
    if os.path.exists(os.path.join(working_dir, 'stride')):
        os.remove(os.path.join(working_dir, 'stride'))

    # Copy executables to working directory
    shutil.copy(rdcon_executable, working_dir)
    shutil.copy(stride_executable, working_dir)
    assert os.path.isfile(working_dir+'/rdcon')
    assert os.path.isfile(working_dir+'/stride')

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

    rdcon_output_name = str(output_prefix + eq_stem(eq_filename) + '_rdcon_n'+str(nn)+'.nc')
    stride_output_name = str(output_prefix + eq_stem(eq_filename) + '_stride_n'+str(nn)+'.nc')

    #########################################################################################################
    # Call executables:
    #########################################################################################################
    
    os.chdir(working_dir)  # Change to the working directory

    rdcon_run=0
    stride_run=0

    if run_rdcon:
        if fresh_start and os.path.exists(os.path.join(working_dir, 'rdcon_output_n'+str(nn)+'.nc')):
            os.remove(os.path.join(working_dir, 'rdcon_output_n'+str(nn)+'.nc'))
            if verbose: print(f"Removed existing rdcon output file from working directory before running rdcon")
        if verbose: print("Running rdcon...")
        if save_terminal_output:
            rdcon_run = subprocess.call(working_dir+'/rdcon > rdcon_terminal_output_n'+str(nn)+'.txt', shell=True)
            if verbose: print("rdcon terminal output saved to rdcon_terminal_output_n"+str(nn)+".txt")
        else:
            rdcon_run = subprocess.call(working_dir+'/rdcon')

    if run_stride:
        if fresh_start and os.path.exists(os.path.join(working_dir, 'stride_output_n'+str(nn)+'.nc')):
            os.remove(os.path.join(working_dir, 'stride_output_n'+str(nn)+'.nc'))
            if verbose: print(f"Removed existing stride output file from working directory before running stride")
        if verbose: print("Running stride...")
        if save_terminal_output:
            stride_run = subprocess.call(working_dir+'/stride > stride_terminal_output_n'+str(nn)+'.txt', shell=True)
            if verbose: print("stride terminal output saved to stride_terminal_output_n"+str(nn)+".txt")
        else:
            stride_run = subprocess.call(working_dir+'/stride')

    #########################################################################################################
    # Confirm successful runs, read outputs:
    #########################################################################################################
    rdcon_xr = None
    stride_xr = None
    rdcon_ran = False
    stride_ran = False

    if run_rdcon:
        if rdcon_run != 0:
            rdcon_xr = None
            rdcon_ran = False
        elif not os.path.exists(os.path.join(working_dir, 'rdcon_output_n'+str(nn)+'.nc')):
            rdcon_xr = None
            rdcon_ran = False
        else:
            rdcon_xr = xr.open_dataset(os.path.join(working_dir, 'rdcon_output_n'+str(nn)+'.nc'))
            rdcon_ran = True

    if run_stride:
        if stride_run != 0:
            stride_xr = None
            stride_ran = False
        elif not os.path.exists(os.path.join(working_dir, 'stride_output_n'+str(nn)+'.nc')):
            stride_xr = None
            stride_ran = False
        else:
            stride_xr = xr.open_dataset(os.path.join(working_dir, 'stride_output_n'+str(nn)+'.nc'))
            stride_ran = True

    #########################################################################################################
    # Save results to output locations:
    #########################################################################################################

    if output_location is not None:
        if run_rdcon and rdcon_xr is not None:
            if override_save and os.path.isfile(os.path.join(output_location, rdcon_output_name)):
                os.remove(os.path.join(output_location, rdcon_output_name))
            rdcon_xr.to_netcdf(os.path.join(output_location, rdcon_output_name), engine="scipy")
            if verbose: print(f"Saved rdcon output to {os.path.join(output_location, rdcon_output_name)}")
        if run_stride and stride_xr is not None:
            if override_save and os.path.isfile(os.path.join(output_location, stride_output_name)):
                os.remove(os.path.join(output_location, stride_output_name))
            stride_xr.to_netcdf(os.path.join(output_location, stride_output_name), engine="scipy")
            if verbose: print(f"Saved stride output to {os.path.join(output_location, stride_output_name)}")
        if save_input:
            if override_save and os.path.isfile(os.path.join(output_location, output_prefix + eq_stem(eq_filename) + '_rdcon_stride_input_n'+str(nn)+'.pkl')):
                os.remove(os.path.join(output_location, output_prefix + eq_stem(eq_filename) + '_rdcon_stride_input_n'+str(nn)+'.pkl'))
            fpkl = open(os.path.join(output_location, output_prefix + eq_stem(eq_filename) + '_rdcon_stride_input_n'+str(nn)+'.pkl'),"wb")
            pkl.dump(rdcon_stride_input_dict,fpkl)
            fpkl.close()
            if verbose: print(f"Saved rdcon and stride input to {os.path.join(output_location, output_prefix + eq_stem(eq_filename) + '_rdcon_stride_input_n'+str(nn)+'.pkl')}")
        if save_terminal_output:
            if run_rdcon and os.path.exists(os.path.join(working_dir, 'rdcon_terminal_output_n'+str(nn)+'.txt')):
                shutil.copy(os.path.join(working_dir, 'rdcon_terminal_output_n'+str(nn)+'.txt'), output_location+ '/' + output_prefix + eq_stem(eq_filename) + '_rdcon_terminal_output_n'+str(nn)+'.txt')
                if verbose: print(f"Saved rdcon terminal output to {os.path.join(output_location, output_prefix + eq_stem(eq_filename) + '_rdcon_terminal_output_n'+str(nn)+'.txt')}")
            elif run_rdcon:
                #raise an error with FileNotFoundError
                raise FileNotFoundError(f"Rdcon terminal output file rdcon_terminal_output_n{nn}.txt not found in the working directory.")
            if run_stride and os.path.exists(os.path.join(working_dir, 'stride_terminal_output_n'+str(nn)+'.txt')):
                shutil.copy(os.path.join(working_dir, 'stride_terminal_output_n'+str(nn)+'.txt'), output_location+ '/' + output_prefix + eq_stem(eq_filename) + '_stride_terminal_output_n'+str(nn)+'.txt')
                if verbose: print(f"Saved stride terminal output to {os.path.join(output_location, output_prefix + eq_stem(eq_filename) + '_stride_terminal_output_n'+str(nn)+'.txt')}")
            elif run_stride:
                #raise an error with FileNotFoundError
                raise FileNotFoundError(f"Stride terminal output file stride_terminal_output_n{nn}.txt not found in the working directory.")

    return rdcon_xr, stride_xr, rdcon_ran, stride_ran, rdcon_stride_input_dict
