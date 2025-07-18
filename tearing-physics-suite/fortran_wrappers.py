# Python functions to call GPEC and PEST3 for delta prime calculations

import os
import sys
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np

import tearing-physics-suite.environment import home_dir
import tearing-physics-suite.GPEC_write_inputs

def PEST3_resistive_calculation(eq_filename, make_working_dir=True,
        working_dir=os.path.join(home_dir, 'working_dir'),
        pest3_dir=os.path.join(home_dir, 'submodules/PEST3/build'),
        verbose=False,
        fresh_start=True,
        output_location=None,
        output_prefix='',
        eq_type_pest=3,             # 3 = efit eqdsk, see pest3_dir/pest3.hh for other options
        nn=1,                       # Toroidal mode number
        kband_pest=14,              # Poloidal Fourier modes span -|kband_pest|,...+|kband_pest|
        rational_surface_control='''-m"xxxxxxxxxxxxxxx"''', # String to control the computation of rational surfaces. Use 'xxxxxxxxxx' to compute Delta's for first 10 rational surfaces. Use '.' symbol to skip a rational surface.
        psilow=1e-4,
        psihigh_pest=0.995,         # This behaves differently from the GPEC psihigh
        pest_b_wall=20,             # Distance of the conformal ideal wall from the plasma in units of minor radius. pest_b_wall > 10 <=> wall at infinity, pest_b_wall = 0 <=> internal mode only. See pest3.hh for more details. 
        mtheta=257,                 # Number of poloidal rays for eqdsk mapping. Large values (~800) likely introduce numerical instabilities.
        mpsi=400,                   # Number of radial grid intervals for equilibrium quantities for eqdsk mapping. Large values (~800) likely introduce numerical instabilities.
        nx_pest_string='''-k"200 100 150 250"''', # String for the number of radial finite elements per non-singular interval. Convergence should obey nx^(-2) going to zero, hence multiple values are specified. Use nxpest for a single value.
        nx_pest=0,                  # Number of radial finite elements per non-singular interval.
        pest3_large_sol_extent=0.9, # See pest3_dir/pest3.hh for details.
        pest3_solver=0,             # String for the PEST3 solver options, see pest3_dir/pest3.hh for details.
        terminal_output_file='',    # To print terminal output to pest3_terminal_output.txt, set terminal_output_file = 'pest3_terminal_output.txt',
        extra_input_string='',      # Extra input string to pass to PEST3, e.g. for additional control parameters. See pest3_dir/pest3.hh for details.
        ):
    """
    Run PEST3 resistive calculation by calling the PEST3 fortran executable to run in a working directory of user choice.
    Prints files both to the working directory and to output_location if specified. Use output_prefix for customizing output filenames.
    For a full list of keyword arguments, see pest3.hh in pest3_dir. Alternatively, run the pest3 executable with ./pest3x -h in pest3_dir.

    Parameters:
    eq_filename (str): Path to the equilibrium file. Cannot be too long (strange old fortran...)
    make_working_dir (bool): If True, create a working directory.
    working_dir (str): Path to the working directory. Will be created if it does not exist and make_working_dir is True.
    pest3_dir (str): Path to the PEST3 build directory.
    verbose (bool): If True, print verbose output.
    fresh_start (bool): If True, remove existing calculations from the working directory.
    output_location (str): If specified, save the output files to this location.
    output_prefix (str): Prefix for the output files.

    eq_type_pest (int): Type of the input equilibrium file for PEST3. Default is 3 (efit eqdsk). See pest3_dir/pest3.hh for other options.
    nn (int): Toroidal mode number.
    kband_pest (int): Poloidal Fourier modes span -|kband_pest|,...+|kband_pest|.
    rational_surface_control (str): String to control the computation of rational surfaces. Use 'xxx' to compute Delta's for the first 3 rational surfaces, '.' to skip.
    psilow (float): Lower bound for the magnetic flux surface. Default is 1e-4.
    psihigh_pest (float): Upper bound for the magnetic flux surface. Default is 0.995. Does not behave the same as GPEC's psihigh (debugging needed).
    pest_b_wall (float): Distance of the conformal ideal wall from the plasma in units of minor radius. Default is 20. pest_b_wall > 10 implies wall at infinity, pest_b_wall = 0 implies internal mode only. See pest3.hh for more details.
    mtheta (int): Number of poloidal rays for eqdsk mapping. Large values (~800) likely introduce numerical instabilities. Default is 257.
    mpsi (int): Number of radial grid intervals for equilibrium quantities for eqdsk mapping. Large values (~800) likely introduce numerical instabilities. Default is 400.
    nx_pest_string (str): String for the number of radial finite elements per non-singular interval. Convergence should obey nx^(-2) going to zero, hence multiple values are specified. 
    nx_pest (int): Number of radial finite elements per non-singular interval. Default is 0, which means nx_pest_string is used.
    pest3_large_sol_extent (float): Large solution extent for PEST3, see pest3_dir/pest3.hh for details. Default is 0.9.
    pest3_solver (str): String for the PEST3 solver options, see pest3_dir/pest3.hh for details.
    terminal_output_control (str): To print terminal output to pest3_terminal_output.txt, use terminal_output_control = ' >& pest3_terminal_output.txt'
    extra_input_string (str): Extra input string to pass to PEST3, e.g. for additional control parameters. See pest3_dir/pest3.hh for details.

    Returns:
    pest3_xr (xarray): Xarray containing the PEST3 output.
    pest3_ran (bool): True if PEST3 was run successfully.
    pest3_input_dict (dict): Dictionary containing all input parameters used for PEST3 calculation.
    """

    # Create a dictionary of input parameters used for PEST3 calculation
    pest3_input_dict = {
        'eq_filename': os.path.basename(eq_filename),
        'eq_type_pest': eq_type_pest,
        'nn': nn,
        'kband_pest': kband_pest,
        'rational_surface_control': rational_surface_control,
        'psilow': psilow,
        'psihigh_pest': psihigh_pest,
        'pest_b_wall': pest_b_wall,
        'mtheta': mtheta,
        'mpsi': mpsi,
        'nx_pest_string': nx_pest_string,
        'nx_pest': nx_pest,
        'pest3_large_sol_extent': pest3_large_sol_extent,
        'pest3_solver': pest3_solver,
        'terminal_output_file': terminal_output_file,
        'extra_input_string': extra_input_string
    }

    #########################################################################################################
    # Set up the working directory and executables:
    #########################################################################################################

    # Set up the environment
    pest3_executable = os.path.join(pest3_dir, 'pest3x')
    # Check if file exists
    if not os.path.exists(pest3_executable):
        raise FileNotFoundError(f"PEST3 executable not found at {pest3_executable}")

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

    # Clean executable in working directory
    if os.path.exists(os.path.join(working_dir, 'pest3x')):
        os.remove(os.path.join(working_dir, 'pest3x'))

    # Copy executable to working directory
    shutil.copy(pest3_executable, working_dir)
    assert os.path.isfile(working_dir+'/pest3x')

    # Check if equilibrium file exists
    if not os.path.exists(eq_filename):
        raise FileNotFoundError(f"Equilibrium file {eq_filename} does not exist.")

    # Move the equilibrium file to the working directory
    shutil.copy(eq_filename, working_dir)
    eq_filename = os.path.basename(eq_filename)  # Get the base name of the equilibrium

    #########################################################################################################
    # Generate input string for PEST3:
    #########################################################################################################

    if nx_pest != 0:
        nx_pest_string = ' -k' + str(nx_pest)
    if len(terminal_output_file) > 0:
        terminal_output_control = ' >& ' + terminal_output_file
        # Remove the file if it exists and fresh_start is True
        if fresh_start and os.path.exists(os.path.join(working_dir, terminal_output_file)):
            os.remove(os.path.join(working_dir, terminal_output_file))
            if verbose: print(f"Removed existing terminal output file {terminal_output_file} from working directory before running pest3")
    else:
        terminal_output_control = ''

    input_string = str('./pest3x -i' + str(eq_type_pest) 
                        + ' -f' + eq_filename 
                        + ' -n' + str(nn) 
                        + ' -b' + str(pest_b_wall) 
                        + ' -l' + str(kband_pest)
                        + ' ' + rational_surface_control
                        + ' -a' + str(psilow) 
                        + ' -P' + str(mtheta) 
                        + ' -R' + str(mpsi) 
                        + ' -E' + str(psihigh_pest)
                        + ' ' + nx_pest_string
                        + ' -d' + str(pest3_large_sol_extent) 
                        + ' -s' + str(pest3_solver)
                        + ' ' + terminal_output_control
                        + ' ' + extra_input_string)

    #########################################################################################################
    # Run PEST3:
    #########################################################################################################

    os.chdir(working_dir)  # Change to the working directory
    if fresh_start and os.path.exists(os.path.join(working_dir, 'pest3.nc')):
        os.remove(os.path.join(working_dir, 'pest3.nc'))
        if verbose: print(f"Removed existing PEST3 output file from working directory before running pest3")

    os.system(input_string)

    #########################################################################################################
    # Confirm successful runs, read outputs:
    #########################################################################################################

    pest3_xr = None
    pest3_ran = False

    if os.path.exists(os.path.join(working_dir, 'pest3.nc')):
        try:
            pest3_xr = xr.open_dataset(os.path.join(working_dir, 'pest3.nc'))
            pest3_ran = True
        except Exception as e:
            print(f"Error reading PEST3 output file: {e}")
            pest3_xr = None
            pest3_ran = False
    else:
        print("PEST3 output file not found in the working directory.")
        pest3_xr = None
        pest3_ran = False

    #########################################################################################################
    # Save results to output locations:
    #########################################################################################################

    if output_location is not None:
        if pest3_xr is not None:
            pest3_output_name = str(output_prefix + eq_filename + '_pest3.nc')
            pest3_xr.to_netcdf(os.path.join(output_location, pest3_output_name))
            if verbose: print(f"Saved PEST3 output to {os.path.join(output_location, pest3_output_name)}")

def GPEC_resistive_calculation(eq_filename, run_rdcon=False, run_stride=False, 
        make_working_dir=True, 
        working_dir=os.path.join(home_dir, 'working_dir'),
        gpec_dir=os.path.join(home_dir, 'submodules/GPEC'), 
        verbose=False,
        fresh_start=True,
        output_location=None,
        output_prefix='',
        **kwargs):
    """
    Run GPEC resistive calculation by calling the rdcon and stride fortran executables to run in a working directory of user choice.
    Prints files both to the working directory and to output_location if specified. Use output_prefix for customizing output filenames.
    
    Parameters:
    eq_filename (str): Path to the equilibrium file. Cannot be too long (strange old fortran...)
    run_rdcon (bool): If True, run rdcon.
    run_stride (bool): If True, run stride.
    make_working_dir (bool): If True, create a working directory.
    working_dir (str): Path to the working directory. Will be created if it does not exist and make_working_dir is True.
    gpec_dir (str): Path to the GPEC directory.
    verbose (bool): If True, print verbose output.
    fresh_start (bool): If True, remove existing calculations from the working directory.
    output_location (str): If specified, save the output files to this location.
    output_prefix (str): Prefix for the output files.
    kwargs: keyword arguments for write_rdcon_stride_inputs.

    Returns:
    rdcon_xr (xarray): Xarray containing the rdcon output.
    stride_xr (xarray): Xarray containing the stride output.
    rdcon_ran (bool): True if rdcon was run successfully.
    stride_ran (bool): True if stride was run successfully.
    rdcon_stride_input_dict (dict): Dictionary containing all input parameters used for rdcon and stride calculations.
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

    rdcon_stride_input_dict = write_rdcon_stride_inputs(working_dir, eq_filename, run_stride, **kwargs)

    rdcon_output_name = str(output_prefix + eq_filename + '_rdcon.nc')
    stride_output_name = str(output_prefix + eq_filename + '_stride.nc')

    #########################################################################################################
    # Call executables:
    #########################################################################################################
    
    os.chdir(working_dir)  # Change to the working directory

    rdcon_run=0
    stride_run=0

    if run_rdcon:
        if fresh_start and os.path.exists(os.path.join(working_dir, 'rdcon_output_n1.nc')):
            os.remove(os.path.join(working_dir, 'rdcon_output_n1.nc'))
            if verbose: print(f"Removed existing rdcon output file from working directory before running rdcon")
        if verbose: print("Running rdcon...")
        rdcon_run = subprocess.call(working_dir+'/rdcon')  

    if run_stride:
        if fresh_start and os.path.exists(os.path.join(working_dir, 'stride_output_n1.nc')):
            os.remove(os.path.join(working_dir, 'stride_output_n1.nc'))
            if verbose: print(f"Removed existing stride output file from working directory before running stride")
        if verbose: print("Running stride...")
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
        elif not os.path.exists(os.path.join(working_dir, 'rdcon_output_n1.nc')):
            rdcon_xr = None
            rdcon_ran = False
        else:
            rdcon_xr = xr.open_dataset(os.path.join(working_dir, 'rdcon_output_n1.nc'))
            rdcon_ran = True

    if run_stride:
        if stride_run != 0:
            stride_xr = None
            stride_ran = False
        elif not os.path.exists(os.path.join(working_dir, 'stride_output_n1.nc')):
            stride_xr = None
            stride_ran = False
        else:
            stride_xr = xr.open_dataset(os.path.join(working_dir, 'stride_output_n1.nc'))
            stride_ran = True

    #########################################################################################################
    # Save results to output locations:
    #########################################################################################################

    if output_location is not None:
        if run_rdcon and rdcon_xr is not None:
            rdcon_xr.to_netcdf(os.path.join(output_location, rdcon_output_name))
            if verbose: print(f"Saved rdcon output to {os.path.join(output_location, rdcon_output_name)}")
        if run_stride and stride_xr is not None:
            stride_xr.to_netcdf(os.path.join(output_location, stride_output_name))
            if verbose: print(f"Saved stride output to {os.path.join(output_location, stride_output_name)}")

    return rdcon_xr, stride_xr, rdcon_ran, stride_ran, rdcon_stride_input_dict