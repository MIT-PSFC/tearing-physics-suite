# Python functions to call GPEC and PEST3 fortran codes for delta prime calculations

import os
import sys
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import CubicSpline
import pickle as pkl
import copy
import netCDF4

from tearing_physics_suite.environment import home_dir
from tearing_physics_suite.GPEC_write_inputs import write_rdcon_stride_inputs

def run_resistive_calculation(eq_filename, nn, run_rdcon=False, run_stride=False, run_pest3=False,
        make_working_dir=True,
        working_dir=os.path.join(home_dir, 'working_dir'),
        gpec_dir=os.path.join(home_dir, 'submodules/GPEC'), 
        pest3_dir=os.path.join(home_dir, 'submodules/PEST3/build/pest3'),
        verbose=False,
        fresh_start=True,
        output_location=None,
        output_prefix='',
        save_input=True,
        save_terminal_output=False,
        pest_match_truncation=True,
        override_save=True, 
        debug_GPEC_resistive_calculation=False,
        **kwargs):
    """
    Run resistive toroidal calculation for a single toroidal mode number by calling the GPEC and PEST3 fortran executables to run in a working directory of user choice.
    Prints files both to the working directory and to output_location if specified. Use output_prefix for customizing output filenames.

    Parameters:
    eq_filename (str): Path to the equilibrium file. Cannot be too long (strange old fortran...)
    nn (int): Toroidal mode number.
    run_rdcon (bool): If True, run rdcon.
    run_stride (bool): If True, run stride.
    run_pest3 (bool): If True, run PEST3.
    make_working_dir (bool): If True, create a working directory.
    working_dir (str): Path to the working directory. Will be created if it does not exist and make_working_dir is True.
    gpec_dir (str): Path to the GPEC directory.
    pest3_dir (str): Path to the PEST3 build directory.
    verbose (bool): If True, print verbose output.
    fresh_start (bool): If True, remove existing calculations from the working directory.
    output_location (str): If specified, save the output files to this location.
    output_prefix (str): Prefix for the output files.
    save_input (bool): If True, save the input parameters used for rdcon, stride, and PEST3 calculations to a netCDF file.
    save_terminal_output (bool): If True, save the terminal output to rdcon_terminal_output_n{nn}.txt, stride_terminal_output_n{nn}.txt, and pest3_terminal_output_n{nn}.txt in the working directory. If fresh_start is True, the files will be removed if they exist.
    pest_match_truncation (bool): If True, match the truncation of the PEST3 calculation to the GPEC/STRIDE calculations. If you enter kwarg psihigh_pest, however, this will override the automatic truncation loop.
    override_save (bool): If True, will override pre-existing saved files with the same name during operation.
    kwargs: keyword arguments for write_rdcon_stride_inputs and PEST3_resistive_calculation.

    Returns:
    rdcon_xr (xarray): Xarray containing the rdcon output.
    stride_xr (xarray): Xarray containing the stride output.
    pest3_xr (xarray): Xarray containing the PEST3 output.
    rdcon_ran (bool): True if rdcon was run successfully.
    stride_ran (bool): True if stride was run successfully.
    pest3_ran (bool): True if PEST3 was run successfully.
    rdcon_stride_input_dict (dict): contains all input parameters used for rdcon, stride calculations.
    pest3_input_dict (dict): contains all input parameters used for PEST3 calculation.
    """

    #Extract keyword arguments for PEST3
    pest3_kwargs_dict = {k: v for k, v in kwargs.items() if k.endswith('_pest')}
    if not override_save:
        pest3_kwargs_dict.update({"override_save":False})

    #Remove PEST3 specific keyword arguments from kwargs
    for key in pest3_kwargs_dict.keys():
        if key in kwargs:
            del kwargs[key]

    # Run GPEC resistive calculation
    rdcon_xr, stride_xr, rdcon_ran, stride_ran, rdcon_stride_input_dict = GPEC_resistive_calculation(
        eq_filename=eq_filename, nn=nn, run_rdcon=run_rdcon, run_stride=run_stride,
        make_working_dir=make_working_dir, working_dir=working_dir, gpec_dir=gpec_dir,
        verbose=verbose, fresh_start=fresh_start, output_location=output_location,
        output_prefix=output_prefix, save_input=save_input, save_terminal_output=save_terminal_output,
        **kwargs)

    if debug_GPEC_resistive_calculation:
        return rdcon_xr, stride_xr, rdcon_ran, stride_ran, rdcon_stride_input_dict

    #########################################################################################################
    # Set up pest3 calculation:
    #########################################################################################################

    # Extract key truncation values from GPEC calculation:
    qlim_actual = -100
    psilow_actual = 100
    qlim_actuals = -100
    psilow_actuals = 100
    # Define maximum poloidal fourier harmonic with the same logic as in GPEC:
    m_max=0
    m_maxs=0
    m_min=0
    m_mins=0
    delta_mhigh=rdcon_stride_input_dict['delta_mhigh']
    delta_mlow=rdcon_stride_input_dict['delta_mlow']
    if run_rdcon:
        m_max = int(np.ceil(rdcon_xr.qmax*nn+delta_mhigh))
        m_min = int(np.floor(min(rdcon_xr.qmin*nn,0)-4-delta_mlow))
        qlim_actual = max(rdcon_xr.qlim, qlim_actual)
        psilow_actual = min(rdcon_xr.psilow, psilow_actual)
    if run_stride:
        m_maxs = int(np.ceil(stride_xr.qmax*nn+delta_mhigh))
        m_mins = int(np.floor(min(stride_xr.qmin*nn,0)-4-delta_mlow))
        qlim_actuals = max(stride_xr.qlim, qlim_actuals)
        psilow_actuals = min(stride_xr.psilow, psilow_actuals)
    m_max = max(m_max, m_maxs)
    m_min = min(m_min, m_mins)
    if run_stride and run_rdcon and 2*(abs(qlim_actual-qlim_actuals)/(abs(qlim_actual)+abs(qlim_actual))) > 1e-4:
        print("WARNING, rdcon and stride truncation is differing.")
        print("qlims: ", qlim_actual, qlim_actuals)
    if run_stride and run_rdcon and 2*(abs(psilow_actual-psilow_actuals)/(abs(psilow_actuals)+abs(psilow_actuals))) > 1e-4:
        print("WARNING, rdcon and stride truncation is differing.")
        print("psilows: ", psilow_actual,psilow_actuals)
    if run_stride and run_rdcon:
        qlim_actual = min(qlim_actual, qlim_actuals)
        psilow_actual = max(psilow_actual,psilow_actuals)
    elif run_stride:
        qlim_actual=qlim_actuals
        psilow_actual=psilow_actuals
    m_absmax = max(abs(m_max), abs(m_min))
    if verbose: print("Truncation values: psilow", psilow_actual,"qlim", qlim_actual)

    # Check for explicit pest inputs, else use the values from rdcon_stride_input_dict:
    if ('kband_pest' not in pest3_kwargs_dict) and m_absmax>0:
        pest3_kwargs_dict['kband_pest'] = m_absmax

    if 'psilow_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['psilow_pest'] = rdcon_stride_input_dict['psilow']

    if 'psihigh_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['psihigh_pest'] = rdcon_stride_input_dict['psihigh']
        allow_trunc_loop=True 
    else:
        if verbose: print("Using user defined PEST3 truncation value psihigh_pest = ", pest3_kwargs_dict['psihigh_pest'])
        allow_trunc_loop=False #If you enter psihigh_pest, will override automatic truncation loop 

    if 'a_wall_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['a_wall_pest'] = rdcon_stride_input_dict['a_wall']

    #if 'mtheta_pest' not in pest3_kwargs_dict:
        #pest3_kwargs_dict['mtheta_pest'] = rdcon_stride_input_dict['mtheta']
        #mtheta_pest=129 appears to be hardcoded into PEST3, changing this causes issues in my experience.

    if 'mpsi_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['mpsi_pest'] = rdcon_stride_input_dict['mpsi']

    #########################################################################################################
    # Run PEST3 resistive calculation:
    #########################################################################################################
    
    if run_pest3:
        pest3_trunc_ran = False
        if pest_match_truncation and (run_rdcon or run_stride) and allow_trunc_loop:
            if verbose: print("Running pest3 truncation algorithm. qlim_actual = ", qlim_actual)
            if psilow_actual > rdcon_stride_input_dict['psilow'] and (psilow_actual != 100):
                print("WARNING: axis truncation in RDCON/STRIDE differs from PEST3 truncation. Results may not be comparable.")
            if qlim_actual > -100:
                psihigh_trunc_pest, pest3_trunc_ran = pest3_special_truncation_loop(eq_filename, nn, qlim_actual, pest3_kwargs_dict,
                    make_working_dir=make_working_dir,
                    working_dir=working_dir,
                    pest3_dir=pest3_dir,
                    verbose=verbose,
                    fresh_start=fresh_start,
                    output_location=None,
                    save_input=False,
                    output_prefix_special=output_prefix,
                    save_terminal_output=save_terminal_output,
                )
                if pest3_trunc_ran:
                    pest3_kwargs_dict['psihigh_pest'] = psihigh_trunc_pest

        pest3_xr, pest3_ran, pest3_input_dict = PEST3_resistive_calculation(
            eq_filename=eq_filename, nn=nn, make_working_dir=make_working_dir,
            working_dir=working_dir, pest3_dir=pest3_dir, verbose=verbose,
            fresh_start=fresh_start, output_location=output_location,
            output_prefix=output_prefix, save_input=save_input,
            save_terminal_output=save_terminal_output, **pest3_kwargs_dict)

        if verbose:
            print("Pest3 ran:",pest3_ran, "Pest3 truncation ran:", pest3_trunc_ran)        

        # See how accurate the truncation was:
        if pest3_ran and pest3_trunc_ran:
            if verbose:
                print(f"Verbose output: PEST3 q-truncation point {pest3_xr['qa'].max().values}, GPEC q-truncation point {qlim_actual}.")
            if abs(pest3_xr['qa'].max().values - qlim_actual) > 0.01:
                print(f"WARNING: PEST3 q-truncation point {pest3_xr['qa'].max().values} differs from GPEC q-truncation point {qlim_actual}. Results may not be comparable.")
    else:
        pest3_xr = None
        pest3_ran = False
        pest3_input_dict = None

    # Return all results:
    return rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict

def PEST3_resistive_calculation(eq_filename, nn, make_working_dir=True,
        working_dir=os.path.join(home_dir, 'working_dir'),
        pest3_dir=os.path.join(home_dir, 'submodules/PEST3/build/pest3'),
        verbose=True,
        debug=False,
        fresh_start=True,
        output_location=None,
        output_prefix='',
        save_input=True,
        save_terminal_output=False,
        override_save=True,
        # PEST3 specific parameters:
        eq_type_pest=3,             # 3 = efit eqdsk, see pest3_dir/pest3.hh for other options
        kband_pest=14,              # Poloidal Fourier modes span -|kband_pest|,...+|kband_pest|
        rational_surface_control_pest='''-m"xxxxxxxxxxxxxxx"''', # String to control the computation of rational surfaces. Use 'xxxxxxxxxx' to compute Delta's for first 10 rational surfaces. Use '.' symbol to skip a rational surface.
        psilow_pest=1e-4,
        psihigh_pest=0.995,         # This behaves differently from the GPEC psihigh
        a_wall_pest=20,             # Distance of the conformal ideal wall from the plasma in units of minor radius. a_wall_pest > 10 <=> wall at infinity, a_wall_pest = 0 <=> internal mode only. See pest3.hh for more details. 
        mtheta_pest=129, #DO NOT CHANGE - HARDCODED INTO PEST3 - # Number of poloidal rays for eqdsk mapping. Large values (~800) likely introduce numerical instabilities.
        mpsi_pest=400,                   # Number of radial grid intervals for equilibrium quantities for eqdsk mapping. Large values (~800) likely introduce numerical instabilities.
        nx_string_pest='''-k"100 70 140 200"''', # String for the number of radial finite elements per non-singular interval. Convergence should obey nx^(-2) going to zero, hence multiple values are specified. Use nxpest for a single value.
        nx_pest=0,                  # Number of radial finite elements per non-singular interval.
        large_sol_extent_pest=0.9, # See pest3_dir/pest3.hh for details.
        solver_pest=0,             # String for the PEST3 solver options, see pest3_dir/pest3.hh for details.
        extra_input_string_pest='',      # Extra input string to pass to PEST3, e.g. for additional control parameters. See pest3_dir/pest3.hh for details.
        ):
    """
    Run PEST3 resistive calculation by calling the PEST3 fortran executable to run in a working directory of user choice.
    Prints files both to the working directory and to output_location if specified. Use output_prefix for customizing output filenames.
    For a full list of keyword arguments, see pest3.hh in pest3_dir. Alternatively, run the pest3 executable with ./pest3x -h in pest3_dir.

    Parameters:
    eq_filename (str): Path to the equilibrium file. Cannot be too long (strange old fortran...)
    nn (int): Toroidal mode number.
    make_working_dir (bool): If True, create a working directory.
    working_dir (str): Path to the working directory. Will be created if it does not exist and make_working_dir is True.
    pest3_dir (str): Path to the PEST3 build directory.
    verbose (bool): If True, print verbose output.
    fresh_start (bool): If True, remove existing calculations from the working directory.
    output_location (str): If specified, save the output files to this location.
    output_prefix (str): Prefix for the output files.
    save_input (bool): If True, save the input parameters used for PEST3 calculation to a netCDF file.
    save_terminal_output (bool): If True, save the terminal output to pest3_terminal_output_n{nn}.txt in the working directory. If fresh_start is True, the file will be removed if it exists.

    eq_type_pest (int): Type of the input equilibrium file for PEST3. Default is 3 (efit eqdsk). See pest3_dir/pest3.hh for other options.
    kband_pest (int): Poloidal Fourier modes span -|kband_pest|,...+|kband_pest|.
    rational_surface_control_pest (str): String to control the computation of rational surfaces. Use 'xxx' to compute Delta's for the first 3 rational surfaces, '.' to skip.
    psilow_pest (float): Lower bound for the magnetic flux surface. Default is 1e-4.
    psihigh_pest (float): Upper bound for the magnetic flux surface. Default is 0.995. Does not behave the same as GPEC's psihigh (debugging needed).
    a_wall_pest (float): Distance of the conformal ideal wall from the plasma in units of minor radius. Default is 20. a_wall_pest > 10 implies wall at infinity, a_wall_pest = 0 implies internal mode only. See pest3.hh for more details.
    mtheta_pest (int): Number of poloidal rays for eqdsk mapping. Large values (~800) likely introduce numerical instabilities. Default is 257.
    mpsi_pest (int): Number of radial grid intervals for equilibrium quantities for eqdsk mapping. Large values (~800) likely introduce numerical instabilities. Default is 400.
    nx_string_pest (str): String for the number of radial finite elements per non-singular interval. Convergence should obey nx^(-2) going to zero, hence multiple values are specified. 
    nx_pest (int): Number of radial finite elements per non-singular interval. Default is 0, which means nx_string_pest is used.
    large_sol_extent_pest (float): Large solution extent for PEST3, see pest3_dir/pest3.hh for details. Default is 0.9.
    solver_pest (str): String for the PEST3 solver options, see pest3_dir/pest3.hh for details.
    extra_input_string_pest (str): Extra input string to pass to PEST3, e.g. for additional control parameters. See pest3_dir/pest3.hh for details.

    Returns:
    pest3_xr (xarray): Xarray containing the PEST3 output.
    pest3_ran (bool): True if PEST3 was run successfully.
    pest3_input_dict (dict): Dictionary containing all input parameters used for PEST3 calculation.
    """

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
        nx_string_pest = ' -k' + str(nx_pest)

    terminal_output_file = 'pest3_terminal_output_n'+str(nn)+'.txt' 
    if save_terminal_output:
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
                        + ' -b' + str(a_wall_pest) 
                        + ' -l' + str(kband_pest)
                        + ' ' + rational_surface_control_pest
                        + ' -a' + str(psilow_pest) 
                        + ' -P' + str(mtheta_pest) 
                        + ' -R' + str(mpsi_pest) 
                        + ' -E' + str(psihigh_pest)
                        + ' ' + nx_string_pest
                        + ' -d' + str(large_sol_extent_pest) 
                        + ' -s' + str(solver_pest)
                        + ' -x0'
                        + ' -a0.001'
                        + ' ' + terminal_output_control
                        + ' ' + extra_input_string_pest)

    #########################################################################################################
    # Run PEST3:
    #########################################################################################################

    os.chdir(working_dir)  # Change to the working directory
    if fresh_start and os.path.exists(os.path.join(working_dir, 'pest3_n'+str(nn)+'.nc')):
        os.remove(os.path.join(working_dir, 'pest3_n'+str(nn)+'.nc'))
        if verbose: print(f"Removed existing PEST3 output file from working directory before running pest3")

    os.system(input_string)

    #########################################################################################################
    # Create a dictionary of input parameters used for PEST3 calculation
    #########################################################################################################
       
    pest3_input_dict = {
        'eq_filename': os.path.basename(eq_filename),
        'eq_type_pest': eq_type_pest,
        'nn': nn,
        'kband_pest': kband_pest,
        'rational_surface_control_pest': rational_surface_control_pest,
        'psilow_pest': psilow_pest,
        'psihigh_pest': psihigh_pest,
        'a_wall_pest': a_wall_pest,
        'mtheta_pest': mtheta_pest,
        'mpsi_pest': mpsi_pest,
        'nx_string_pest': nx_string_pest,
        'nx_pest': nx_pest,
        'large_sol_extent_pest': large_sol_extent_pest,
        'solver_pest': solver_pest,
        'save_terminal_output': save_terminal_output,
        'extra_input_string_pest': extra_input_string_pest
    }

    #########################################################################################################
    # Confirm successful runs, read outputs:
    #########################################################################################################

    pest3_xr = None
    pest3_ran = False

    if os.path.exists(os.path.join(working_dir, 'pest3.nc')):
        #try:
        ps3 = xr.open_dataset(os.path.join(working_dir, 'pest3.nc'))
        if debug: 
            print(f"Successfully read PEST3 output file: {os.path.join(working_dir, 'pest3.nc')}")
            print(ps3)

        pest3_xr = pest3_clean_netcdf(ps3,debug=debug)
        pest3_ran = True
        #except Exception as e:
        #    print(f"Error reading PEST3 output fi`le: {e}")
        #    pest3_xr = None
        #    pest3_ran = False
    else:
        print("PEST3 output file not found in the working directory.")
        pest3_xr = None
        pest3_ran = False

    #########################################################################################################
    # Save results to output locations:
    #########################################################################################################

    if output_location is not None:
        if pest3_xr is not None:
            pest3_output_name = str(output_prefix + eq_filename + '_pest3_n'+str(nn)+'.nc')
            if override_save and os.path.isfile(os.path.join(output_location, pest3_output_name)):
                os.remove(os.path.join(output_location, pest3_output_name))
            pest3_xr.to_netcdf(os.path.join(output_location, pest3_output_name))
            if verbose: print(f"Saved PEST3 output to {os.path.join(output_location, pest3_output_name)}")
        if save_terminal_output:
            if os.path.exists(os.path.join(working_dir, terminal_output_file)):
                shutil.copy(os.path.join(working_dir, terminal_output_file), output_location+ '/' + output_prefix + eq_filename + terminal_output_file)
                if verbose: print(f"Saved terminal output to {os.path.join(output_location, output_prefix + eq_filename + terminal_output_file)}")
            else:
                #raise an error with FileNotFoundError
                raise FileNotFoundError(f"Pest3 terminal output file {terminal_output_file} not found in the working directory.")
        if save_input:
            if override_save and os.path.isfile(os.path.join(output_location, output_prefix + eq_filename + '_pest3_input_n'+str(nn)+'.pkl')):
                os.remove(os.path.join(output_location, output_prefix + eq_filename + '_pest3_input_n'+str(nn)+'.pkl'))
            fpkl = open(os.path.join(output_location, output_prefix + eq_filename + '_pest3_input_n'+str(nn)+'.pkl'),"wb")
            pkl.dump(pest3_input_dict,fpkl)
            fpkl.close()
            if verbose: print(f"Saved PEST3 input to {os.path.join(output_location, output_prefix + eq_filename + '_pest3_input_n'+str(nn)+'.pkl')}")

    return pest3_xr, pest3_ran, pest3_input_dict


def pest3_special_truncation_loop(eq_filename, nn, qlim_actual, pest3_kwargs_dict, output_prefix_special='', 
                                mpsi_trunc_loop=129, 
                                nx_trunc_loop=20,
                                debug=False,
                                verbose=True,
                                truncimax=10,
                                nx_truncdebug=False, **kwargs):
    """
    Sets psihigh values for PEST3 based on the q-truncation point of the GPEC calculation (qlim_actual). Does so by running 
    a very fast PEST3 calculation with a single toroidal mode number (nn=1), creating a spline from the PEST3 q output vs psi 
    output, and finding the root of the spline at the value of qlim_actual.
    
    Parameters:
    eq_filename (str): Path to the equilibrium file.
    nn (int): Toroidal mode number.
    qlim_actual (float): Actual q limit value from GPEC/STRIDE calculations.

    Returns:
    float: The psihigh value to be used in PEST3 calculation.
    """

    pest3_kwargs_dict_local=copy.deepcopy(pest3_kwargs_dict)

    #Remove save_terminal_output from pest3_kwargs_dict_local
    pest3_kwargs_dict_local.pop('save_terminal_output', None)
    pest3_kwargs_dict_local.pop('verbose',None)
    kwargs.pop('save_terminal_output', None)
    kwargs.pop('verbose',None)

    psihigh_trunc_single, pest3_trunc_single_ran = pest3_special_truncation_single(
        eq_filename, nn, qlim_actual, pest3_kwargs_dict_local, save_terminal_output=True, verbose=False, **kwargs)
    if not pest3_trunc_single_ran:
        print("PEST3 truncation single run failed.")
        return 0, False

    output_prefix_special+='_truncloop_'

    # Setting up a very fast, basic pest run:
    pest3_kwargs_dict_local['mpsi_pest'] = mpsi_trunc_loop
    pest3_kwargs_dict_local['mtheta_pest'] = 129
    pest3_kwargs_dict_local['kband_pest'] = 15
    pest3_kwargs_dict_local['a_wall_pest'] = 0
    pest3_kwargs_dict_local['psilow_pest'] = 0.0001
    pest3_kwargs_dict_local['large_sol_extent_pest'] = 0.1
    pest3_kwargs_dict_local['nx_pest'] = nx_trunc_loop
    pest3_kwargs_dict_local['rational_surface_control_pest'] = '''-m"."'''

    # Combine pest3_kwargs_dict_local and kwargs
    pest3_kwargs_dict_local.update(kwargs)

    psihigh_bounds = [psihigh_trunc_single,1.0]

    #Checking start:
    pest3_kwargs_dict_local['psihigh_pest'] = psihigh_bounds[0]
    pest3_xr, pest3_trunc_ran, pest3_input_dict = PEST3_resistive_calculation(
        eq_filename,
        1, # nn = 1 is fastest
        output_prefix=output_prefix_special,
        verbose=False,
        **pest3_kwargs_dict_local
    )
    if abs(pest3_xr['qa'].values[-1]-qlim_actual) < 0.01:
        return pest3_kwargs_dict_local['psihigh_pest'], True
    assert pest3_xr['qa'].values[-1] <= qlim_actual, "PEST3 initial run did not return a q value below qlim_actual. Check input parameters."

    #Bisection method:
    trunci=0
    pest3_kwargs_dict_local['psihigh_pest'] = psihigh_bounds[0]+(psihigh_bounds[1]-psihigh_bounds[0])/2.0

    while trunci < truncimax:
        pest3_xr, pest3_trunc_ran, pest3_input_dict = PEST3_resistive_calculation(
            eq_filename,
            1, # nn = 1 is fastest
            save_terminal_output=True,
            output_prefix=output_prefix_special,
            verbose=False,
            **pest3_kwargs_dict_local
        )

        if verbose:
            print(f"PEST3 run {trunci}: psihigh_pest = {pest3_kwargs_dict_local['psihigh_pest']}, qlim_actual = {qlim_actual}, qa = {pest3_xr['qa'].values[-1]}")

        if abs(pest3_xr['qa'].values[-1]-qlim_actual) < 0.01:
            return pest3_kwargs_dict_local['psihigh_pest'], True
        elif pest3_xr['qa'].values[-1] < qlim_actual:
            psihigh_bounds[0] = pest3_kwargs_dict_local['psihigh_pest']
        else:
            psihigh_bounds[1] = pest3_kwargs_dict_local['psihigh_pest']
        pest3_kwargs_dict_local['psihigh_pest'] = psihigh_bounds[0]+(psihigh_bounds[1]-psihigh_bounds[0])/2.0

        if not pest3_trunc_ran:
            print("PEST3 truncation run failed.")
            return 0, False
        trunci += 1

    return psi_trunc_frac, pest3_trunc_ran

def pest3_special_truncation_single(eq_filename, nn, qlim_actual, pest3_kwargs_dict, output_prefix_special='', 
                                mpsi_trunc=400, 
                                nx_trunc=1000,
                                debug=False,
                                nx_truncdebug=False, **kwargs):
    """
    Sets psihigh values for PEST3 based on the q-truncation point of the GPEC calculation (qlim_actual). Does so by running 
    a very fast PEST3 calculation with a single toroidal mode number (nn=1), creating a spline from the PEST3 q output vs psi 
    output, and finding the root of the spline at the value of qlim_actual.
    
    Parameters:
    eq_filename (str): Path to the equilibrium file.
    nn (int): Toroidal mode number.
    qlim_actual (float): Actual q limit value from GPEC/STRIDE calculations.

    Returns:
    float: The psihigh value to be used in PEST3 calculation.
    """

    pest3_kwargs_dict_local=copy.deepcopy(pest3_kwargs_dict)

    output_prefix_special+='_trunc_single_'

    # Setting up a high-res spline to get a good starting point
    pest3_kwargs_dict_local['psihigh_pest'] = 1.0 #won't break pest (?)
    pest3_kwargs_dict_local['mpsi_pest'] = mpsi_trunc
    pest3_kwargs_dict_local['mtheta_pest'] = 129
    pest3_kwargs_dict_local['kband_pest'] = 15
    pest3_kwargs_dict_local['a_wall_pest'] = 0
    pest3_kwargs_dict_local['psilow_pest'] = 0.0001
    pest3_kwargs_dict_local['large_sol_extent_pest'] = 0.1
    pest3_kwargs_dict_local['nx_pest'] = nx_trunc
    pest3_kwargs_dict_local['rational_surface_control_pest'] = '''-m"."'''

    # Combine pest3_kwargs_dict_local and kwargs
    pest3_kwargs_dict_local.update(kwargs)

    pest3_xr, pest3_trunc_ran, pest3_input_dict = PEST3_resistive_calculation(
        eq_filename,
        1, # nn = 1 is fastest
        output_prefix=output_prefix_special,
        **pest3_kwargs_dict_local
    )

    if not pest3_trunc_ran:
        print("PEST3 truncation run failed.")
        return 0, False

    # Check if qlim_actual is within the range of q values in pest3_xr['qa'].values:
    if qlim_actual < pest3_xr['qa'].values[0] or qlim_actual > pest3_xr['qa'].values[-1]:
        print(f"Warning: qlim_actual {qlim_actual} is outside the range of q values in PEST3: {pest3_xr['qa'].values[0]} to {pest3_xr['qa'].values[-1]}.")
        return 0, False

    # Make cubic spline of pest3_xr.psinew.values and psipest3_xr.qa.values, shifted to the value of qlim_actual:
    cs = CubicSpline(pest3_xr['psinew'].values, (pest3_xr['qa'].values-qlim_actual))
    roots = cs.roots()

    # Check if the cubic spline has roots:
    if len(roots) == 0:
        print("No roots found in the cubic spline. Cannot determine truncation point.")
        return 0, False

    if len(roots) > 1:
        print("WARNING: Multiple roots found for the pest3 q truncation point. Using the smallest one: " + str(roots[0]))

    psi_trunc_val = roots[0]

    #Convert psi_trunc_val to a fraction of pest3_xr['psinew'].values[-1]-pest3_xr['psinew'].values[0]:
    psi_trunc_frac = (psi_trunc_val - pest3_xr['psinew'].values[0]) / (pest3_xr['psinew'].values[-1] - pest3_xr['psinew'].values[0])

    if debug:
        print(" Psi norm values: ", pest3_xr['psinew'].values/(pest3_xr['psinew'].values[-1] - pest3_xr['psinew'].values[0]))
        print(" q values: ", pest3_xr['qa'].values)

    if debug:
        return psi_trunc_frac, pest3_trunc_ran, pest3_xr['qa'].values, (pest3_xr['psinew'].values/(pest3_xr['psinew'].values[-1] - pest3_xr['psinew'].values[0]))
    return psi_trunc_frac, pest3_trunc_ran

def pest3_clean_netcdf(ps3, debug=False):
    """
    Process the PEST3 netCDF output to standardize dimensions and variable names. If multiple rational
    surfaces are calculated, redefinies rescaled Delta primes in GPEC units.

    Parameters:
    ps3 (xarray.Dataset): The PEST3 output dataset.
    debug (bool): If True, print debug information.

    Returns:
    xarray.Dataset: The processed PEST3 dataset with standardized dimensions and variable names.
    """

    #########################################################################################################
    # Standardize dimensions and variable names in PEST3 output:
    #########################################################################################################
    #Names of dims:
    if len(ps3.cmatch.dims) > 0:
        surfdim=ps3.cmatch.dims[0]
        unknowndim=ps3.x1frbo_re.dims[1]
    else: #No rational surfaces calculated
        surfdim=None
        unknowndim=ps3.x1frbo_re.dims[0]
    profdim_1=ps3.psinod.dims[0]
    profdim_2=ps3.qa.dims[0]
    thetadim=ps3.xjacob.dims[1]
    #Rename dimensions to standardize & remove duplicates
    for varname, da in ps3.data_vars.items():
        new_dims = []
        for dim_i in da.dims:
            if (dim_i == surfdim) and ('r' in new_dims):
                new_dims.append('r_prime')
            elif dim_i == surfdim:
                new_dims.append('r')
            elif dim_i == profdim_1:
                new_dims.append('psinod_dim')
            elif dim_i == profdim_2:
                new_dims.append('qprof_dim')
            elif dim_i == thetadim:
                new_dims.append('theta_dim')
            elif dim_i == unknowndim:
                new_dims.append('ukn_dim')
            else:
                new_dims.append(dim_i)
        if debug: print(len(da.dims),new_dims)
        if len(da.dims) > 0:
            tempvals = da.values
            temp_da = xr.DataArray(tempvals, dims=tuple(new_dims))
            ps3[varname] = temp_da
    
    if len(ps3.cmatch.dims) > 0:
        return pest3_rescale_deltaprimes(ps3)
    return ps3

def pest3_rescale_deltaprimes(ps3, debug=False):
    """
    Rescale the Delta prime values in the PEST3 output to GPEC units.
    This function converts the Delta prime values from PEST3's radial flux coordinate to GPEC's radial flux coordinate.

    Parameters:
    ps3 (xarray.Dataset): The PEST3 output dataset.
    debug (bool): If True, print debug information.

    Returns:
    xarray.Dataset: The PEST3 dataset with rescaled Delta prime values.

    More info:
    PEST3 uses radial flux coordinate = poloidal flux / 2pi, while GPEC uses radial flux coordinate
    = (poloidal flux/2pi) normalised to be zero at the magnetic axis and one at the edge.
    Delta prime values (on diagnoal) in GPEC have units psi^{-2sqrt(-Di)} for GPEC flux coordinate
    psi. For true non-dimensionalisation, we need to multiply the GPEC delta prime values by
    psi_s^{-2sqrt(-Di)} (on diagnoal) where psi_s is psi at the rational surface.
    See PEST3/pest/pest3.m for the original version of this non-dimensionalisation.
    Also A_prime_perr, B_prime_perr, Gamma_prime_perr, Delta_prime_perr are the errors in these values
    assuming you used multiple nx values in the PEST3 calculation to get convergence (see Rosenburg PoP 2002 Fig. 3).   
    """

    ps3=ps3.assign(psio=ps3.psimax-ps3.psimin)
    ps3=ps3.assign(psio_alt=ps3.psia-ps3.psi0)

    assert ps3.psio.values-ps3.psio_alt.values < 1e-4, "Competing values of total (pol. flux/2pi). Check PEST3 output."

    #List of variables to rename:
    varlista=[['aprim_re','aprim_im'], ['bprim_re', 'bprim_im'], ['gprim_re', 'gprim_im'], ['dprim_re', 'dprim_im']]
    namelista=['A_prime', 'B_prime', 'Gamma_prime', 'Delta_prime']
    varlistb=['error_aprim','error_bprim','error_gprim','error_dprim']
    namelistb=['A_prime_perr', 'B_prime_perr', 'Gamma_prime_perr', 'Delta_prime_perr']

    assert len(varlista) == len(namelista) == len(varlistb) == len(namelistb)

    #Make 2D matrix of ps3.xmu vector as ps3.xmu[0]ps3.xmu[0], ps3.xmu[0]ps3.xmu[1], ..., ps3.xmu[-1]ps3.xmu[-1]:
    psio_norm_array=xr.DataArray(ps3.psio.values**np.add.outer(ps3.xmu.values, ps3.xmu.values), dims=('r', 'r_prime'))
    if debug: print("xmu ", ps3.xmu.values)
    ps3['psio_norm_array']= psio_norm_array
    #if debug: print("xmu sum matrix ", ps3.xmu_sum_matrix.values)
    if debug: print("ps3.psio.values ", ps3.psio.values)
    #ps3 = ps3.assign(psio_norm_array=np.power(ps3.psio.values,ps3.xmu_sum_matrix.values)+0.0*ps3.xmu_sum_matrix.values)  
    if debug: print("psio_norm_array ", ps3.psio_norm_array.values)
    ps3 = ps3.assign(cmatch_norm_matrix=(('r', 'r_prime'), np.sqrt(np.outer(ps3.cmatch.values, ps3.cmatch.values))))
    if debug: print("cmatch_norm_matrix ", ps3.cmatch_norm_matrix.values)
    ps3 = ps3.assign(norm_matrix=np.multiply(ps3.psio_norm_array,ps3.cmatch_norm_matrix)+0.0*ps3.cmatch_norm_matrix)
    if debug: print("norm_matrix ", ps3.norm_matrix.values)

    for i in range(len(varlista)): 
        tempvals1 = np.multiply(ps3[varlista[i][0]].values,ps3.norm_matrix.values)
        tempvals2 = np.multiply(ps3[varlista[i][1]].values,ps3.norm_matrix.values)
        if debug and i == 3:
            print("Delta prime values before rescaling: ", ps3[varlista[i][0]].values)
            print("Delta prime values after rescaling: ", tempvals1)
        temp3D = np.zeros((len(ps3.r.values), len(ps3.r_prime.values), 2))
        temp3D[:, :, 0] = tempvals1
        temp3D[:, :, 1] = tempvals2
        temp_DAa = xr.DataArray(temp3D, dims=('r', 'r_prime', 'i'))

        tempvalsb = np.multiply(ps3[varlistb[i]].values,ps3.norm_matrix.values)
        temp_DAb = xr.DataArray(tempvalsb, dims=('r', 'r_prime'))
        
        ps3[namelista[i]] = temp_DAa
        ps3[namelistb[i]] = temp_DAb

    return ps3

def GPEC_resistive_calculation(eq_filename, nn, run_rdcon=False, run_stride=False, 
        make_working_dir=True, 
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
    Run GPEC resistive calculation by calling the rdcon and stride fortran executables to run in a working directory of user choice.
    Prints files both to the working directory and to output_location if specified. Use output_prefix for customizing output filenames.
    
    Parameters:
    eq_filename (str): Path to the equilibrium file. Cannot be too long (strange old fortran...)
    nn (int): Toroidal mode number.
    run_rdcon (bool): If True, run rdcon.
    run_stride (bool): If True, run stride.
    make_working_dir (bool): If True, create a working directory.
    working_dir (str): Path to the working directory. Will be created if it does not exist and make_working_dir is True.
    gpec_dir (str): Path to the GPEC directory.
    verbose (bool): If True, print verbose output.
    fresh_start (bool): If True, remove existing calculations from the working directory.
    output_location (str): If specified, save the output files to this location.
    output_prefix (str): Prefix for the output files.
    save_input (bool): If True, save the input parameters used for rdcon and stride calculations to a netCDF file.
    save_terminal_output (bool): If True, save the terminal output to rdcon_terminal_output_n{nn}.txt and stride_terminal_output_n{nn}.txt in the working directory. If fresh_start is True, the files will be removed if they exist.
    kwargs: keyword arguments for write_rdcon_stride_inputs.

    Returns:
    rdcon_xr (xarray): Xarray containing the rdcon output.
    stride_xr (xarray): Xarray containing the stride output.
    rdcon_ran (bool): True if rdcon was run successfully.
    stride_ran (bool): True if stride was run successfully.
    rdcon_stride_input_dict (array): xarray containing all input parameters used for rdcon and stride calculations.
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

    rdcon_stride_input_dict = write_rdcon_stride_inputs(working_dir, eq_filename, nn=nn, run_stride=run_stride, run_rdcon=run_rdcon, fresh_start=fresh_start, **kwargs)

    rdcon_output_name = str(output_prefix + eq_filename + '_rdcon_n'+str(nn)+'.nc')
    stride_output_name = str(output_prefix + eq_filename + '_stride_n'+str(nn)+'.nc')

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
            rdcon_xr.to_netcdf(os.path.join(output_location, rdcon_output_name))
            if verbose: print(f"Saved rdcon output to {os.path.join(output_location, rdcon_output_name)}")
        if run_stride and stride_xr is not None:
            if override_save and os.path.isfile(os.path.join(output_location, stride_output_name)):
                os.remove(os.path.join(output_location, stride_output_name))
            stride_xr.to_netcdf(os.path.join(output_location, stride_output_name))
            if verbose: print(f"Saved stride output to {os.path.join(output_location, stride_output_name)}")
        if save_input:
            if override_save and os.path.isfile(os.path.join(output_location, output_prefix + eq_filename + '_rdcon_stride_input_n'+str(nn)+'.pkl')):
                os.remove(os.path.join(output_location, output_prefix + eq_filename + '_rdcon_stride_input_n'+str(nn)+'.pkl'))
            fpkl = open(os.path.join(output_location, output_prefix + eq_filename + '_rdcon_stride_input_n'+str(nn)+'.pkl'),"wb")
            pkl.dump(rdcon_stride_input_dict,fpkl)
            fpkl.close()
            if verbose: print(f"Saved rdcon and stride input to {os.path.join(output_location, output_prefix + eq_filename + '_rdcon_stride_input_n'+str(nn)+'.pkl')}")
        if save_terminal_output:
            if run_rdcon and os.path.exists(os.path.join(working_dir, 'rdcon_terminal_output_n'+str(nn)+'.txt')):
                shutil.copy(os.path.join(working_dir, 'rdcon_terminal_output_n'+str(nn)+'.txt'), output_location+ '/' + output_prefix + eq_filename + '_rdcon_terminal_output_n'+str(nn)+'.txt')
                if verbose: print(f"Saved rdcon terminal output to {os.path.join(output_location, output_prefix + eq_filename + '_rdcon_terminal_output_n'+str(nn)+'.txt')}")
            elif run_rdcon:
                #raise an error with FileNotFoundError
                raise FileNotFoundError(f"Rdcon terminal output file rdcon_terminal_output_n{nn}.txt not found in the working directory.")
            if run_stride and os.path.exists(os.path.join(working_dir, 'stride_terminal_output_n'+str(nn)+'.txt')):
                shutil.copy(os.path.join(working_dir, 'stride_terminal_output_n'+str(nn)+'.txt'), output_location+ '/' + output_prefix + eq_filename + '_stride_terminal_output_n'+str(nn)+'.txt')
                if verbose: print(f"Saved stride terminal output to {os.path.join(output_location, output_prefix + eq_filename + '_stride_terminal_output_n'+str(nn)+'.txt')}")
            elif run_stride:
                #raise an error with FileNotFoundError
                raise FileNotFoundError(f"Stride terminal output file stride_terminal_output_n{nn}.txt not found in the working directory.")

    return rdcon_xr, stride_xr, rdcon_ran, stride_ran, rdcon_stride_input_dict