# Utilities for running PEST3 and reading the results

import os 
import sys
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import Akima1DInterpolator
import pickle as pkl
import copy
import math

home_dir = os.environ['TPSHOME']


def PEST3_resistive_calculation(eq_filename, nn, make_working_dir=True,make_results_dir=True,
        working_dir=os.path.join(home_dir, 'working_dir'),
        pest3_dir=os.path.join(home_dir, 'submodules/PEST3/cmake_build/pest3'),
        verbose=True,
        debug=False,
        fresh_start=True,
        output_location=None,
        output_prefix='',
        save_input=True,
        save_terminal_output=False,
        override_save=True,
        q_rationals=None,
        r=None,
        r_prime=None,
        clean_netcdf=True, #Always keep on unless running truncation loop
        # PEST3 specific parameters:
        eq_type_pest=3,             # 3 = efit eqdsk, see pest3_dir/pest3.hh for other options
        kband_pest=14,              # Poloidal Fourier modes span -|kband_pest|,...+|kband_pest|
        rational_surface_control_pest='''-m"xxxxxxxxxxxxxxx"''', # String to control the computation of rational surfaces. Use 'xxxxxxxxxx' to compute Delta's for first 10 rational surfaces. Use '.' symbol to skip a rational surface.
        psilow_pest=1e-4,
        psihigh_pest=0.995,         # This behaves differently from the GPEC psihigh
        a_wall_pest=20,             # Distance of the conformal ideal wall from the plasma in units of minor radius. a_wall_pest > 10 <=> wall at infinity, a_wall_pest = 0 <=> internal mode only. See pest3.hh for more details. 
        mtheta_pest=129, #Must be odd # Number of poloidal rays for eqdsk mapping. Large values (~800) likely introduce numerical instabilities.
        mpsi_pest=400,                   # Number of radial grid intervals for equilibrium quantities for eqdsk mapping. Large values (~800) likely introduce numerical instabilities.
        nx_string_pest='''-k"100 50 80 140"''', # [Higher is not better!!! see pest3_finite_element_scan for more info] String for the number of radial finite elements per non-singular interval. Convergence should obey nx^(-2) going to zero, hence multiple values are specified. Use nxpest for a single value. Use even numbers!
        nx_pest=0,                  # Must be even! Number of radial finite elements per non-singular interval.
        large_sol_extent_pest=0.9, # See pest3_dir/pest3.hh for details.
        solver_pest=0,             # String for the PEST3 solver options, see pest3_dir/pest3.hh for details.
        extra_input_string_pest='',      # Extra input string to pass to PEST3, e.g. for additional control parameters. See pest3_dir/pest3.hh for details.
        ):
    """
    Run PEST3 resistive calculation by calling the PEST3 Fortran executable.

    For a full list of keyword arguments, see pest3.hh in pest3_dir,
    or run ./pest3x -h in pest3_dir.

    Parameters
    ----------
    eq_filename : str
        Path to the equilibrium file.
    nn : int
        Toroidal mode number.
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
    q_rationals : np.ndarray or None
        Rational surface q values (from GPEC) for dimension alignment.
    r, r_prime : np.ndarray or None
        Surface label coordinates (from GPEC) for dimension alignment.
    **PEST3-specific kwargs**
        See inline comments for eq_type_pest, kband_pest, psihigh_pest, etc.

    Returns
    -------
    pest3_xr : xr.Dataset or None
        PEST3 output dataset.
    pest3_ran : bool
        Whether PEST3 ran successfully.
    pest3_input_dict : dict
        Input parameters used for the calculation.
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
    if make_results_dir:
        if output_location is not None:
            if not os.path.exists(output_location):
                os.makedirs(output_location)
                if verbose: print(f"Created output directory: {output_location}")
            else:
                if verbose: print(f"Output directory {output_location} already exists. Using existing directory.")
        else:
            output_location = working_dir

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
        #Check if nx_pest is odd, if so, raise by 1
        if nx_pest % 2 != 0:
            if verbose: print(f"nx_pest {nx_pest} is odd, raising by 1 to {nx_pest + 1}")
            nx_pest += 1
        nx_string_pest = ' -k' + str(nx_pest)

    if mtheta_pest % 2 == 0:
        if verbose: print(f"mtheta_pest {mtheta_pest} is even, raising by 1 to {mtheta_pest + 1}")
        mtheta_pest += 1

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

        if clean_netcdf:
            pest3_xr = pest3_clean_netcdf(ps3,debug=debug,q_rationals=q_rationals, r=r, r_prime=r_prime)
        else:
            pest3_xr = ps3
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
            pest3_output_name = str(output_prefix + eq_filename.split(".")[0] + '_pest3_n'+str(nn)+'.nc')
            if override_save and os.path.isfile(os.path.join(output_location, pest3_output_name)):
                os.remove(os.path.join(output_location, pest3_output_name))
            pest3_xr.to_netcdf(os.path.join(output_location, pest3_output_name), engine="scipy")
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
    Find psihigh for PEST3 that matches the GPEC q-truncation point, using bisection.

    Runs fast, low-resolution PEST3 calculations to iteratively narrow the
    psihigh_pest value until the output q-boundary matches qlim_actual.

    Parameters
    ----------
    eq_filename : str
        Path to the equilibrium file.
    nn : int
        Toroidal mode number.
    qlim_actual : float
        Target q limit from GPEC/STRIDE calculations.
    pest3_kwargs_dict : dict
        PEST3 keyword arguments.
    truncimax : int
        Maximum bisection iterations.

    Returns
    -------
    psihigh : float
        psihigh_pest value matching qlim_actual.
    success : bool
        Whether the truncation loop converged.
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
    pest3_kwargs_dict_local['rational_surface_control_pest'] = '''-m"x"'''

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
        clean_netcdf=False,
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
            clean_netcdf=False,
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
    """Find psihigh for PEST3 using a single high-resolution spline interpolation.

    Runs one PEST3 calculation at full psi range and interpolates to find the
    psihigh value where q matches qlim_actual. Used inside pest3_special_truncation_loop.

    Parameters
    ----------
    eq_filename : str
        Path to the equilibrium file.
    nn : int
        Toroidal mode number.
    qlim_actual : float
        Target q limit from GPEC/STRIDE calculations.
    pest3_kwargs_dict : dict
        PEST3 keyword arguments.

    Returns
    -------
    psihigh : float
        psihigh_pest value matching qlim_actual.
    success : bool
        Whether PEST3 ran and interpolation succeeded.
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
        clean_netcdf=False,
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
    cs = Akima1DInterpolator(pest3_xr['psinew'].values, (pest3_xr['qa'].values-qlim_actual),extrapolate=False)
    roots = cs.roots()

    # Check if the cubic spline has roots:
    if len(roots) == 0:
        print("No roots found in the cubic spline. Cannot determine truncation point.")
        return 0, False

    if len(roots) > 1:
        print("WARNING: Multiple roots found for the pest3 q truncation point. Using the smallest one that is greater than 0.9")
        print("         Roots found = ", roots)
        roots = [r for r in roots if r > 0.9]

    psi_trunc_val = roots[0]

    #Convert psi_trunc_val to a fraction of pest3_xr['psinew'].values[-1]-pest3_xr['psinew'].values[0]:
    psi_trunc_frac = (psi_trunc_val - pest3_xr['psinew'].values[0]) / (pest3_xr['psinew'].values[-1] - pest3_xr['psinew'].values[0])

    if debug:
        print(" Psi norm values: ", pest3_xr['psinew'].values/(pest3_xr['psinew'].values[-1] - pest3_xr['psinew'].values[0]))
        print(" q values: ", pest3_xr['qa'].values)

    if debug:
        return psi_trunc_frac, pest3_trunc_ran, pest3_xr['qa'].values, (pest3_xr['psinew'].values/(pest3_xr['psinew'].values[-1] - pest3_xr['psinew'].values[0]))
    return psi_trunc_frac, pest3_trunc_ran

def pest3_clean_netcdf(ps3, debug=True, drop_soln_info=True, q_rationals=None, r=None, r_prime=None):
    """Standardize PEST3 NetCDF output dimensions, variables, and unit conventions.

    Renames dimensions, rescales Delta' values to GPEC units via
    pest3_rescale_deltaprimes, and optionally aligns surface labels with
    GPEC rational surfaces.

    Parameters
    ----------
    ps3 : xr.Dataset
        Raw PEST3 output dataset.
    debug : bool
        Print debug information.
    drop_soln_info : bool
        Drop full-solution variables (x1frbo_re, etc.) to avoid dimension conflicts.
    q_rationals : np.ndarray or None
        GPEC rational surface q values for alignment.
    r, r_prime : np.ndarray or None
        GPEC surface label coordinates for alignment.

    Returns
    -------
    xr.Dataset
        Processed dataset with standardized dimensions and rescaled Delta' values.
    """

    #########################################################################################################
    # Standardize dimensions and variable names in PEST3 output:
    #########################################################################################################
    #Names of dims:
    missing_m_from_gpec = []
    missing_m_from_pest = []
    if len(ps3.cmatch.dims) > 0:
        surfdim=ps3.cmatch.dims[0]
        unknowndim=ps3.x1frbo_re.dims[1]
        if q_rationals is not None:
            # We convert q_rationals to m values:
            m_gpec = np.round(np.squeeze(q_rationals*ps3.n.values))
            m_pest = np.round(np.squeeze(ps3.qslay.values*ps3.n.values))
            
            #Check no values in m_pest are missing from m_gpec:
            missing_m_from_gpec = [m for m in m_pest if m not in m_gpec]
            missing_m_from_pest = [m for m in m_gpec if m not in m_pest]

            assert len(missing_m_from_gpec) == 0, "Some m values from GPEC are not in PEST3 output. Check truncation logic."
        
            if len(missing_m_from_pest) > 0:
                assert min(missing_m_from_pest) > max(m_pest), "Some interior m values from PEST3 are not in GPEC output. Check truncation logic."            
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
            if (dim_i == surfdim) and ('r_temp' in new_dims):
                new_dims.append('r_prime_temp')
            elif dim_i == surfdim:
                new_dims.append('r_temp')
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

    for varname, da in ps3.data_vars.items():
        if drop_soln_info:
            # Drop variables with more than 3 dimensions (e.g. x1frbo_re, x1frbo_im):
            if len(da.dims) > 2:
                if debug: print(f"Dropping variable {varname} with {len(da.dims)} dimensions.")
                ps3 = ps3.drop_vars(varname)
            elif len(da.dims) == 2 and 'ukn_dim' in da.dims:
                if debug: print(f"Dropping variable {varname} with {len(da.dims)} dimensions.")
                ps3 = ps3.drop_vars(varname)
            elif len(da.dims) == 2 and not ('r_temp' in da.dims and 'r_prime_temp' in da.dims):
                if debug: print(f"Dropping variable {varname} with {len(da.dims)} dimensions.")
                ps3 = ps3.drop_vars(varname)

    assert 'r' not in ps3.data_vars, "PEST3 output already has a variable named 'r'. Check PEST3 output and cleaning logic."
    assert 'r_prime' not in ps3.data_vars, "PEST3 output already has a variable named 'r_prime'. Check PEST3 output and cleaning logic."
    if 'r_temp' not in ps3.dims:
        print("PEST3 data variables: ", ps3.dims)
        raise ValueError("PEST3 output does not have a variable named 'r_temp'. Check PEST3 output and cleaning logic.")
    
    for varname, da in ps3.data_vars.items():
        if len(missing_m_from_pest) > 0:
            # We expand ps3[varname] such that ps3[varname].r matches input DataArray r:
            if 'r_temp' in da.dims and 'r_prime_temp' not in da.dims:
                tempvals = da.values
                # Add extra nans to the end:
                tempvals_new = np.full((len(r)), np.nan)
                tempvals_new[:len(da.r_temp)] = tempvals
                temp_da = xr.DataArray(tempvals_new, dims=('r'))
                # Set the coordinates of temp_da to match r:
                temp_da.coords['r'] = r
                # Replace da with temp_da in ps3:
                ps3 = ps3.drop_vars(varname)
                ps3[varname] = temp_da
            elif 'r_temp' in da.dims and 'r_prime_temp' in da.dims:
                # We expand ps3[varname] such that ps3[varname].r matches input DataArray r and ps3[varname].r_prime matches input DataArray r_prime:
                tempvals = da.values
                # Add extra nans to the end:
                tempvals_new = np.full((len(r), len(r_prime)), np.nan)
                tempvals_new[:len(da.r_temp), :len(da.r_prime_temp)] = tempvals
                temp_da = xr.DataArray(tempvals_new, dims=('r', 'r_prime'))
                # Set the coordinates of temp_da to match r and r_prime:
                temp_da.coords['r'] = r
                temp_da.coords['r_prime'] = r_prime
                # Replace da with temp_da in ps3:
                ps3 = ps3.drop_vars(varname)
                ps3[varname] = temp_da
        else:
            # Just rename 'r_temp' to 'r' and 'r_prime_temp' to 'r_prime':
            if 'r_temp' in da.dims and 'r_prime_temp' not in da.dims:
                tempvals = da.values
                temp_da = xr.DataArray(tempvals, dims=('r'))
                # Set the coordinates of temp_da to match r:
                temp_da.coords['r'] = r
                # Replace da with temp_da in ps3:
                ps3 = ps3.drop_vars(varname)
                ps3[varname] = temp_da
            elif 'r_temp' in da.dims and 'r_prime_temp' in da.dims:
                tempvals = da.values
                temp_da = xr.DataArray(tempvals, dims=('r', 'r_prime'))
                # Set the coordinates of temp_da to match r and r_prime:
                temp_da.coords['r'] = r
                temp_da.coords['r_prime'] = r_prime
                # Replace da with temp_da in ps3:
                ps3 = ps3.drop_vars(varname)
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


