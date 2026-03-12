# Python functions to call GPEC and PEST3 fortran codes for delta prime calculations

import os
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
import pickle as pkl
import copy

from tearing_physics_suite.environment import home_dir
from tearing_physics_suite.GPEC_write_inputs import write_rdcon_stride_inputs
from tearing_physics_suite.PEST3_wrappers import pest3_special_truncation_loop,pest3_clean_netcdf,PEST3_resistive_calculation
from tearing_physics_suite.delta_prime_extraction import extract_delta_primes

def run_resistive_calculation(eq_filename, nn, run_rdcon=True, run_stride=True, run_pest3=True,
        make_working_dir=True,
        make_results_dir=True,
        working_dir=os.path.join(home_dir, 'working_dir'),
        gpec_dir=os.path.join(home_dir, 'submodules/GPEC'), 
        pest3_dir=os.path.join(home_dir, 'submodules/PEST3/cmake_build/pest3'),
        verbose=True,
        fresh_start=True,
        output_location=None,
        output_prefix='',
        save_input=True,
        save_terminal_output=False, #Currently broken
        pest_match_truncation=True,
        override_save=True, 
        pest_pull_mtheta=True, # Change at your own risk, see mtheta_scan scan results
        debug_GPEC_resistive_calculation=False, #Quick exit after GPEC resistive calculation
        **kwargs):
    """
    Run resistive toroidal calculation for a single toroidal mode number by calling
    the GPEC and PEST3 fortran executables in a working directory.
    Prints files both to the working directory and to output_location if specified.

    Parameters
    ----------
    eq_filename : str
        Path to the equilibrium file. Cannot be too long (Fortran path length limitation).
    nn : int
        Toroidal mode number.
    run_rdcon, run_stride, run_pest3 : bool
        Whether to run each code.
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
    pest_match_truncation : bool
        Match PEST3 truncation to GPEC/STRIDE. Overridden by psihigh_pest kwarg.
    **kwargs
        Forwarded to write_rdcon_stride_inputs and PEST3_resistive_calculation.

    Returns
    -------
    rdcon_xr, stride_xr, pest3_xr : xr.Dataset or None
        Output xarrays from each code.
    rdcon_ran, stride_ran, pest3_ran : bool
        Whether each code ran successfully.
    rdcon_stride_input_dict, pest3_input_dict : dict or None
        Input parameters used for each calculation.
    """

    #Extract keyword arguments for PEST3
    pest3_kwargs_dict = {k: v for k, v in kwargs.items() if k.endswith('_pest')}
    if not override_save:
        pest3_kwargs_dict.update({"override_save":False})

    #Remove PEST3 specific keyword arguments from kwargs
    for key in pest3_kwargs_dict.keys():
        if key in kwargs:
            del kwargs[key]

    # Make equilibrium type consistent (default case is eqdsk):
    if 'eq_type' in kwargs:
        if kwargs['eq_type'] == """'ldp_i'""" or kwargs['eq_type'] == '''"ldp_i"''':
            pest3_kwargs_dict['eq_type_pest']=8

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
    q_rationals = None
    r = None
    r_prime = None

    # Define maximum poloidal fourier harmonic with the same logic as in GPEC:
    m_max=0
    m_maxs=0
    m_min=0
    m_mins=0
    delta_mhigh=rdcon_stride_input_dict['delta_mhigh']
    delta_mlow=rdcon_stride_input_dict['delta_mlow']
    num_rat_surfaces=0 

    # Pull truncation and poloidal mode information from GPEC calculations: 
    if (run_rdcon and rdcon_ran):
        m_max = int(np.ceil(rdcon_xr.qmax*nn+delta_mhigh))
        m_min = int(np.floor(min(rdcon_xr.qmin*nn,0)-4-delta_mlow))
        qlim_actual = max(rdcon_xr.qlim, qlim_actual)
        psilow_actual = min(rdcon_xr.psilow, psilow_actual)
        q_rationals = rdcon_xr.q_rational.values
        r = rdcon_xr.r
        r_prime = rdcon_xr.r_prime
        # Check if rdcon generated Delta_prime 
        if "Delta_prime" in rdcon_xr.data_vars:
            num_rat_surfaces=max(len(rdcon_xr.Delta_prime.isel(i=0,r_prime=0).values),num_rat_surfaces)
    if (run_stride and stride_ran):
        m_maxs = int(np.ceil(stride_xr.qmax*nn+delta_mhigh))
        m_mins = int(np.floor(min(stride_xr.qmin*nn,0)-4-delta_mlow))
        qlim_actuals = max(stride_xr.qlim, qlim_actuals)
        psilow_actuals = min(stride_xr.psilow, psilow_actuals)
        if "Delta_prime" in stride_xr.data_vars:
            num_rat_surfaces=max(len(stride_xr.Delta_prime.isel(i=0,r_prime=0).values),num_rat_surfaces)
        if q_rationals is None:
            q_rationals = stride_xr.q_rational.values
            r = stride_xr.r
            r_prime = stride_xr.r_prime
        else:
            #Check they are close in values:
            if len(q_rationals) == len(stride_xr.q_rational.values):
                if np.max(np.abs(q_rationals - stride_xr.q_rational.values)) > 1e-5:
                    raise ValueError("Rational surfaces from rdcon and stride differ. Cannot match truncation.")
            else:
                raise ValueError("Rational surfaces from rdcon and stride differ. Cannot match truncation.")
    m_max = max(m_max, m_maxs)
    m_min = min(m_min, m_mins)
    m_absmax = max(abs(m_max), abs(m_min))

    # Set truncation values:
    if (run_stride and stride_ran) and (run_rdcon and rdcon_ran):
        if 2*(abs(qlim_actual-qlim_actuals)/(abs(qlim_actual)+abs(qlim_actual))) > 1e-4:
            print("WARNING, rdcon and stride truncation is differing.")
            print("qlims: ", qlim_actual, qlim_actuals)
        if 2*(abs(psilow_actual-psilow_actuals)/(abs(psilow_actuals)+abs(psilow_actuals))) > 1e-4:
            print("WARNING, rdcon and stride truncation is differing.")
            print("psilows: ", psilow_actual,psilow_actuals)
        qlim_actual = min(qlim_actual, qlim_actuals)
        psilow_actual = max(psilow_actual,psilow_actuals)
    elif (run_stride and stride_ran):
        qlim_actual=qlim_actuals
        psilow_actual=psilow_actuals
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

    if ('mtheta_pest' not in pest3_kwargs_dict) and pest_pull_mtheta:
        pest3_kwargs_dict['mtheta_pest'] = rdcon_stride_input_dict['mtheta']

    if 'mpsi_pest' not in pest3_kwargs_dict:
        pest3_kwargs_dict['mpsi_pest'] = rdcon_stride_input_dict['mpsi']

    if 'rational_surface_control_pest' not in pest3_kwargs_dict:
        if num_rat_surfaces > 15:
            pest3_kwargs_dict['rational_surface_control_pest'] = '''-m"''' + 'x'*num_rat_surfaces + '''"''' # Compute Delta's for first num_rat_surfaces rational surfaces
        if num_rat_surfaces == 0 and nn*5 < 15: # Large n, don't forget to include all these rational surfaces
            pest3_kwargs_dict['rational_surface_control_pest'] = '''-m"''' + 'x'*nn*5 + '''"''' 

    #########################################################################################################
    # Run PEST3 resistive calculation:
    #########################################################################################################
    
    if run_pest3:
        pest3_trunc_ran = False
        if pest_match_truncation and ((run_rdcon and rdcon_ran) or (run_stride and stride_ran)) and allow_trunc_loop:
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

        if not ((run_rdcon and rdcon_ran) or (run_stride and stride_ran)):
            print("**************************************************************** WARNING **********************************************************************")
            print("PEST3 calculation is running without rdcon or stride results to match truncation to. Results may not be comparable to GPEC calculations.")
            print("**************************************************************** WARNING **********************************************************************")
            raise RuntimeError

        pest3_xr, pest3_ran, pest3_input_dict = PEST3_resistive_calculation(
            eq_filename=eq_filename, nn=nn, make_working_dir=make_working_dir,
            working_dir=working_dir, pest3_dir=pest3_dir, verbose=verbose,
            fresh_start=fresh_start, output_location=output_location,
            output_prefix=output_prefix, save_input=save_input,
            save_terminal_output=save_terminal_output, q_rationals=q_rationals, r=r, r_prime=r_prime, **pest3_kwargs_dict)

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

def compile_xarrays(rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict, calc_dps=True, **kwargs):
    """
    Combine rdcon, stride, and pest3 xarrays into a single dataset.

    Merges outputs from run_resistive_calculation into one xarray with a 'code'
    dimension. Optionally computes coupled Delta' values.

    Parameters
    ----------
    rdcon_xr, stride_xr, pest3_xr : xr.Dataset or None
        Per-code output datasets.
    rdcon_ran, stride_ran, pest3_ran : bool
        Whether each code ran successfully.
    rdcon_stride_input_dict, pest3_input_dict : dict or None
        Input parameter dictionaries.
    calc_dps : bool
        If True, compute coupled Delta' values via extract_delta_primes.

    Returns
    -------
    combined_xr : xr.Dataset or None
        Merged dataset with 'code' dimension.
    pest3_xr : xr.Dataset or None
        PEST3 dataset (returned separately if it couldn't be merged).
    input_dict : dict
        Combined input parameters.
    """
    # Combine input dictionaries:
    if not (rdcon_stride_input_dict is None): #RDCON dict present
        if not (pest3_input_dict is None): # PEST3 dict present
            rdcon_stride_input_dict.update(pest3_input_dict)
        input_dict = rdcon_stride_input_dict
    elif not (pest3_input_dict is None):
        input_dict = pest3_input_dict
    else:
        input_dict = {}
    
    
    # Combine xarrays:
    xarrays = []
    
    #########################################################################################################
    # RDCON delta xarray and delta prime calculation
    #########################################################################################################
    if not (rdcon_xr is None): 
        # Turn all attributes into variables:
        for attr_key in rdcon_xr.attrs.keys():
            rdcon_xr[attr_key] = rdcon_xr.attrs[attr_key]
        rdcon_xr.attrs = {}
        rdcon_xr = rdcon_xr.drop_attrs(deep=True)
        # Add new dimension for code to rdcon_xr
        rdcon_xr_expanded = rdcon_xr.expand_dims(dim='code', axis=0)
        rdcon_xr_expanded['code'] = ['rdcon']
        if calc_dps and 'Delta_prime' in rdcon_xr_expanded:
            rdcon_xr_expanded = extract_delta_primes(rdcon_xr_expanded)
        # Add to xarrays list
        xarrays.append(rdcon_xr_expanded)
    
    #########################################################################################################
    # STRIDE delta xarray and delta prime calculation
    #########################################################################################################
    if not (stride_xr is None):
        # Turn all attributes into variables:
        for attr_key in stride_xr.attrs.keys():
            stride_xr[attr_key] = stride_xr.attrs[attr_key]
        stride_xr.attrs = {}
        stride_xr = stride_xr.drop_attrs(deep=True)
        # Add new dimension for code to stride_xr
        stride_xr_expanded = stride_xr.expand_dims(dim='code', axis=0)
        stride_xr_expanded['code'] = ['stride']
        if calc_dps and 'Delta_prime' in stride_xr_expanded:
            # Calculate delta' values for stride_xr
            stride_xr_expanded = extract_delta_primes(stride_xr_expanded)
        xarrays.append(stride_xr_expanded)
    
    #########################################################################################################
    # PEST3 delta xarray and delta prime calculation
    #########################################################################################################
    pest3_xr_expanded = None
    if not (pest3_xr is None):
        # Turn all attributes into variables:
        for attr_key in pest3_xr.attrs.keys():
            pest3_xr[attr_key] = pest3_xr.attrs[attr_key]
        pest3_xr.attrs = {}
        pest3_xr = pest3_xr.drop_attrs(deep=True)
        # Add new dimension for code to pest3_xr
        pest3_xr_expanded = pest3_xr.expand_dims(dim='code', axis=0)
        pest3_xr_expanded['code'] = ['pest3']
        if calc_dps and 'Delta_prime' in pest3_xr_expanded:
            assert 'Delta_prime_perr' in pest3_xr_expanded, "Current version of extract_delta_primes assumes this."
            pest3_xr_expanded = extract_delta_primes(pest3_xr_expanded)
        xarrays.append(pest3_xr_expanded)

    # Combine all xarrays into one xarray:
    # Breaks if different number of rational surfaces across different codes at the axis
    #   - beware psilow =/= 0 while also running pest3 (pest3 has no psilow truncation)
    #   - for this reason, we also output pest3_xr_out separately if something goes wrong
    pest3_xr_out = None
    combined_xr = None

    #########################################################################################################
    # Concatenating xarrays
    #########################################################################################################
    if len(xarrays) > 0:
        pest3_xr_out = pest3_xr_expanded
        try:
            combined_xr = xr.concat(xarrays, dim='code', coords='all', **kwargs)
            pest3_xr_out = None
        except Exception as e:
            if not (pest3_xr is None): #We remove pest3_xr_expanded from xarrays and retry
                xarrays = xarrays[:-1]  # Remove the last element (pest3_xr_expanded)
                combined_xr = xr.concat(xarrays, dim='code', coords='all', **kwargs)
            print("Error combining xarrays:", e)

    #########################################################################################################
    # Adding nn to combined_xr:
    #########################################################################################################
    if combined_xr is not None:
        # Check nn isn't already defined:
        assert not 'nn' in combined_xr, 'nn already defined, debug this function.'
        # We expand dims to add nn:
        combined_xr = combined_xr.expand_dims(dim='nn', axis=0)
        combined_xr['nn'] = [input_dict['nn']]

    return combined_xr, pest3_xr_out, input_dict


