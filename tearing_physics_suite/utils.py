# Utilities for tearing physics suite 

import math
import xarray as xr
from tearing_physics_suite.delta_prime_extraction import delta_primes


def compile_xarrays(rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict,calc_dps=True,**kwargs):
    """
    Combine xarrays and input dictionaries from rdcon, stride, and pest3 xarrays (outputs of fortran_wrappers.run_resistive_calculation).
    We also add delta' values to the xarrays if requested, using the delta_primes function.
    
    Parameters:
    rdcon_xr: xarray from rdcon, output from fortran_wrappers.run_resistive_calculation
    stride_xr: xarray from stride, output from fortran_wrappers.run_resistive_calculation
    pest3_xr: xarray from pest3, output from fortran_wrappers.run_resistive_calculation
    rdcon_ran: Boolean indicating if rdcon ran successfully
    stride_ran: Boolean indicating if stride ran successfully
    pest3_ran: Boolean indicating if pest3 ran successfully
    rdcon_stride_input_dict: Input dictionary from rdcon and stride
    pest3_input_dict: Input dictionary from pest3
    calc_dps: Boolean indicating if coupled delta' values should be calculated

    Returns:
    combined_xr: Combined xarray with all data from rdcon, stride, and pest3
    input_dict: Combined input dictionary with all parameters from rdcon, stride, and pest3
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
        # Add new dimension for code to rdcon_xr
        rdcon_xr_expanded = rdcon_xr.expand_dims(dim='code', axis=0)
        rdcon_xr_expanded['code'] = ['rdcon']
        if calc_dps and 'Delta_prime' in rdcon_xr_expanded:
            # Calculate delta' values for rdcon_xr
            dp_sh, dp_eff, dp_nn_eff, dp_2nn_eff, divisors = delta_primes(rdcon_xr_expanded['Delta_prime'].sel(i=0).values)
            rdcon_xr_expanded = rdcon_xr_expanded.assign(dprim_single_helicity=dp_sh + 0.0 * rdcon_xr_expanded['psi_n_rational'])
            rdcon_xr_expanded = rdcon_xr_expanded.assign(dprim_eff=dp_eff + 0.0 * rdcon_xr_expanded['psi_n_rational'])
            rdcon_xr_expanded = rdcon_xr_expanded.assign(dprim_nn_eff=dp_nn_eff + 0.0 * rdcon_xr_expanded['psi_n_rational'])
            rdcon_xr_expanded = rdcon_xr_expanded.assign(dprim_2nn_eff=dp_2nn_eff + 0.0 * rdcon_xr_expanded['psi_n_rational'])
        # Add to xarrays list
        xarrays.append(rdcon_xr_expanded)
    
    #########################################################################################################
    # STRIDE delta xarray and delta prime calculation
    #########################################################################################################
    if not (stride_xr is None):
        # Add new dimension for code to stride_xr
        stride_xr_expanded = stride_xr.expand_dims(dim='code', axis=0)
        stride_xr_expanded['code'] = ['stride']
        if calc_dps and 'Delta_prime' in stride_xr_expanded:
            # Calculate delta' values for stride_xr
            dp_sh, dp_eff, dp_nn_eff, dp_2nn_eff, divisors = delta_primes(stride_xr_expanded['Delta_prime'].sel(i=0).values)
            stride_xr_expanded = stride_xr_expanded.assign(dprim_single_helicity=dp_sh + 0.0 * stride_xr_expanded['psi_n_rational'])
            stride_xr_expanded = stride_xr_expanded.assign(dprim_eff=dp_eff + 0.0 * stride_xr_expanded['psi_n_rational'])
            stride_xr_expanded = stride_xr_expanded.assign(dprim_nn_eff=dp_nn_eff + 0.0 * stride_xr_expanded['psi_n_rational'])
            stride_xr_expanded = stride_xr_expanded.assign(dprim_2nn_eff=dp_2nn_eff + 0.0 * stride_xr_expanded['psi_n_rational'])
        xarrays.append(stride_xr_expanded)
    
    #########################################################################################################
    # PEST3 delta xarray and delta prime calculation
    #########################################################################################################
    pest3_xr_expanded = None
    if not (pest3_xr is None):
        # Add new dimension for code to pest3_xr
        pest3_xr_expanded = pest3_xr.expand_dims(dim='code', axis=0)
        pest3_xr_expanded['code'] = ['pest3']
        if calc_dps and 'Delta_prime' in pest3_xr_expanded:
            # Calculate delta' values for pest3_xr
            dp_sh, dp_eff, dp_nn_eff, dp_2nn_eff, divisors = delta_primes(pest3_xr_expanded['Delta_prime'].isel(i=0).values)
            pest3_xr_expanded = pest3_xr_expanded.assign(dprim_single_helicity=dp_sh + 0.0 * pest3_xr_expanded['cmatch'])
            pest3_xr_expanded = pest3_xr_expanded.assign(dprim_eff=dp_eff + 0.0 * pest3_xr_expanded['cmatch'])
            pest3_xr_expanded = pest3_xr_expanded.assign(dprim_nn_eff=dp_nn_eff + 0.0 * pest3_xr_expanded['cmatch'])
            pest3_xr_expanded = pest3_xr_expanded.assign(dprim_2nn_eff=dp_2nn_eff + 0.0 * pest3_xr_expanded['cmatch'])
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

    return combined_xr, pest3_xr_out, input_dict

def create_dense_log_paramvals(start=1e-1, end=1e-7, points_per_decade=10, extra_density_regions=None):
    """
    Create densely packed parameter values in logarithmic scale.
    
    Parameters:
    start: Starting value (1e-1)
    end: Ending value (1e-7)
    points_per_decade: Base number of points per decade
    extra_density_regions: List of tuples (min_val, max_val, extra_points) for regions needing more density <- MIGHT BE BROKEN
    
    Returns:
    list of parameter values in descending order
    """
    
    # Calculate the log range
    log_start = math.log10(start)
    log_end = math.log10(end)
    
    # Base logarithmic spacing
    num_decades = abs(log_start - log_end)
    base_points = int(num_decades * points_per_decade) + 1
    
    # Create base logarithmic array using list comprehension
    paramvals_base = []
    for i in range(base_points):
        log_val = log_start + (log_end - log_start) * i / (base_points - 1)
        paramvals_base.append(10**log_val)
    
    # Add extra density in specific regions if specified
    extra_vals = []
    if extra_density_regions:
        for min_val, max_val, extra_points in extra_density_regions:
            log_min = math.log10(min_val)
            log_max = math.log10(max_val)
            for i in range(extra_points):
                log_val = log_min + (log_max - log_min) * i / (extra_points - 1)
                extra_vals.append(10**log_val)
    
    # Combine all values and remove duplicates
    all_vals = paramvals_base + extra_vals
    paramvals_unique = list(set(all_vals))
    
    # Sort in descending order (from 1e-1 to 1e-7)
    paramvals_sorted = sorted(paramvals_unique, reverse=True)
    
    return paramvals_sorted
