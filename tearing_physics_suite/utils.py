# Utilities for tearing physics suite 

import math
import xarray as xr
import numpy as np

def trim_nans(delta_primes,delta_prime_errs=None):
    """ Cuts nans off the final rows/columns of a square delta_prime matrix. Assumes 
    the underlying matrix (no nans) is also square. If delta_prime_errs is also provided,
    this matrix will be trimmed of the same number of rows and columns as the delta_primes matrix. """
    assert delta_primes.shape[0] == delta_primes.shape[1], "Input matrix is not square."
    if not delta_prime_errs is None:
        assert delta_primes.shape[0] == delta_prime_errs.shape[0] == delta_prime_errs.shape[1], "Delta prime errors have different shape to Delta primes."
    dp_col1 = delta_primes[:,0]
    nans_in_col1=0
    for i in dp_col1:
        if np.isnan(i):
            nans_in_col1+=1
    dp_row1 = delta_primes[0,:]
    nans_in_row1=0
    for i in dp_row1:
        if np.isnan(i):
            nans_in_row1+=1
    assert nans_in_col1 == nans_in_row1, "Input matrix has inconsistent number of nans in rows and columns."
    #We cut the matrix to remove nans:
    if nans_in_col1 > 0:
        delta_primes = delta_primes[:-nans_in_col1,:-nans_in_row1]
        if not delta_prime_errs is None:
            delta_prime_errs = delta_prime_errs[:-nans_in_col1,:-nans_in_row1]
    # Now delta_primes is a square matrix with no nans. Check no nans present:
    if np.isnan(delta_primes).any():
        print(delta_primes)
        raise ValueError("Input matrix has interior nans present. Error somewhere in truncation")
    if not delta_prime_errs is None:
        if np.isnan(delta_prime_errs).any():
            print(delta_prime_errs)
            raise ValueError("Input matrix has interior nans present. Error somewhere in truncation")
    return delta_primes, nans_in_col1, delta_prime_errs

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
