# Python functions to scan input parameters during the resistive calculation, testing numerical and physics sensitivities

import pandas as pd
import xarray as xr
import numpy as np
import os

home_dir = os.environ['TPSHOME']
import tearing_physics_suite.fortran_wrappers as tfw
from tearing_physics_suite.fortran_wrappers import compile_xarrays
from tearing_physics_suite.tearing_physics_suite import delta_prime_variability

def scan_1D_input(input_name,input_values,eq_filename,
        nn,
        debug=False,
        abs_threshold=0.05,
        rel_threshold=0.05,
        **kwargs):
    """Run a 1D parameter scan of the resistive calculation.

    Parameters
    ----------
    input_name : str
        Name of the input parameter to scan.
    input_values : list
        Values to scan over.
    eq_filename : str
        Path to the equilibrium file.
    nn : int
        Toroidal mode number.
    debug : bool
        If True, return raw results list before post-processing.
    abs_threshold : float
        Absolute threshold for Delta' variation check.
    rel_threshold : float
        Relative threshold for Delta' variation check.
    **kwargs
        Forwarded to run_resistive_calculation.

    Returns
    -------
    tuple
        Output of extract_scanned_xrs: (xarrays, pest3_xarrays, input_dicts,
        input_values, input_name, message, deltaprimes).
    """
    #########################################################################################################
    # Check input name isn't in kwargs
    #########################################################################################################

    if input_name in kwargs:
        kwargs.pop(input_name)
    if not 'output_prefix' in kwargs:
        kwargs["output_prefix"] = ''
        output_prefix_tmp = ''
    else:
        output_prefix_tmp = kwargs["output_prefix"]
    
    results = []
    casenum=1

    for value in input_values:
    #########################################################################################################
    # Add input name and value to output_prefix
    #########################################################################################################
        kwargs["output_prefix"] = output_prefix_tmp
        if not (input_name == 'nx_string_pest' or input_name == 'jac_type' or input_name == 'grid_type' or input_name == 'solver'):
            kwargs["output_prefix"] += f"{input_name}_{value}"
            print(f"Running scan for {input_name} = {value}")
        elif input_name == 'nx_string_pest':
            kwargs["output_prefix"] += f"{input_name}_case{casenum}"
            print(f"Running scan for {input_name} = case{casenum}")
            casenum += 1
        elif input_name == 'jac_type' or input_name == 'grid_type' or input_name == 'solver':
            #Remove instances of '"""' from value
            valuetemp = value.replace('"""', '')
            #Replace instances of "'" with "_"
            valuetemp = valuetemp.replace("'", "_")
            kwargs["output_prefix"] += f"{input_name}_{valuetemp}"
            print(f"Running scan for {input_name} = {valuetemp}")

    #########################################################################################################
    # Compute result
    #########################################################################################################
        result = tfw.run_resistive_calculation(
            eq_filename,
            nn,
            **{input_name: value},
            **kwargs
        )
        results.append(result)

    if debug:
        return results, input_name, abs_threshold, rel_threshold

    return extract_scanned_xrs(results, input_name, abs_threshold, rel_threshold, output_prefix_tmp)

def extract_scanned_xrs(results, input_name, abs_threshold, rel_threshold, output_prefix_tmp):
    """Post-process raw scan results into structured outputs.

    Parameters
    ----------
    results : list of tuple
        Raw outputs from run_resistive_calculation, one per scan value.
    input_name : str
        Name of the scanned input parameter.
    abs_threshold : float
        Absolute threshold for Delta' variation check.
    rel_threshold : float
        Relative threshold for Delta' variation check.

    Returns
    -------
    xarrays : list of xr.Dataset
        Combined xarray per scan value.
    pest3_xarrays : list of xr.Dataset
        PEST3-specific xarray per scan value.
    input_dicts : list of dict
        Input parameters for each run.
    input_values : list
        Scanned parameter values.
    input_name : str
        Name of the scanned parameter.
    message : str
        Summary of Delta' results and q-surface information.
    deltaprimes : list of list
        First-surface single-helicity Delta' for each code and scan value.
    """

    xarrays = []
    pest3_xarrays = []
    input_dicts = []
    input_values = []

    #########################################################################################################
    # Simple loop to extract xarrays and input_dicts from results
    #########################################################################################################

    for result in results:
        combined_xr, pest3_xr,input_dict = compile_xarrays(*result)
        assert input_name in input_dict, f"Debug scan_1D_input: {input_name} not found in input_dict, available keys: {input_dict.keys()}"
        input_value = input_dict[input_name]
        input_values.append(input_value)
        xarrays.append(combined_xr)
        pest3_xarrays.append(pest3_xr)
        input_dicts.append(input_dict)

    #########################################################################################################
    # Build message, and delta primes (first surface, single helicity only)
    #########################################################################################################

    deltaprimes = []
    if output_prefix_tmp != '':
        message = str(" "+input_name+"("+output_prefix_tmp+"): "+str(input_values)+'\n')
    else:
        message = str(" "+input_name+": "+str(input_values)+'\n')
    xarrays_msg = []
    for i in range(len(xarrays)):
        if xarrays[i] is not None:  
            if 'Delta_prime' in xarrays[i]:
                xarrays_msg.append(xarrays[i])
    if 'rdcon' in xarrays_msg[0].code:
        if not np.isnan(xarrays_msg[0].Delta_prime.sel(code='rdcon',i=0).isel(r=0,r_prime=0)):
            message+=str("   RDCON delta prime:"+str([xarrays_msg[i].Delta_prime.sel(code='rdcon',i=0).isel(r=0,r_prime=0).values for i in range(len(xarrays_msg))])+'\n')
            deltaprimes.append([xarrays_msg[i].Delta_prime.sel(code='rdcon',i=0).isel(r=0,r_prime=0).values for i in range(len(xarrays_msg))])
    if 'stride' in xarrays_msg[0].code:
        if not np.isnan(xarrays_msg[0].Delta_prime.sel(code='stride',i=0).isel(r=0,r_prime=0)):
            message+=str("   STRIDE delta prime:"+str([xarrays_msg[i].Delta_prime.sel(code='stride',i=0).isel(r=0,r_prime=0).values for i in range(len(xarrays_msg))])+'\n')
            deltaprimes.append([xarrays_msg[i].Delta_prime.sel(code='stride',i=0).isel(r=0,r_prime=0).values for i in range(len(xarrays_msg))])
    if 'pest3' in xarrays_msg[0].code:
        if not np.isnan(xarrays_msg[0].Delta_prime.sel(code='pest3',i=0).isel(r=0,r_prime=0)):
            message+=str("   PEST3 delta prime:"+str([xarrays_msg[i].Delta_prime.sel(code='pest3',i=0).isel(r=0,r_prime=0).values for i in range(len(xarrays_msg))])+'\n')
            deltaprimes.append([xarrays_msg[i].Delta_prime.sel(code='pest3',i=0).isel(r=0,r_prime=0).values for i in range(len(xarrays_msg))])

    #########################################################################################################
    # Add q-surface info to the message:
    #########################################################################################################

    # Check if all values in a 1D array are the same [xarrays_msg[i].r.values[0] for i in range(len(xarrays_msg))] are the same
    if np.all(np.array([xarrays_msg[i].r.values[0] for i in range(len(xarrays_msg))]) == xarrays_msg[0].r.values[0]): 
        message+=str("   at q-surface "+ str(xarrays_msg[0].r.values[0])+"\n")
    else:
        message+=str("   q-surfaces:"+ str([xarrays_msg[i].r.values[0] for i in range(len(xarrays_msg))])+"\n")

    #########################################################################################################
    # Use delta_prime_variability to formalise result:
    #########################################################################################################
    DP_surf_xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere = delta_prime_variability_runner(xarrays, input_values, input_name, message, abs_threshold, rel_threshold)

    message+=str(" "+f"Delta' relative diff across core modes > {rel_threshold*100}%:  {rel_thresh_exceeded_psi95_anywhere.values} for {rel_thresh_exceeded_psi95_anywhere.code.values}\n")

    results_dict = {
        'xarrays': xarrays,
        'pest3_xarrays': pest3_xarrays,
        'input_dicts': input_dicts,
        'input_values': input_values,
        'input_name': input_name,
        'deltaprimes': deltaprimes,
        'DP_surf_xarray': DP_surf_xarray,
        'abs_thresh_exceeded_anywhere': abs_thresh_exceeded_anywhere,
        'rel_thresh_exceeded_anywhere': rel_thresh_exceeded_anywhere,
        'abs_thresh_exceeded_psi95_anywhere': abs_thresh_exceeded_psi95_anywhere,
        'rel_thresh_exceeded_psi95_anywhere': rel_thresh_exceeded_psi95_anywhere,
        'message': message,
    }

    return results_dict, message


def delta_prime_variability_runner(xarrays, input_values, input_name, message, abs_threshold, rel_threshold):
    """
    Function to run delta_prime_variability, to check whether Delta' changed a lot across the input scan.

    Parameters
    ----------
    xarrays : list of xr.Dataset
        List of xarrays containing Delta_prime_surf for each scan value.
    input_values : list
        List of scanned parameter values corresponding to each xarray.
    input_name : str
        Name of the scanned parameter, used for labeling the output xarray.
    message : str
        Message to be printed during processing.
    abs_threshold : float
        Absolute threshold for delta prime variability check.
    rel_threshold : float
        Relative threshold for delta prime variability check.

    Returns
    -------
    tuple
        Output of delta_prime_variability: (xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere)

    """

    #########################################################################################################
    # Compile delta prime xarrays into one xarray with an added coordinate for the scanned parameter.
    #########################################################################################################
    delta_prime_xarray = delta_prime_compiler(xarrays, input_values, input_name)

    #########################################################################################################
    # Run delta_prime_variability  
    #########################################################################################################

    xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere = delta_prime_variability(delta_prime_xarray, comparison_var = input_name, run_bool_check=True, abs_threshold=abs_threshold, rel_threshold=rel_threshold)
    
    return xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere

def delta_prime_compiler(xarray_list, input_values, input_name, run_concatenation=True):
    """
    Function takes a list of xarrays (one per scan value), extracts Delta_prime_surf and psi_n_rational from each, and compiles them into a single xarray for comparison. 

    Parameters
    ----------
    xarray_list : list of xr.Dataset
        List of xarrays containing Delta_prime_surf, psi_n_rational for each scan value.
    input_values : list
        List of scanned parameter values corresponding to each xarray.
    input_name : str
        Name of the scanned parameter, used for labeling the output xarray.

    Returns
    -------
    xr.Dataset
        Compiled xarray containing Delta_prime_surf for all scan values, with an added coordinate for the scanned parameter.
    """

    #########################################################################################################
    # Extract Delta_prime_surf from each xarray, and add a coordinate for the scanned parameter.
    #########################################################################################################

    delta_prime_arrays = []
    psi_n_arrays = []
    for i, xarr in enumerate(xarray_list):
        if xarr is not None and 'Delta_prime_surf' in xarr:
            dp_surf = xarr['Delta_prime_surf']
            dp_surf = dp_surf.expand_dims({input_name: [input_values[i]]})
            delta_prime_arrays.append(dp_surf)
            if 'psi_n_rational' in xarr:
                psi_n = xarr['psi_n_rational']
                psi_n = psi_n.expand_dims({input_name: [input_values[i]]})
                psi_n_arrays.append(psi_n)
        else:
            print(f"Warning: 'Delta_prime_surf' not found in xarray for scan value {input_values[i]}. Skipping this value.")
    if len(delta_prime_arrays) == 0:
        raise ValueError("No valid 'Delta_prime_surf' found in any xarrays. Cannot compile results.")

    #########################################################################################################
    # Check each delta_prime_array has the same dimensions and coordinates (except for the scanned parameter)
    #########################################################################################################
    reference_coords = delta_prime_arrays[0].drop_vars(input_name).coords
    same_coords = True
    for i, dp_array in enumerate(delta_prime_arrays):
        dp_coords = dp_array.drop_vars(input_name).coords
        if not dp_coords.equals(reference_coords):
            print(f"Warning: Coordinates of 'Delta_prime_surf' in xarray for scan value {input_values[i]} do not match reference coordinates. Skipping this value.")
            same_coords = False

    #########################################################################################################
    # Trim edge r, r_prime values if they are nans across all scan values, to allow concatenation. 
    #########################################################################################################
    if not same_coords and run_concatenation:
        print("Warning: Some xarrays have different coordinates. Trimming edge values.")
        # Find minimum and maximum r values across all delta_prime_arrays, ignoring nans:
        min_r = -np.inf
        max_r = np.inf
        for dp_array in delta_prime_arrays:
            if dp_array is not None:
                r_values = dp_array.coords['r'].values
                min_r = max(min_r, np.nanmin(r_values))
                max_r = min(max_r, np.nanmax(r_values))
        # Trim each delta_prime_array and psi_n_array to the common r range:
        for i, dp_array in enumerate(delta_prime_arrays):
            if dp_array is not None:
                dp_array = dp_array.sel(r=slice(min_r, max_r))
                delta_prime_arrays[i] = dp_array
        for i, psi_n in enumerate(psi_n_arrays):
            if psi_n is not None:
                psi_n = psi_n.sel(r=slice(min_r, max_r))
                psi_n_arrays[i] = psi_n

    # Double check all delta_prime_arrays now have the same coordinates after trimming:
    reference_coords = delta_prime_arrays[0].drop_vars(input_name).coords
    for i, dp_array in enumerate(delta_prime_arrays):
        dp_coords = dp_array.drop_vars(input_name).coords
        if not dp_coords.equals(reference_coords):
            print(f"Warning: Coordinates of 'Delta_prime_surf' in xarray for scan value {input_values[i]} are :{dp_array.coords}, which still do not match reference coordinates {reference_coords} after trimming.")
            if not run_concatenation:
                raise ValueError(f"After trimming, coordinates of 'Delta_prime_surf' in xarray for scan value {input_values[i]} still do not match reference coordinates. Cannot compile results.")

    compiled_xr = xr.concat(delta_prime_arrays, dim=input_name)

    assert len(psi_n_arrays) == len(delta_prime_arrays), "Number of psi_n_arrays does not match number of delta_prime_arrays. Cannot compile results."
    compiled_psi_n = xr.concat(psi_n_arrays, dim=input_name)
    compiled_xr = xr.merge([compiled_xr, compiled_psi_n])

    return compiled_xr
