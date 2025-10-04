# Python functions to scan input parameters during the resistive calculation 

import pandas as pd
import xarray as xr
import numpy as np
import os

from tearing_physics_suite.environment import home_dir
import tearing_physics_suite.fortran_wrappers as tfw
from tearing_physics_suite.fortran_wrappers import compile_xarrays

def scan_1D_input(input_name,input_values,eq_filename,
        nn,
        debug=False,
        **kwargs):
    """
    Runs a 1D scan over a specified input parameter for the resistive calculation.

    Parameters:
        input_name: str, name of the input parameter to scan
        input_values: list, values to scan over
        eq_filename: str, path to the equilibrium file
        nn: int, number of modes to use in the calculation
        kwargs: additional keyword arguments to pass to the run_resistive_calculation function

    Returns output of extract_scanned_xrs:
        xarrays: list of xarrays containing the results of the scan
        pest3_xarrays: list of xarrays containing the results from pest3 (separate in case there is a discrepancy in the number of modes)
        input_dicts: list of dictionaries containing the input parameters for each run
        input_values: list of values used for the scan
        input_name: str, name of the input parameter scanned
        message: str, message containing the input values, single-helicity delta prime results and q-surface information
        deltaprimes: list of lists containing first set of delta prime values for each run
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
        return results

    return extract_scanned_xrs(results, input_name)

def extract_scanned_xrs(results, input_name):
    """
    Extracts the results from the scan and returns them in an xarray.
    Parameters:
        results: list of tuples, each tuple contains the xarrays and input dictionaries from the resistive calculation  
        input_name: str, name of the input parameter scanned

    Returns:
        xarrays: list of xarrays containing the results of the scan
        pest3_xarrays: list of xarrays containing the results from pest3 (separate in case there is a discrepancy in the number of modes)
        input_dicts: list of dictionaries containing the input parameters for each run
        input_values: list of values used for the scan
        input_name: str, name of the input parameter scanned
        message: str, message containing the input values, single-helicity delta prime results and q-surface information
        deltaprimes: list of lists containing first set of delta prime values for each run
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
    message = str("   "+input_name+":"+str(input_values)+'\n')
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
        message+=str("   at q-surface "+ str(xarrays_msg[0].r.values[0]))
    else:
        message+=str("   q-surfaces:"+ str([xarrays_msg[i].r.values[0] for i in range(len(xarrays_msg))]))
    return xarrays, pest3_xarrays, input_dicts, input_values, input_name, message, deltaprimes

'''
# UNFINISHED, DEPRECATED
def extract_Delta_primes_1D(results, input_name):
    """
    Extracts the results from the scan and returns them in an xarray.
    """
    data = []
    for result in results:
        rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict = result
        # Combine rdcon_xr['Delta_prime'], stride_xr['Delta_prime'], pest3_xr['Delta_prime'] into one DataFrame
        rdcon_delta_prime = None
        stride_delta_prime = None
        pest3_delta_prime = None
        pest3_delta_prime_errs = None
        if not rdcon_stride_input_dict is None:
            if not pest3_input_dict is None:
                rdcon_stride_input_dict.update(pest3_input_dict)
                output_dict = rdcon_stride_input_dict
        elif not pest3_input_dict is None:
            output_dict = pest3_input_dict

        assert input_name in output_dict, f"Debug scan_1D_input: {input_name} not found in output_dict, available keys: {output_dict.keys()}"

        delta_prime_DAs = []
        delta_prime_perr_DAs = []
        if not (rdcon_xr is None):
            if 'Delta_prime' in rdcon_xr:
                rdcon_delta_prime_da = rdcon_xr['Delta_prime']
                rdcon_delta_prime_da = rdcon_delta_prime_da.expand_dims(dim='code', axis=0)
                rdcon_delta_prime_da['code'] = ['r']
                delta_prime_DAs.append(rdcon_delta_prime_da)

        if not (stride_xr is None):
            if 'Delta_prime' in stride_xr:
                stride_delta_prime_da = stride_xr['Delta_prime']
                stride_delta_prime_da = stride_delta_prime_da.expand_dims(dim='code', axis=0)
                stride_delta_prime_da['code'] = ['s']
                delta_prime_DAs.append(stride_delta_prime_da)


        if not (pest3_xr is None):
            if 'Delta_prime' in pest3_xr:
                pest3_delta_prime_da = pest3_xr['Delta_prime']
                pest3_delta_prime_da = pest3_delta_prime_da.expand_dims(dim='code', axis=0)
                pest3_delta_prime_da['code'] = ['p']
                delta_prime_DAs.append(pest3_delta_prime_da)
                if 'Delta_prime_perr' in pest3_xr:
                    pest3_delta_prime_err_da = pest3_xr['Delta_prime_perr']
                    pest3_delta_prime_err_da = pest3_delta_prime_err_da.expand_dims(dim='code', axis=0)
                    pest3_delta_prime_err_da['code'] = ['p']
                    pest3_delta_prime_err_da = pest3_delta_prime_err_da.expand_dims(dim='i', axis=3)
                    pest3_delta_prime_err_da['i'] = [0]
                    delta_prime_perr_DAs.append(pest3_delta_prime_err_da)

        # Take 5, then:  get all concatenation working
        # Then, add qsing or something... (one for each code)
        # Then, change verbose to extract first few diagonal elements

        #pest3_delta_prime_errs = pest3_xr['Delta_prime_perr']
        # Add 1d dimension for i
        #pdper_dataarray = pest3_delta_prime_errs.expand_dims(dim='i', axis=2)
        # Set i to 0
        #pdper_dataarray['i'] = [0]
        # Add 1d dimension for code
        #pdper_dataarray = pdper_dataarray.expand_dims(dim='code', axis=0)
        #pdper_dataarray['code'] = ['p']

        delta_prime_xr = xr.concat(delta_prime_DAs, dim='code').to_dataset(name='Delta_prime')

        if len(delta_prime_perr_DAs) > 0:
            delta_prime_err_da = xr.concat(delta_prime_perr_DAs, dim='code')
            # Combine the two xarrays
            delta_prime_xr = xr.merge([delta_prime_xr, delta_prime_err_da.to_dataset(name='Delta_prime_perr')])

        # Set 'input_name' as a coordinate, with values from the output_dict
        delta_prime_xr[input_name] = xr.DataArray(
            output_dict[input_name],
            dims=[input_name],
            coords={input_name: [output_dict[input_name]]}
        )
        data.append(delta_prime_xr)

    # Combine all xarrays into one
    if len(data) > 0:
        xr_data = xr.concat(data, dim=input_name)
        # Set the input_name as a coordinate
        xr_data[input_name] = xr.DataArray()
    else:
        xr_data = xr.Dataset()

    #
    df = pd.DataFrame(data)
    df.set_index(input_name, inplace=True)
    #Get xarray from df
    xr_data = xr.Dataset.from_dataframe(df)

    return xr_data

# UNFINISHED, # DEPRECATED
def scan_2D_inputs(input_names, input1_values, input2_values, eq_filename, nn,
        **kwargs):
    """
    Runs a 2D scan over specified input parameters for the resistive calculation.
    input_names: list of str, names of the input parameters to scan
    input1_values: list, values to scan over for the first input parameter
    input2_values: list, values to scan over for the second input parameter
    eq_filename: str, path to the equilibrium file
    nn: int, number of modes to use in the calculation
    kwargs: additional keyword arguments to pass to the run_resistive_calculation function
    """

    # Check input_names are length 2
    assert len(input_names) == 2

    # Check input1_name and input2_name are not in kwargs
    if input_names[0] in kwargs:
        kwargs.pop(input_names[0])
    if input_names[1] in kwargs:
        kwargs.pop(input_names[1])

    results = []
    for value1 in input1_values:
        for value2 in input2_values:
            # Add input names and values to output_prefix
            kwargs["output_prefix"] += f"{input_names[0]}_{value1}_{input_names[1]}_{value2}"

            print(f"Running scan for {input_names[0]} = {value1}, {input_names[1]} = {value2}")
            result = tfw.run_resistive_calculation(
                eq_filename,
                nn,
                **{input_names[0]: value1, input_names[1]: value2},
                **kwargs
            )
            results.append(result)

    return extract_m_Delta_primes_2D(results, input_names), results

#UNFINISHED
def extract_m_Delta_primes_2D(results, input_names):
    """
    Extracts the results from the 2D scan and returns them in a DataFrame.

    """
    data = []
    for result in results:
        rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict = result
        # Combine rdcon_xr['Delta_prime'], stride_xr['Delta_prime'], pest3_xr['Delta_prime'] into one DataFrame
        rdcon_delta_prime = None
        stride_delta_prime = None
        pest3_delta_prime = None
        pest3_delta_prime_errs = None
        
        if not (rdcon_xr is None):
            if 'Delta_prime' in rdcon_xr:
                rdcon_delta_prime = rdcon_xr['Delta_prime'].values
        if not (stride_xr is None):
            if 'Delta_prime' in stride_xr:
                stride_delta_prime = stride_xr['Delta_prime'].values
        if not (pest3_xr is None):
            if 'Delta_prime' in pest3_xr:
                pest3_delta_prime = pest3_xr['Delta_prime'].values
            if 'Delta_prime_perr' in pest3_xr:
                pest3_delta_prime_errs = pest3_xr['Delta_prime_perr'].values

        data.append({
            'DP_rdcon': rdcon_delta_prime,
            'DP_stride': stride_delta_prime,
            'DP_pest3': pest3_delta_prime,
            'DP_pest3_err': pest3_delta_prime_errs,
            input_names[0]: rdcon_stride_input_dict[input_names[0]],
            input_names[1]: rdcon_stride_input_dict[input_names[1]]
        })

    df = pd.DataFrame(data)
    df.set_index([input_names[0], input_names[1]], inplace=True)
    #Get xarray from df
    xr_data = xr.Dataset.from_dataframe(df)

    return xr_data
'''