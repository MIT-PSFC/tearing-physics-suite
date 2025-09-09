# This contains the core functionalities of tearing physics suite

import xarray as xr
import numpy as np
import copy
import tearing_physics_suite.global_vars
from tearing_physics_suite.environment import home_dir
from tearing_physics_suite.mre_analysis import analyse_with_mre
from tearing_physics_suite.fortran_wrappers import run_resistive_calculation, compile_xarrays
from tearing_physics_suite.delta_prime_extraction import extract_delta_primes

def nonlinear_resistive_calculation(eq_filename, ni_spline, ne_spline, te_keV_spline, ti_keV_spline, 
    nvec = [1], 
    energy_confinement_time = None,
    chi_perp_spline=None,
    k0=0.8227,
    k1=1.7,
    C0=0.6,
    wd_static=False, # Set true to ignore the variation in the ratio of perpendicular to parallel transport across the island, as island width varies
    test_numerical_stability=False,
    debug=True,
    debug_global_mre_quantities=False,
    psi_pedestal_cutoff=0.9,
    **kwargs):
    """ Runs linear and nonlinear tearing analysis on an equilibrium over a range 
    of toroidal mode numbers set by nvec. **kwargs are sent directly to the function 'run_resistive_calculation',
    setting the operational parameters of STRIDE, RDCON and PEST3.
    """

    xarray_vec = []
    pest3_xr_vec = []
    input_dict_vec = []

    #########################################################################################################
    # run calculation over vector of ns:
    #########################################################################################################

    for nn in nvec:
        #if test_numerical_stability:
        #   Run numerical stability test...

        comb_n_xr, n_pest3_xr, n_input_dict = analyse_with_mre(eq_filename, nn, ni_spline, ne_spline, te_keV_spline, ti_keV_spline,
            energy_confinement_time=energy_confinement_time,
            chi_perp_spline=chi_perp_spline,
            k0=k0,
            k1=k1,
            C0=C0,
            wd_static=False,
            **kwargs)

        xarray_vec.append(comb_n_xr)
        pest3_xr_vec.append(n_pest3_xr)
        input_dict_vec.append(n_input_dict)

    #########################################################################################################
    # look for failed pest3 runs:
    #########################################################################################################

    message = ''
    for xrp in pest3_xr_vec:
        if not (xrp is None):
            message += f'Warning:\n    PEST3 n = {xrp.n} failed to combine with other xarrays.'

    #########################################################################################################
    # clean up input dicts:
    #########################################################################################################

    # Remove n from all dicts in input_dict_vec:
    input_dict_vec2 = []
    for d in input_dict_vec:
        dcopy = copy.deepcopy(d)
        # Put nn-dependent inputs into a sub-dictionary:
        nn=dcopy['nn']
        # Check if ran pest3:
        if 'kband_pest' in input_dict_vec[0] and 'psihigh_pest' in input_dict_vec[0]:
            nn_dep_dict = {'kband_pest': dcopy['kband_pest'],
                            'psihigh_pest': dcopy['psihigh_pest']}
            nn_dep_dict_name = f'n{nn}_dependent_inputs'
            dcopy[nn_dep_dict_name] = nn_dep_dict
            # Remove the nn-dependent inputs from the main dictionary:
            dcopy.pop('kband_pest', None)
            dcopy.pop('psihigh_pest', None)
        dcopy.pop('nn', None)
        input_dict_vec2.append(dcopy)

    # Check that all dicts in input_dict_vec2 are identical (except for nn-dependent inputs):
    if len(input_dict_vec2) > 1:
        first_dict = input_dict_vec2[0]
        first_dict_comp = copy.deepcopy(first_dict)
        first_dict_comp.pop('n'+str(nvec[0])+'_dependent_inputs', None)
        for i, dict_item in enumerate(input_dict_vec2[1:], 1):
            # Add nn-dependent inputs to first_dict for output:
            if 'n'+str(nvec[i])+'_dependent_inputs' in dict_item:
                first_dict['n'+str(nvec[i])+'_dependent_inputs'] = dict_item['n'+str(nvec[i])+'_dependent_inputs']
            dict_item.pop('n'+str(nvec[i])+'_dependent_inputs', None)
            # Compare all dicts except for the nn-dependent inputs:
            assert compare_dicts(dict_item, first_dict_comp) , f"Dictionary at index {i} differs from the first dictionary"

    input_dict_out = first_dict

    #########################################################################################################
    # combine successfull xarrays and output them
    #########################################################################################################

    combined_xr = None

    # Concatenate xarray_vec:
    try:
        combined_xr = xr.concat(xarray_vec, dim='nn', coords='all')
        # Elevate variable nn to a coordinate:
        combined_xr = combined_xr.assign_coords(nn=combined_xr.nn)
        xarray_vec = None
    except ValueError as e:
        print(e)
        if debug:
            raise e

    #########################################################################################################
    # evaluate how different the Delta primes computed by each code are...
    #########################################################################################################
    if not (combined_xr is None) and len(combined_xr['code']) > 1:
        combined_xr = delta_prime_variability(combined_xr)

    #########################################################################################################
    # define global mre quantities 
    #########################################################################################################

    if debug_global_mre_quantities:
        return combined_xr, input_dict_out, pest3_xr_vec, xarray_vec

    if not (combined_xr is None):
        combined_xr = global_mre_quantities(combined_xr,psi_pedestal_cutoff=psi_pedestal_cutoff)

    return combined_xr, input_dict_out, pest3_xr_vec, xarray_vec


def compare_dicts(d1,d2):
    """ Compares two dictionaries, returning True if they are identical, False otherwise.
    """
    if d1.keys() != d2.keys():
        print("Dictionaries have different keys:")
        print("Different keys in d1:", set(d1.keys()) - set(d2.keys()))
        print("Different keys in d2:", set(d2.keys()) - set(d1.keys()))
        return False
    diff_vals=False
    for key in d1.keys():
        if d1[key] != d2[key]:
            print(f"Different values for key '{key}': d1 has {d1[key]}, d2 has {d2[key]}")
            diff_vals=True
    return not diff_vals

def linear_resistive_calculation(eq_filename, nvec = [1], test_numerical_stability=False,  debug=True, **kwargs):
    """ Runs linear tearing analysis on an equilibrium over a range 
    of toroidal mode numbers set by nvec. **kwargs are sent directly to the function 'run_resistive_calculation',
    setting the operational parameters of STRIDE, RDCON and PEST3.
    """

    xarray_vec = []
    pest3_xr_vec = []
    input_dict_vec = []

    #########################################################################################################
    # run calculation over vector of ns:
    #########################################################################################################

    for nn in nvec:
        #if test_numerical_stability:
        #   Run numerical stability test...
        rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict=run_resistive_calculation(eq_filename, nn, **kwargs)
        comb_n_xr, n_pest3_xr, n_input_dict = compile_xarrays(rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict)
        xarray_vec.append(comb_n_xr)
        pest3_xr_vec.append(n_pest3_xr)
        input_dict_vec.append(n_input_dict)

    #########################################################################################################
    # look for failed pest3 runs:
    #########################################################################################################

    message = ''
    for xrp in pest3_xr_vec:
        if not (xrp is None):
            message += f'Warning:\n    PEST3 n = {xrp.n} failed to combine with other xarrays.'

    #########################################################################################################
    # clean up input dicts:
    #########################################################################################################

    # Remove n from all dicts in input_dict_vec:
    input_dict_vec2 = []
    for d in input_dict_vec:
        dcopy = copy.deepcopy(d)
        # Put nn-dependent inputs into a sub-dictionary:
        nn=dcopy['nn']
        # Check if ran pest3:
        if 'kband_pest' in input_dict_vec[0] and 'psihigh_pest' in input_dict_vec[0]:
            nn_dep_dict = {'kband_pest': dcopy['kband_pest'],
                            'psihigh_pest': dcopy['psihigh_pest']}
            nn_dep_dict_name = f'n{nn}_dependent_inputs'
            dcopy[nn_dep_dict_name] = nn_dep_dict
            # Remove the nn-dependent inputs from the main dictionary:
            dcopy.pop('kband_pest', None)
            dcopy.pop('psihigh_pest', None)
        dcopy.pop('nn', None)
        input_dict_vec2.append(dcopy)

    # Check that all dicts in input_dict_vec2 are identical (except for nn-dependent inputs):
    if len(input_dict_vec2) > 1:
        first_dict = input_dict_vec2[0]
        first_dict_comp = copy.deepcopy(first_dict)
        first_dict_comp.pop('n'+str(nvec[0])+'_dependent_inputs', None)
        for i, dict_item in enumerate(input_dict_vec2[1:], 1):
            # Add nn-dependent inputs to first_dict for output:
            if 'n'+str(nvec[i])+'_dependent_inputs' in dict_item:
                first_dict['n'+str(nvec[i])+'_dependent_inputs'] = dict_item['n'+str(nvec[i])+'_dependent_inputs']
            dict_item.pop('n'+str(nvec[i])+'_dependent_inputs', None)
            # Compare all dicts except for the nn-dependent inputs:
            assert compare_dicts(dict_item, first_dict_comp) , f"Dictionary at index {i} differs from the first dictionary"

    input_dict_out = first_dict

    #########################################################################################################
    # combine successfull xarrays and output them
    #########################################################################################################

    combined_xr = None

    # Concatenate xarray_vec:
    try:
        combined_xr = xr.concat(xarray_vec, dim='nn', coords='all')
        # Elevate variable nn to a coordinate:
        combined_xr = combined_xr.assign_coords(nn=combined_xr.nn)
        xarray_vec = None
    except ValueError as e:
        print(e)
        if debug:
            raise e

    #########################################################################################################
    # evaluate how different the Delta primes computed by each code are...
    #########################################################################################################
    if not (combined_xr is None) and len(combined_xr['code']) > 1:
        combined_xr = delta_prime_variability(combined_xr)

    return combined_xr, input_dict_out, pest3_xr_vec, xarray_vec

def global_mre_quantities(combined_xr,psi_pedestal_cutoff=0.9):
    """ 
    Ranks nonlinear stability of all modes, and computes least stable modes
    via two metrics: largest nondimensional island growth rate (max_dwdtau), and smallest
    seed island needed to initiate an NTM (min_w_marg).
    """
    # Min w_marg_surf over all m, n
    # Max dwdtau over all m, n
    min_w_marg = combined_xr.w_marg_surf.min(dim=['r', 'nn'])
    max_dwdtau = combined_xr.dwdtau_max_surf.max(dim=['r', 'nn'])
    # Add these to combined_xr:
    combined_xr = combined_xr.assign(
        min_w_marg_allsurf=min_w_marg,
        max_dwdtau_allsurf=max_dwdtau
    )
    # For each Delta_prime_type, code, we want to rank the modes by min_w_marg and max_dwdtau over all r, nn:
    w_marg_rank = xr.full_like(combined_xr.w_marg_surf, np.nan)
    dwdtau_rank = xr.full_like(combined_xr.dwdtau_max_surf, np.nan)
    assert combined_xr.w_marg_surf.dims == combined_xr.dwdtau_max_surf.dims, "Dimensions of w_marg_surf and dwdtau_max_surf do not match"
    
    #########################################################################################################
    # Applying psi_n_rational mask 
    #########################################################################################################
    psi_n_rational_like_surfaces = combined_xr.psi_n_rational+0.0*combined_xr['Delta_prime_surf'] 
    # Set psi_n_rational_like_surfaces.loc[code='pest3'] equal to psi_n_rational_like_surfaces.loc[code='rdcon'] (since pest3 doesn't compute psi_n_rational)
    psi_n_rational_like_surfaces.loc[dict(code='pest3')] = psi_n_rational_like_surfaces.loc[dict(code='rdcon')]

    #########################################################################################################
    # Loop over all Delta_prime_type, code combinations to make local rankings
    #########################################################################################################
    for dpt in combined_xr.Delta_prime_type.values:
        for code in combined_xr.code.values:
            # Select the subset of combined_xr corresponding to this Delta_prime_type and code:
            subset = combined_xr.sel(Delta_prime_type=dpt, code=code)
            psi_n_rational_like_surfaces_subset = psi_n_rational_like_surfaces.sel(Delta_prime_type=dpt, code=code)
            assert subset.w_marg_surf.dims == subset.dwdtau_max_surf.dims == psi_n_rational_like_surfaces_subset.dims, "Dimensions of w_marg_surf, dwdtau_max_surf, and psi_n_rational_like_surfaces do not match"

            # Skip if no valid data:
            if subset.w_marg_surf.count() == 0 or subset.dwdtau_max_surf.count() == 0:
                continue

    #########################################################################################################
    # Rank w_marg_surf (smallest to largest):
    #########################################################################################################
            w_marg_surf_vals = subset.w_marg_surf.where(psi_n_rational_like_surfaces_subset < psi_pedestal_cutoff).values
            w_marg_ranks = np.argsort(np.argsort(w_marg_surf_vals, axis=None)) + 1 # +1 to make ranks start from 1
            w_marg_ranks = np.array(w_marg_ranks.reshape(w_marg_surf_vals.shape)).astype(float) # Convert to float to allow for NaNs
            # Make w_marg_ranks nan where w_marg_surf is nan:
            w_marg_ranks[np.isnan(w_marg_surf_vals)] = np.nan
            # Turn w_marg_ranks into a DataArray with the same coords as subset.w_marg_surf:
            w_marg_ranks_da = xr.DataArray(w_marg_ranks, coords=subset.w_marg_surf.coords, dims=subset.w_marg_surf.dims)
            # Put w_marg_ranks_da into combined_xr:
            w_marg_rank.loc[dict(Delta_prime_type=dpt, code=code)] = w_marg_ranks_da
            
    #########################################################################################################
    # Rank dwdtau_max_surf (largest to smallest):
    #########################################################################################################
            dwdtau_surf_vals = subset.dwdtau_max_surf.where(psi_n_rational_like_surfaces_subset < psi_pedestal_cutoff).values
            dwdtau_ranks = np.argsort(np.argsort(-dwdtau_surf_vals, axis=None)) + 1 # +1 to make ranks start from 1
            dwdtau_ranks = np.array(dwdtau_ranks.reshape(dwdtau_surf_vals.shape)).astype(float) # Convert to float to allow for NaNs
            # Make dwdtau_ranks nan where dwdtau_max_surf is nan:
            dwdtau_ranks[np.isnan(dwdtau_surf_vals)] = np.nan
            # Turn dwdtau_ranks into a DataArray with the same coords as subset.dwdtau_max_surf:
            dwdtau_ranks_da = xr.DataArray(dwdtau_ranks, coords=subset.dwdtau_max_surf.coords, dims=subset.dwdtau_max_surf.dims)
            # Put dwdtau_ranks_da into combined_xr:
            dwdtau_rank.loc[dict(Delta_prime_type=dpt, code=code)] = dwdtau_ranks_da
    
    combined_xr = combined_xr.assign(
        min_w_marg_rank=w_marg_rank,
        max_dwdtau_rank=dwdtau_rank
    )

    return combined_xr

def delta_prime_variability(xarray,comparison_var='code',
        run_bool_check=False,
        abs_threshold=0.05,
        rel_threshold=0.05
        #abs_PEST_threshold=0.3,
        #rel_PEST_threshold=0.1
        ):
    """ 
    Function that compares Delta_prime_surf across the variable 'comparison_var'. 
    The comparison is performed for all m,n modes, across all Delta_prime_surf types.
    
    Boolean True False values are generated to check whether the differences in Delta_prime_surf lie within the bounds 
    set by abs_threshold, rel_threshold, abs_PEST_threshold, and rel_PEST_threshold.
    """

    #########################################################################################################
    # check xarray has comparison_var in it
    #########################################################################################################

    assert comparison_var in xarray.dims, f"'{comparison_var}' not found in xarray dimensions: {list(xarray.dims.keys())}"

    if len(xarray[comparison_var]) == 0:
        print(f"Only one value of {comparison_var}' found, cross-variable comparisons not carried out.")
        return xarray

    #########################################################################################################
    # Record the maximum range of variation in Delta_prime_surf across comparison_var 
    #########################################################################################################

    xarray = add_comparison_across_var(xarray,xarray.Delta_prime_surf,comparison_var)
    if run_bool_check: # This is used specifically for input scans.
        xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere = add_bool_checks(xarray, comparison_var, abs_threshold, rel_threshold)

    #########################################################################################################
    # Special case: comparison_var = 'code', just compare stride and rdcon
    #   Will overlap with code case if only STRIDE and RDCON were ran
    #########################################################################################################

    if comparison_var == 'code' and 'stride' in xarray.Delta_prime_surf.code and 'rdcon' in xarray.Delta_prime_surf.code: 
        Delta_prime_surf_stride_and_rdcon = xarray.Delta_prime_surf.sel(code=xarray.Delta_prime_surf.code.isin(['stride', 'rdcon']))
        xarray = add_comparison_across_var(xarray,Delta_prime_surf_stride_and_rdcon,comparison_var,override_name='GPEC')

    #########################################################################################################
    # Special case: comparison_var = 'code', just compare stride, rdcon, and pest3 (no cylindrical)
    #   Will overlap with code case if only STRIDE and RDCON were ran
    #########################################################################################################

    if comparison_var == 'code' and 'pest3' in xarray.Delta_prime_surf.code and ('rdcon' in xarray.Delta_prime_surf.code or 'stride' in xarray.Delta_prime_surf.code): 
        Delta_prime_surf_pest3_stride_andor_rdcon = xarray.Delta_prime_surf.sel(code=xarray.Delta_prime_surf.code.isin(['stride', 'rdcon','pest3']))
        xarray = add_comparison_across_var(xarray,Delta_prime_surf_pest3_stride_andor_rdcon,comparison_var,override_name='GPECvsPEST')

    if run_bool_check:
        return xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere
    return xarray

def add_comparison_across_var(xarray, Delta_prime_surf, comparison_var,override_name=''):
    """ 
    Computes absolute and relative differences in Delta_prime_surf across the variable 'comparison_var'. 
    Will use comparison_var for the new variable name unless override_name is specified. """

    Delta_prime_diffs_across_var = np.abs(Delta_prime_surf.max(dim=comparison_var)-Delta_prime_surf.min(dim=comparison_var))
    Delta_prime_reldiffs_across_var = Delta_prime_diffs_across_var / np.abs(Delta_prime_surf).mean(dim=comparison_var)
    Delta_prime_reldiffs_across_var2 = Delta_prime_diffs_across_var / Delta_prime_surf.mean(dim=comparison_var)

    if len(override_name) == 0: 
        override_name = comparison_var

    # Add Delta_prime_diffs_across_var to xarray, with str(comparison_var) included in name:
    xarray = xarray.assign({
        f'Delta_prime_diff_across_{override_name}': Delta_prime_diffs_across_var,
        f'Delta_prime_reldiff_across_{override_name}': Delta_prime_reldiffs_across_var,
        f'Delta_prime_reldiff_across_{override_name}_2': Delta_prime_reldiffs_across_var2
    })

    return xarray

def add_bool_checks(xarray, comparison_var, abs_threshold, rel_threshold, Delta_prime_type='single helicity'):
    """ Checks if the relative difference in Delta primes across comparison_var
    exceeds the specified thresholds. Does so in for all m,n modes, as well as all m,n within psi95. """

    absdiffs_name = f'Delta_prime_diff_across_{comparison_var}'
    reldiffs_name = f'Delta_prime_reldiff_across_{comparison_var}'

    xarray = xarray.assign({
        f'{absdiffs_name}_thresh_exceeded': xarray[absdiffs_name] > abs_threshold,
        f'{reldiffs_name}_thresh_exceeded': xarray[reldiffs_name] > rel_threshold
    })

    #########################################################################################################
    # Applying psi_n_rational mask to check if thresh exceeded within the q95 window
    #########################################################################################################
    psi_n_rational_like_absdiffs = xarray.psi_n_rational+0.0*xarray[absdiffs_name]

    abs_thresh_exceeded_within_q95 = xarray[absdiffs_name].where(psi_n_rational_like_absdiffs < 0.95)
    rel_thresh_exceeded_within_q95 = xarray[reldiffs_name].where(psi_n_rational_like_absdiffs < 0.95)

    #xarray = xarray.assign({
    #    f'{absdiffs_name}_thresh_exceeded_psi95': abs_thresh_exceeded_within_q95,
    #    f'{reldiffs_name}_thresh_exceeded_psi95': rel_thresh_exceeded_within_q95
    #})

    #########################################################################################################
    # Cycling over all m,n
    #########################################################################################################

    abs_thresh_exceeded_anywhere_da = xarray[f'{absdiffs_name}_thresh_exceeded'].any(dim=["r","n"])
    rel_thresh_exceeded_anywhere_da = xarray[f'{reldiffs_name}_thresh_exceeded'].any(dim=["r","n"])

    abs_thresh_exceeded_psi95_anywhere_da = abs_thresh_exceeded_within_q95.any(dim=["r","n"])
    rel_thresh_exceeded_psi95_anywhere_da = rel_thresh_exceeded_within_q95.any(dim=["r","n"])

    #########################################################################################################
    # Choosing a specific type of Delta prime for the comparison:
    #########################################################################################################

    abs_thresh_exceeded_anywhere = abs_thresh_exceeded_anywhere_da.sel(Delta_prime_type=Delta_prime_type)
    rel_thresh_exceeded_anywhere = rel_thresh_exceeded_anywhere_da.sel(Delta_prime_type=Delta_prime_type)
    abs_thresh_exceeded_psi95_anywhere = abs_thresh_exceeded_psi95_anywhere_da.sel(Delta_prime_type=Delta_prime_type)
    rel_thresh_exceeded_psi95_anywhere = rel_thresh_exceeded_psi95_anywhere_da.sel(Delta_prime_type=Delta_prime_type)

    return xarray, abs_thresh_exceeded_anywhere, rel_thresh_exceeded_anywhere, abs_thresh_exceeded_psi95_anywhere, rel_thresh_exceeded_psi95_anywhere

