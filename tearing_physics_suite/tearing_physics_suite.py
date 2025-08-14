# This contains the core functionalities of tearing physics suite

import xarray as xr
import numpy as np
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
        input_dict_vec2.append(d.pop('n', None))
    # Check that all dicts in input_dict_vec2 are identical:
    if len(input_dict_vec2) > 1:
        first_dict = input_dict_vec2[0]
        for i, dict_item in enumerate(input_dict_vec2[1:], 1):
            assert dict_item == first_dict, f"Dictionary at index {i} differs from the first dictionary"
    input_dict_out = first_dict

    #########################################################################################################
    # combine successfull xarrays and output them
    #########################################################################################################

    combined_xr = None

    # Concatenate xarray_vec:
    try:
        combined_xr = xr.concat(xarray_vec, dim='n', coords='all', **kwargs)
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

    if not (combined_xr is None):
        combined_xr = global_mre_quantities(combined_xr)

    return combined_xr, input_dict_out, pest3_xr_vec, xarray_vec

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
        dcopy = d.copy(deep=True)
        dcopy.pop('n', None)
        input_dict_vec2.append(dcopy)
        if True:
            print(input_dict_vec2[-1])
    # Check that all dicts in input_dict_vec2 are identical:
    if len(input_dict_vec2) > 1:
        first_dict = input_dict_vec2[0]
        for i, dict_item in enumerate(input_dict_vec2[1:], 1):
            assert dict_item == first_dict, f"Dictionary at index {i} differs from the first dictionary"
    input_dict_out = first_dict

    #########################################################################################################
    # combine successfull xarrays and output them
    #########################################################################################################

    combined_xr = None

    # Concatenate xarray_vec:
    try:
        combined_xr = xr.concat(xarray_vec, dim='n', coords='all')
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

def global_mre_quantities(combined_xarray):
    """ 
    Ranks nonlinear stability of all modes, and computes least stable modes
    via two metrics: largest nondimensional island growth rate (max_dwdtau), and smallest
    seed island needed to initiate an NTM (min_w_marg).
    """

    # Min w_marg_surf over all m, n
    # Max dwdtau over all m, n
    min_w_marg = combined_xarray.w_marg_surf.min(dim=['r', 'n'])
    max_dwdtau = combined_xarray.dwdtau_max_surf.max(dim=['r', 'n'])

    # Add these to combined_xarray:
    combined_xarray = combined_xarray.assign(
        min_w_marg_allsurf=min_w_marg,
        max_dwdtau_allsurf=max_dwdtau
    )

    # A ranking of the minimum w_marg and maximum dwdtau over all m, n
    negative_max_dwdtau = -1.0*combined_xarray.max_dwdtau # negative_max_dwdtau.rank is equal to ranking max_dwdtau from largest to smallest
    combined_xarray = combined_xarray.assign(
        min_w_marg_rank=combined_xarray.min_w_marg.rank(dim=['r', 'n']),
        max_dwdtau_rank=negative_max_dwdtau.rank(dim=['r', 'n'])
    )

    return combined_xarray

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
    Delta_prime_reldiffs_across_var = Delta_prime_diffs_across_var / Delta_prime_surf.mean(dim=comparison_var)

    if len(override_name) == 0: 
        override_name = comparison_var

    # Add Delta_prime_diffs_across_var to xarray, with str(comparison_var) included in name:
    xarray = xarray.assign({
        f'Delta_prime_diff_across_{override_name}': Delta_prime_diffs_across_var,
        f'Delta_prime_reldiff_across_{override_name}': Delta_prime_reldiffs_across_var
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

