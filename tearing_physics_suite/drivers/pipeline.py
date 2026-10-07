# This contains the core functionalities of tearing physics suite

import xarray as xr
import numpy as np
import copy
import os
home_dir = os.environ['TPSHOME']
from tearing_physics_suite.wrappers.run_codes import run_resistive_calculation
from tearing_physics_suite.physics.combine import compile_xarrays
from tearing_physics_suite.physics.delta_prime_extraction import extract_delta_primes
from tearing_physics_suite.wrappers.gpec_inputs import zeff_dict
from tearing_physics_suite.physics.cross_field_transport import chi_para_lmfp_no_w_on_modes, chi_para_smfp_on_modes, chi_para_lmfp_noisland_on_modes, chi_perp_on_modes
from tearing_physics_suite.physics.combine import _uniquify_r
from tearing_physics_suite.physics.global_quantities import delta_prime_variability, global_mre_quantities
from tearing_physics_suite.physics.mre_model import extract_critical_mre_factors_on_modes
from tearing_physics_suite.physics.surface_terms import deltaprime_crit_on_modes, mre_terms_on_modes

def nonlinear_resistive_calculation(eq_filename, ni_spline, ne_spline, te_keV_spline, ti_keV_spline, 
    Zeff = None, 
    average_ion_mass = None,
    diamagnetic_rotation_ion_charge=None,
    # Rotation splines
    Er_spline=None, # Assuming input units of V/m
    omega_splines=None, # Dictionary of splines for rotation frequencies in rad/s.
    q_surfs_of_interest=[1.0],
    psi_surfs_of_interest=[0.95],
    Coulomb_logarithm=None, # If None, calculate using Wesson formula. Otherwise use this value for all rational surfaces, to match M3DC1 simulations for example.
    eta_fac=1.0, # Factor to multiply Spitzer resistivity by, to match artificial manipulation in resistive simulations.
    nvec = [1],
    energy_confinement_time = None,
    chi_perp_spline=None,
    k0=0.8227,
    k1=1.7,
    C0=0.6,
    wd_static=False, # Set true to ignore the variation in the ratio of perpendicular to parallel transport across the island, as island width varies
    force_lmfp=False, 
    test_numerical_stability=False,
    debug=True,
    debug_global_mre_quantities=False,
    psi_pedestal_cutoff=0.9, # Surfaces inside this cutoff in norm. pol. flux are included when finding the minimum marginally stable island width 
    **kwargs):
    """Run linear and nonlinear tearing analysis on an equilibrium for multiple toroidal mode numbers set by nvec.

    Wraps run_resistive_calculation (Delta' computation) and analyse_with_mre
    (modified Rutherford equation island evolution model). Requires Zeff and average_ion_mass to ensure user has chosen
    self-consistent kinetic profiles.

    Parameters
    ----------
    eq_filename : str
        Path to the equilibrium file.
    ni_spline, ne_spline, te_keV_spline, ti_keV_spline : 1DSpline
        Ion/electron density [m^-3] and temperature [keV] vs psi_n.
    Zeff : dict or scalar
        Effective ion charge (for chi_para and bootstrap current).
        If dict, 'x' are psi_n values, 'y' are Zeff(psi_n) values.
    average_ion_mass : float
        Mean ion mass in AMU (for Alfven speed / mass density).
    nvec : list of int
        Toroidal mode numbers to analyse.
    **kwargs
        Forwarded to run_resistive_calculation (STRIDE/RDCON/PEST3 options).

    Returns
    -------
    combined_xr : xr.Dataset or None
        Combined xarray with Delta primes, MRE quantities, and global metrics across all n.
        'r'/'r_prime' are positional indices (see _uniquify_r); real values are in
        r_value/r_prime_value. Use sel_rational or collapse_to_primary to select by value.
    input_dict_out : dict
        Merged input parameters from RDCON/STRIDE/PEST3 across all n.
    pest3_xr_vec : list of xr.Dataset
        PEST3-specific xarray outputs per n (None entries if PEST3 failed).
    xarray_vec : list of xr.Dataset
        Per-n combined xarray datasets before concatenation.
    """

    if Zeff is None:
        raise ValueError("Zeff must be provided for nonlinear resistive calculation.")
    Zeff = zeff_dict(Zeff)
    if average_ion_mass is None:
        raise ValueError("average_ion_mass must be provided for nonlinear resistive calculation.")

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
            Er_spline=Er_spline,
            omega_splines=omega_splines,
            q_surfs_of_interest=q_surfs_of_interest,
            psi_surfs_of_interest=psi_surfs_of_interest,
            energy_confinement_time=energy_confinement_time,
            chi_perp_spline=chi_perp_spline,
            k0=k0,
            k1=k1,
            C0=C0,
            wd_static=wd_static,
            Zeff=Zeff,
            average_ion_mass=average_ion_mass,
            diamagnetic_rotation_ion_charge=diamagnetic_rotation_ion_charge,
            force_lmfp=force_lmfp,
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

    input_dict_out = clean_multi_n_dictionaries(input_dict_vec)

    #########################################################################################################
    # combine successfull xarrays and output them
    #########################################################################################################

    combined_xr = None

    # Concatenate xarray_vec:
    try:
        xarray_vec = [_uniquify_r(da) for da in xarray_vec]
        combined_xr = xr.concat(xarray_vec, dim='nn', coords='all', join='outer')
        # Elevate variable nn to a coordinate:
        combined_xr = combined_xr.assign_coords(nn=combined_xr.nn)
        if not debug:
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

def clean_multi_n_dictionaries(input_dict_vec):
    """Merge per-n input dictionaries into a single dict, grouping n-dependent entries.

    Keys that vary across n values are placed into sub-dictionaries named
    'n{nn}_dependent_inputs'. The 'nn' key is removed from all dicts.

    Parameters
    ----------
    input_dict_vec : list of dict
        One dictionary per toroidal mode number, each containing an 'nn' key.

    Returns
    -------
    dict
        Merged dictionary. If only one n, returns that dict directly.
    """

    # Remove n from all dicts in input_dict_vec:
    input_dict_vec2 = []
    nn_vec = []
    for d in input_dict_vec:
        nn_vec.append(d['nn'])
        dcopy = copy.deepcopy(d)
        dcopy.pop('nn', None)
        input_dict_vec2.append(dcopy)

    # Log first dict and return if only one dict:
    first_dict = input_dict_vec2[0]
    if len(input_dict_vec2) == 1:
        return first_dict

    # Check that the keys of all dicts in input_dict_vec2 are identical:
    first_dict_keys = set(input_dict_vec2[0].keys())
    extra_keys_alln = []
    one_missing_key=False
    for i, d in enumerate(input_dict_vec2[1:], 1):
        if set(d.keys()) != first_dict_keys:
            print(f"Dictionary at index {i} has different keys than the first dictionary.")
            print(" Differing keys in first dict:", first_dict_keys - set(d.keys()))
            print(" Differing keys in this dict:", set(d.keys()) - first_dict_keys)
            # convert set(d.keys()) - first_dict_keys into a list and append to extra_keys_alln:
            extra_keys_alln.append(list(set(d.keys()) - first_dict_keys))
            one_missing_key=True
        else:
            extra_keys_alln.append([])

    # Go through every key in first_dict_keys and check that all dicts have the same value for that key:
    keys_with_varied_values = []
    for key in first_dict_keys:
        first_value = input_dict_vec2[0][key]
        all_same = True
        for i, d in enumerate(input_dict_vec2[1:], 1):
            if key not in d:
                all_same = False
            else:
                try:
                    not_equal = d[key] != first_value
                    # Handle numpy arrays and other iterables
                    if hasattr(not_equal, '__iter__'):
                        not_equal = np.any(not_equal)
                    if not_equal:
                        all_same = False
                except (ValueError, TypeError):
                    all_same = False
        if not all_same:
            keys_with_varied_values.append(key)

    if len(keys_with_varied_values) > 0 or one_missing_key:
        print("The following keys have varied values across the dictionaries, and will be grouped into nn-dependent sub-dictionaries:")
        print(keys_with_varied_values)
        print("The following keys are missing in some dictionaries, and will be grouped into nn-dependent sub-dictionaries:")
        print(extra_keys_alln)
        for i, nn in enumerate(nn_vec[1:], 1):
            nn_dep_dict = {}
            for key in keys_with_varied_values:
                nn_dep_dict[key] = input_dict_vec2[i].get(key, None)
            for key in extra_keys_alln[i-1]:
                nn_dep_dict[key] = input_dict_vec2[i][key]
            nn_dep_dict_name = f'n{nn}_dependent_inputs'
            first_dict[nn_dep_dict_name] = nn_dep_dict
        # Remove the nn-dependent inputs from the main dictionary:
        for key in keys_with_varied_values:
            first_dict.pop(key, None)

    return first_dict

def compare_dicts(d1,d2):
    """Compare two dictionaries, returning True if all keys and values match.

    Prints which keys or values differ.
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

    Returns
    -------
    combined_xr : xr.Dataset or None
        Combined xarray with Delta primes across all n.
    input_dict_out : dict
        Merged input parameters from RDCON/STRIDE/PEST3.
    pest3_xr_vec : list of xr.Dataset
        PEST3-specific outputs per n.
    xarray_vec : list of xr.Dataset
        Per-n combined xarray datasets.
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
        if not comb_n_xr is None:
            print("  n = ",nn,":",comb_n_xr.Delta_prime_surf.isel(nn=0,Delta_prime_type=0).where(comb_n_xr.Delta_prime_surf.isel(Delta_prime_type=0).r<(comb_n_xr.Delta_prime_surf.isel(Delta_prime_type=0).r.min()+3),drop=True))
            print("  at q = ",comb_n_xr.q_rational.isel(nn=0,code=0).values[0:3])

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

    input_dict_out = clean_multi_n_dictionaries(input_dict_vec)

    #########################################################################################################
    # combine successfull xarrays and output them
    #########################################################################################################

    combined_xr = None

    # Concatenate xarray_vec:
    try:
        xarray_vec = [_uniquify_r(da) for da in xarray_vec]
        combined_xr = xr.concat(xarray_vec, dim='nn', coords='all', join='outer')
        # Elevate variable nn to a coordinate:
        combined_xr = combined_xr.assign_coords(nn=combined_xr.nn)
        if not debug:
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


# To do:
# Add pressure check (kinetic vs equilibrium)

def analyse_with_mre(eq_filename, nn, ni_spline, ne_spline, te_keV_spline, ti_keV_spline,
        # Rotation splines
        Er_spline=None, # Assuming input units of V/m
        omega_splines=None, # Dictionary of splines for rotation frequencies in rad/s.
        q_surfs_of_interest=[1.0],
        psi_surfs_of_interest=[0.95],
        energy_confinement_time = None,
        chi_perp_spline=None,
        k0=0.8227,
        k1=1.7,
        C0=0.6,
        wd_static=False, # Set true to ignore the variation in the ratio of perpendicular to parallel transport across the island, as island width varies
        debug_mre_terms=False,
        debug=False,
        delete_attrs=True,
        average_ion_mass=2.5, # Average ion mass in amu, used for alfven time calculation
        force_lmfp=False, # Default False <=> whichever parallel transport calculation is more physical, lmfp or smfp, is used. Set True to always use lmfp.
        Coulomb_logarithm=None, # If None, calculate using Wesson formula. Otherwise use this value for all rational surfaces.
        eta_fac=1.0, # Factor to multiply Spitzer resistivity by, to match artificial manipulation in resistive simulations.
        diamagnetic_rotation_ion_charge=None, # Ion charge for diamagnetic rotation calculation. If None, inferred from on-axis ne/ni.
        **kwargs):
    """ 
    Executive function that calculates Delta primes with run_resistive_calculation, then runs MRE analysis on output deltaprimes, returning
    a fully combined xarray.

    Parameters
    ----------
    eq_filename : str
        Path to MHD equilibrium file.
    nn : int
        Toroidal mode number.
    ni_spline, ne_spline, te_keV_spline, ti_keV_spline : 1DSpline
        Density (m^-3) and temperature (keV) profiles as functions of normalised poloidal flux.

    Returns
    -------
    combined_xr : xr.Dataset
        Combined xarray containing Delta primes, MRE terms, and island width analysis.
    pest3_xr : xr.Dataset or None
        Separate PEST3 output if it could not be merged into combined_xr.
    input_dict : dict
        Dictionary of all input parameters used in the calculation.
    """ 

    # Check that either energy_confinement_time or chi_perp_spline is defined:
    assert not (chi_perp_spline is None and energy_confinement_time is None), "Must either define energy_confinement_time or chi_perp_spline for MRE analysis."

    #########################################################################################################
    # Run resistive delta prime calculation: 
    #########################################################################################################
    rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict = run_resistive_calculation(eq_filename,nn,**kwargs)

    # Combine input dictionaries:
    if not (rdcon_stride_input_dict is None): #RDCON dict present
        if not (pest3_input_dict is None): # PEST3 dict present
            rdcon_stride_input_dict.update(pest3_input_dict)
        input_dict = rdcon_stride_input_dict
    elif not (pest3_input_dict is None):
        input_dict = pest3_input_dict
    else:
        input_dict = {}

    # Add wd_static, energy_confinement_time, k0, k1, C0, wd_static to input_dict:
    input_dict['wd_static'] = wd_static
    input_dict['energy_confinement_time'] = energy_confinement_time
    input_dict['k0'] = k0
    input_dict['k1'] = k1
    input_dict['C0'] = C0

    #########################################################################################################
    # Fill out rdcon_xr with important MRE terms:
    #########################################################################################################
    rdcon_xr = mre_terms_on_modes(rdcon_xr, ni_spline, ne_spline, te_keV_spline, ti_keV_spline, average_ion_mass=average_ion_mass, Coulomb_logarithm=Coulomb_logarithm, eta_fac=eta_fac, Er_spline=Er_spline, omega_splines=omega_splines, q_surfs_of_interest=q_surfs_of_interest, psi_surfs_of_interest=psi_surfs_of_interest, diamagnetic_rotation_ion_charge=diamagnetic_rotation_ion_charge)
    rdcon_xr = chi_para_lmfp_no_w_on_modes(rdcon_xr)
    rdcon_xr = chi_para_lmfp_noisland_on_modes(rdcon_xr)
    rdcon_xr = chi_para_smfp_on_modes(rdcon_xr)
    rdcon_xr = chi_perp_on_modes(rdcon_xr, energy_confinement_time=energy_confinement_time, chi_perp_spline=chi_perp_spline)
    rdcon_xr = deltaprime_crit_on_modes(rdcon_xr, force_lmfp=force_lmfp)

    if debug_mre_terms:
        return rdcon_xr, None, None

    # Combine xarrays:
    xarrays = []

    #########################################################################################################
    # RDCON delta xarray, delta prime and MRE calculation
    #########################################################################################################
    if not (rdcon_xr is None): 
        # Turn all attributes into variables:
        for attr_key in rdcon_xr.attrs.keys():
            rdcon_xr[attr_key] = rdcon_xr.attrs[attr_key]
        if delete_attrs:
            rdcon_xr.attrs = {}
        # Add new dimension for code to rdcon_xr
        rdcon_xr_expanded = rdcon_xr.expand_dims(dim='code', axis=0)
        rdcon_xr_expanded['code'] = ['rdcon']
        if 'Delta_prime' in rdcon_xr_expanded:
            rdcon_xr_expanded = extract_delta_primes(rdcon_xr_expanded)
            rdcon_xr_expanded = extract_critical_mre_factors_on_modes(rdcon_xr_expanded,rdcon_xr_expanded,k0=k0,k1=k1,C0=C0,iterator=wd_static, force_lmfp=force_lmfp)
        # Add to xarrays list
        xarrays.append(rdcon_xr_expanded)

    #########################################################################################################
    # STRIDE delta xarray, delta prime and MRE calculation
    #########################################################################################################
    if not (stride_xr is None):
        # Turn all attributes into variables:
        for attr_key in stride_xr.attrs.keys():
            stride_xr[attr_key] = stride_xr.attrs[attr_key]
        if delete_attrs:
            stride_xr.attrs = {}
        # Add new dimension for code to stride_xr
        stride_xr_expanded = stride_xr.expand_dims(dim='code', axis=0)
        stride_xr_expanded['code'] = ['stride']
        if 'Delta_prime' in stride_xr_expanded:
            # Calculate delta' values for stride_xr
            stride_xr_expanded = extract_delta_primes(stride_xr_expanded)
            stride_xr_expanded = extract_critical_mre_factors_on_modes(stride_xr_expanded,rdcon_xr_expanded,k0=k0,k1=k1,C0=C0,iterator=wd_static, force_lmfp=force_lmfp)
        xarrays.append(stride_xr_expanded)
            
    #########################################################################################################
    # PEST3 delta xarray and delta prime calculation
    #########################################################################################################
    pest3_xr_expanded = None
    if not (pest3_xr is None):
        # Turn all attributes into variables:
        for attr_key in pest3_xr.attrs.keys():
            pest3_xr[attr_key] = pest3_xr.attrs[attr_key]
        if delete_attrs:
            pest3_xr.attrs = {}
        # Add new dimension for code to pest3_xr
        pest3_xr_expanded = pest3_xr.expand_dims(dim='code', axis=0)
        pest3_xr_expanded['code'] = ['pest3']
        if 'Delta_prime' in pest3_xr_expanded:
            assert 'Delta_prime_perr' in pest3_xr_expanded, "Current version of extract_delta_primes assumes this."
            pest3_xr_expanded = extract_delta_primes(pest3_xr_expanded)
            pest3_xr_expanded = extract_critical_mre_factors_on_modes(pest3_xr_expanded,rdcon_xr_expanded,k0=k0,k1=k1,C0=C0,iterator=wd_static, force_lmfp=force_lmfp)
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
            combined_xr = xr.concat(xarrays, dim='code', coords='all')
            pest3_xr_out = None
        except Exception as e:
            if not (pest3_xr is None): #We remove pest3_xr_expanded from xarrays and retry
                xarrays = xarrays[:-1]  # Remove the last element (pest3_xr_expanded)
                combined_xr = xr.concat(xarrays, dim='code', coords='all')
            print("Error combining xarrays:", e)
            if debug:
                raise e

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
