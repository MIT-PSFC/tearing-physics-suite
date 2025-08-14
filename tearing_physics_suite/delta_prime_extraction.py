# Scripts to calculate effective delta' values using generalisation
# from Brennan & Sugiyama 2006.

import math
import xarray as xr
from sympy import Matrix
import sympy
import numpy as np
import jax.numpy as jnp
from jax import jacfwd
from tearing_physics_suite.utils import trim_nans

def extract_delta_primes(inputxr,debug=False, couple_reals=True):
    """ 
        Takes input xarray with Delta_prime variable, and returns input xarray with complex Delta_prime_surf on rational surfaces, calculated using extract_delta_primes_(). These
        include single helicity complex delta' values, and outer-region coupled delta' values (Brennan & Sugiyama PoP 2006). If 
        Delta_prime_perr (pest error) or Delta_prime_err is present, this function will also propagate those errors through the outer region coupling. 

        Assume Delta_prime_perr and Delta_prime_err are real.  
    """

    # Check if Delta_prime is in inputxr:
    assert "Delta_prime" in inputxr, "Input xarray must contain 'Delta_prime' variable."
    delta_prime_real_component = inputxr["Delta_prime"].sel(i=0).values
    delta_prime_im_component = inputxr["Delta_prime"].sel(i=1).values
    # Make a combined delta_prime with its real and complex values together:
    delta_primes = delta_prime_real_component + 1j * delta_prime_im_component 
    # Check delta_prime is of type complex
    assert np.iscomplexobj(delta_primes), "Delta_prime must be a complex array."

    # Check if Delta_prime_perr or Delta_prime_err is in inputxr:
    if "Delta_prime_perr" in inputxr:
        delta_prime_perr = inputxr["Delta_prime_perr"].values
    else:
        delta_prime_perr = None

    if "Delta_prime_err" in inputxr:
        delta_prime_err = inputxr["Delta_prime_err"].values
    else:
        delta_prime_err = None

    delta_prime_single_helicity_err=None
    delta_prime_single_helicity_perr=None

    if (not delta_prime_perr is None) and (not delta_prime_err is None): #Assume pest3 since Delta_prime_perr is present
        _, _, _, _, _, delta_prime_single_helicity_perr, delta_prime_eff_perr, delta_prime_nn_eff_perr, delta_prime_2nn_eff_perr, divisors_perr = extract_delta_primes_(delta_primes, debug=debug, delta_prime_errs=delta_prime_perr, couple_reals=couple_reals)
        delta_prime_single_helicity, delta_prime_eff, delta_prime_nn_eff, delta_prime_2nn_eff, divisors, delta_prime_single_helicity_err, delta_prime_eff_err, delta_prime_nn_eff_err, delta_prime_2nn_eff_err, divisors_err = extract_delta_primes_(delta_primes, debug=debug, delta_prime_errs=delta_prime_err, couple_reals=couple_reals)
    elif not (delta_prime_perr is None): #Assume pest3 since Delta_prime_perr is present
        delta_prime_single_helicity, delta_prime_eff, delta_prime_nn_eff, delta_prime_2nn_eff, divisors, delta_prime_single_helicity_perr, delta_prime_eff_perr, delta_prime_nn_eff_perr, delta_prime_2nn_eff_perr, divisors_perr = extract_delta_primes_(delta_primes, debug=debug, delta_prime_errs=delta_prime_perr, couple_reals=couple_reals)
    elif not (delta_prime_err is None): #Assume STRIDE or RDCON since Delta_prime_perr is absent 
        delta_prime_single_helicity, delta_prime_eff, delta_prime_nn_eff, delta_prime_2nn_eff, divisors, delta_prime_single_helicity_err, delta_prime_eff_err, delta_prime_nn_eff_err, delta_prime_2nn_eff_err, divisors_err = extract_delta_primes_(delta_primes, debug=debug, delta_prime_errs=delta_prime_err, couple_reals=couple_reals)
    else: #Assume STRIDE or RDCON since Delta_prime_perr is absent
        delta_prime_single_helicity, delta_prime_eff, delta_prime_nn_eff, delta_prime_2nn_eff, divisors = extract_delta_primes_(delta_primes, debug=debug, couple_reals=couple_reals)
        
    #########################################################################################################
    # Check correct input data array type:
    #########################################################################################################
    if 'cmatch' in inputxr:
        copy_da = inputxr['cmatch'].copy()
    elif 'psi_n_rational' in inputxr:
        copy_da = inputxr['psi_n_rational'].copy()
    else: # Error!!
        raise ValueError("Input xarray must include 'cmatch' or 'psi_n_rational'.")

    #########################################################################################################
    # Split complex delta primes into real and imaginary parts, put them into the xarray using coordinate Delta_prime_type
    #########################################################################################################

    Dprime_1 = (delta_prime_single_helicity.real + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
    Dprime_2 = (delta_prime_nn_eff.real + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
    Dprime_3 = (delta_prime_2nn_eff.real + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
    Dprime_4 = (delta_prime_eff.real + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)

    ImDprime_1 = (delta_prime_single_helicity.imag + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
    ImDprime_2 = (delta_prime_nn_eff.imag + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
    ImDprime_3 = (delta_prime_2nn_eff.imag + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
    ImDprime_4 = (delta_prime_eff.imag + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)

    Dprime_1['Delta_prime_type'] = ['single helicity']
    Dprime_2['Delta_prime_type'] = ['nn coupled']
    Dprime_3['Delta_prime_type'] = ['2nn coupled']
    Dprime_4['Delta_prime_type'] = ['full coupled']
    
    ImDprime_1['Delta_prime_type'] = ['single helicity']
    ImDprime_2['Delta_prime_type'] = ['nn coupled']
    ImDprime_3['Delta_prime_type'] = ['2nn coupled']
    ImDprime_4['Delta_prime_type'] = ['full coupled']
    
    # Combine into one DataArray using coordinate Delta_prime_type
    Dprime_all_da = xr.concat(
        [Dprime_1, Dprime_2, Dprime_3, Dprime_4],
        dim='Delta_prime_type'
    )
    Dprime_all_da = Dprime_all_da.assign_coords(Delta_prime_type=['single helicity', 'nn coupled', '2nn coupled', 'full coupled'])

    Im_Dprime_all_da = xr.concat(
        [ImDprime_1, ImDprime_2, ImDprime_3, ImDprime_4],
        dim='Delta_prime_type'
    )
    Im_Dprime_all_da = Im_Dprime_all_da.assign_coords(Delta_prime_type=['single helicity', 'nn coupled', '2nn coupled', 'full coupled'])

    # Add these data arrays to the dataset that is inputxr:
    inputxr = inputxr.assign(
        Delta_prime_surf=Dprime_all_da,
        Im_Delta_prime_surf=Im_Dprime_all_da
    )
    inputxr = inputxr.assign(Delta_prime_ImRatio=np.abs(inputxr['Im_Delta_prime_surf']/inputxr['Delta_prime_surf']))

    #########################################################################################################
    # Put (real) delta prime errors into the xarray using coordinate Delta_prime_type
    #########################################################################################################

    if not (delta_prime_single_helicity_err is None):
        Dprime_err_1 = (delta_prime_single_helicity_err + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
        Dprime_err_2 = (delta_prime_nn_eff_err + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
        Dprime_err_3 = (delta_prime_2nn_eff_err + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
        Dprime_err_4 = (delta_prime_eff_err + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)


        Dprime_err_1['Delta_prime_type'] = ['single helicity']
        Dprime_err_2['Delta_prime_type'] = ['nn coupled']
        Dprime_err_3['Delta_prime_type'] = ['2nn coupled']
        Dprime_err_4['Delta_prime_type'] = ['full coupled']

        Dprime_err_all_da = xr.concat(
            [Dprime_err_1, Dprime_err_2, Dprime_err_3, Dprime_err_4],
            dim='Delta_prime_type'
        )
        Dprime_err_all_da = Dprime_err_all_da.assign_coords(Delta_prime_type=['single helicity', 'nn coupled', '2nn coupled', 'full coupled'])

        # Add these data arrays to the dataset that is inputxr:
        inputxr = inputxr.assign(
            Delta_prime_err_surf=Dprime_err_all_da
        )

    #########################################################################################################
    # Put (real) delta prime pest3 errors into the xarray using coordinate Delta_prime_type
    #########################################################################################################

    if not (delta_prime_single_helicity_perr is None):
        Dprime_perr_1 = (delta_prime_single_helicity_perr + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
        Dprime_perr_2 = (delta_prime_nn_eff_perr + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
        Dprime_perr_3 = (delta_prime_2nn_eff_perr + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)
        Dprime_perr_4 = (delta_prime_eff_perr + 0.0 * copy_da).expand_dims('Delta_prime_type', axis=0)


        Dprime_perr_1['Delta_prime_type'] = ['single helicity']
        Dprime_perr_2['Delta_prime_type'] = ['nn coupled']
        Dprime_perr_3['Delta_prime_type'] = ['2nn coupled']
        Dprime_perr_4['Delta_prime_type'] = ['full coupled']


        Dprime_perr_all_da = xr.concat(
            [Dprime_perr_1, Dprime_perr_2, Dprime_perr_3, Dprime_perr_4],
            dim='Delta_prime_type'
        )
        Dprime_perr_all_da = Dprime_perr_all_da.assign_coords(Delta_prime_type=['single helicity', 'nn coupled', '2nn coupled', 'full coupled'])

        # Add these data arrays to the dataset that is inputxr:
        inputxr = inputxr.assign(
            Delta_prime_perr_surf=Dprime_perr_all_da
        )

    return inputxr

def extract_delta_primes_(delta_primes,debug=False, delta_prime_errs=None, couple_reals=True):
    """
    Calculate delta' values using the outer ideal mode coupling generalisation from Brennan & Sugiyama PoP 2006.
    Applying these delta' values to a singular surface invokes the assumption that there is only one 
    resistive surface in the plasma (the surface of interest). If you use complex values as inputs for delta_primes,
    will provide complex outputs. Optional (complex or real) input delta_prime_errs will be propagated through the coupling equations, 
    however only the real component of the propagated errors is outputted.

    Parameters
    ----------
    delta_primes : numpy.ndarray
        A square matrix representing delta' values, must be numpy array.
    couple_reals : Bool
        Do we couple using the whole complex Delta prime? Or just the real component as in Brennan & Sugiyama PoP 2006?
        Recommended - stick to couple_reals = True (seems like calculation isn't propagating the complex components...)
    Returns
    -------
    delta_prime_single_helicity : numpy.ndarray
        The diagonal elements of the input matrix, representing single helicity delta' values.
    delta_prime_eff : numpy.ndarray
        The delta' values modified by coupling across the whole matrix.
    delta_prime_nn_eff : numpy.ndarray
        The delta' values modified by coupling with the first nearest neighbour surfaces. 
    delta_prime_2nn_eff : numpy.ndarray
        The delta' values modified by coupling with the first and second nearest neighbour surfaces.
    """

    #########################################################################################################
    # check delta_primes inputs:
    #########################################################################################################

    delta_primes=np.squeeze(delta_primes) # Ensure input is 2D
    # Check if input is a numpy array:
    if (not isinstance(delta_primes, np.ndarray)):
        raise TypeError("Input must be a numpy array.")
    delta_primes = jnp.array(delta_primes) # Turn input into 

    # Check if matrix is square:
    if delta_primes.shape[0] != delta_primes.shape[1]:
        print(delta_primes)
        raise ValueError("Input xarray must be a square matrix.")

    # Check if matrix is symmetric:
    #if not np.allclose(delta_primes, delta_primes.T):
    #    print(delta_primes)
    #    raise ValueError("Input delta prime xarray should be a symmetric matrix.")

    #########################################################################################################
    # check delta_prime_errs inputs:
    #########################################################################################################

    run_errs = False
    if not delta_prime_errs is None:
        delta_prime_errs = np.squeeze(delta_prime_errs)
        # Check if input is a numpy array:
        if (not isinstance(delta_prime_errs, np.ndarray)):
            raise TypeError("Input must be a numpy array.")

        # Check if errors are real or complex:
        assert np.isrealobj(delta_prime_errs), " Error propagation requires real errors. "
        delta_prime_errs = jnp.array(delta_prime_errs) # Turn input into JAX array

        # Check shape of delta_prime_errs is the same as delta_primes
        if delta_prime_errs.shape != delta_primes.shape:
            print(delta_prime_errs)
            print(delta_primes)
            raise ValueError("Input delta_prime_errs must have the same shape as delta_primes.")

        run_errs=True

    #########################################################################################################
    # if final few surfaces are nans, we cut them for the coupling calculation (then re-add them at the end)
    #########################################################################################################
    # Check how many nans are present at end of each row and column:
    if run_errs:
        delta_primes,nans_in_col1,delta_prime_errs = trim_nans(delta_primes,delta_prime_errs=delta_prime_errs)
    else:
        delta_primes,nans_in_col1,_ = trim_nans(delta_primes)

    #########################################################################################################
    # Do the calculation using sub functions to enable auto-diff
    #########################################################################################################
    # Extract key terms
    delta_prime_single_helicity = delta_prime_no_couple(delta_primes)
    if couple_reals:
        delta_prime_eff = delta_prime_full_couple(delta_primes.real)
        delta_prime_nn_eff = delta_prime_nn_couple(delta_primes.real)
        delta_prime_2nn_eff = delta_prime_2nn_couple(delta_primes.real)
        divisors = get_delta_prime_divisors(delta_primes.real)
    else:
        delta_prime_eff = delta_prime_full_couple(delta_primes)
        delta_prime_nn_eff = delta_prime_nn_couple(delta_primes)
        delta_prime_2nn_eff = delta_prime_2nn_couple(delta_primes)
        divisors = get_delta_prime_divisors(delta_primes)

    #########################################################################################################
    # Use auto-diff to get errors: following variance formula
    # see - https://en.wikipedia.org/wiki/Propagation_of_uncertainty#cite_note-9 with attached reference
    # doi:10.6028/jres.070c.025
    #########################################################################################################
    if run_errs:
        # Use only real component of delta_primes in error calc, otherwise auto-diff jacobian gets broken
        delta_prime_single_helicity_err, delta_prime_eff_err, delta_prime_nn_eff_err, delta_prime_2nn_eff_err, divisors_err = extract_variances(delta_primes.real, delta_prime_errs, debug=debug)

    #########################################################################################################
    # Debug!
    #########################################################################################################
    if debug:
        print("Delta prime single helicity:")
        print(delta_prime_single_helicity)
        print("Delta prime w. coupling of all surfaces:", delta_prime_eff)
        print("Delta prime w. coupling to nearest neighbour(s):", delta_prime_nn_eff)
        print("Delta prime w. coupling to first and second nearest neighbour(s):", delta_prime_2nn_eff)
        print("Divisors:", divisors)
        if run_errs:
            print(" delta_primes,", delta_primes)
            print(" delta_prime_errs,", delta_prime_errs) 
            print(" delta_prime_single_helicity_err,", delta_prime_single_helicity_err)
            print(" delta_prime_eff_err", delta_prime_eff_err)
            print(" delta_prime_nn_eff_err", delta_prime_nn_eff_err)
            print(" delta_prime_2nn_eff_err", delta_prime_2nn_eff_err)
            print(" Divisors err", divisors_err)

    #########################################################################################################
    # Append nans nans to the end of each array to make them the same length as the input matrix:
    #########################################################################################################

    if nans_in_col1 > 0:
        nanvec = jnp.full((nans_in_col1,), jnp.nan)
        delta_prime_single_helicity = jnp.hstack((delta_prime_single_helicity, nanvec))
        delta_prime_eff = jnp.hstack((delta_prime_eff, nanvec))
        delta_prime_nn_eff = jnp.hstack((delta_prime_nn_eff, nanvec))
        delta_prime_2nn_eff = jnp.hstack((delta_prime_2nn_eff, nanvec))
        divisors = jnp.hstack((divisors, nanvec))
        if run_errs:
            delta_prime_single_helicity_err = jnp.hstack((delta_prime_single_helicity_err, nanvec))
            delta_prime_eff_err = jnp.hstack((delta_prime_eff_err, nanvec))
            delta_prime_nn_eff_err = jnp.hstack((delta_prime_nn_eff_err, nanvec))
            delta_prime_2nn_eff_err = jnp.hstack((delta_prime_2nn_eff_err, nanvec))
            divisors_err = jnp.hstack((divisors_err, nanvec))

    if run_errs:
        return np.array(delta_prime_single_helicity), np.array(delta_prime_eff), np.array(delta_prime_nn_eff), np.array(delta_prime_2nn_eff), np.array(divisors), np.array(delta_prime_single_helicity_err), np.array(delta_prime_eff_err), np.array(delta_prime_nn_eff_err), np.array(delta_prime_2nn_eff_err), np.array(divisors_err)

    return np.array(delta_prime_single_helicity), np.array(delta_prime_eff), np.array(delta_prime_nn_eff), np.array(delta_prime_2nn_eff), np.array(divisors)

def delta_prime_no_couple(delta_primes):
    """ Uncoupled delta' values. """
    return delta_primes.diagonal()

def matrix_cofactor(matrix):
    return jnp.linalg.inv(matrix).T * jnp.linalg.det(matrix)

def delta_prime_full_couple(delta_primes):
    """ Delta' values modified by coupling across the whole matrix. """
    delta_prime_single_helicity = delta_primes.diagonal()
    Mcof = matrix_cofactor(delta_primes)
    # Extract coupled elements
    delta_prime_mod = [] # Delta prime modifier due to coupling of all surfaces
    for i in range(delta_primes.shape[0]):
        divisor = Mcof[i, i]
        DP_coupled_i = 0 # Initialise coupled delta_prime to nothing
        for j in range(delta_primes.shape[0]):
            if i != j:
                DP_coupled_i += delta_primes[i, j]*Mcof[i, j]
        delta_prime_mod.append(DP_coupled_i / divisor)
    # To get coupled delta prime values, add modifiers to single helicity values:
    delta_prime_mod = jnp.array(delta_prime_mod).astype(float)
    delta_prime_eff = delta_prime_single_helicity + jnp.array(delta_prime_mod)
    return jnp.array(delta_prime_eff)

def delta_prime_nn_couple(delta_primes):
    """ Delta' values modified by coupling with the first nearest neighbour surfaces. """
    delta_prime_single_helicity = delta_primes.diagonal()
    Mcof = matrix_cofactor(delta_primes)
    # Extract coupled elements
    delta_prime_nn_mod = [] # Delta prime modifier due to coupling of all surfaces
    for i in range(delta_primes.shape[0]):
        divisor = Mcof[i, i]
        DP_coupled_i_nn = 0
        for j in range(delta_primes.shape[0]):
            if abs(i - j) == 1:
                DP_coupled_i_nn += delta_primes[i, j]*Mcof[i, j]
        delta_prime_nn_mod.append(DP_coupled_i_nn / divisor)
    delta_prime_nn_mod = jnp.array(delta_prime_nn_mod).astype(float)
    # To get coupled delta prime values, add modifiers to single helicity values:
    delta_prime_nn_eff = delta_prime_single_helicity + delta_prime_nn_mod 
    return jnp.array(delta_prime_nn_eff)

def delta_prime_2nn_couple(delta_primes):
    """ Delta' values modified by coupling with the first and second nearest neighbour surfaces. """
    delta_prime_single_helicity = delta_primes.diagonal()
    Mcof = matrix_cofactor(delta_primes)
    # Extract coupled elements
    delta_prime_2nn_mod = [] # Delta prime modifier due to coupling to first and second nearest neighbour(s)
    for i in range(delta_primes.shape[0]):
        divisor = Mcof[i, i]
        DP_coupled_i_2nn = 0
        for j in range(delta_primes.shape[0]):
            if abs(i - j) == 1 or abs(i - j) == 2:
                DP_coupled_i_2nn += delta_primes[i, j]*Mcof[i, j]
        delta_prime_2nn_mod.append(DP_coupled_i_2nn / divisor)
    delta_prime_2nn_mod = jnp.array(delta_prime_2nn_mod).astype(float)
    # To get coupled delta prime values, add modifiers to single helicity values:
    delta_prime_2nn_eff = delta_prime_single_helicity + delta_prime_2nn_mod 
    return jnp.array(delta_prime_2nn_eff)

def get_delta_prime_divisors(delta_primes):
    """ Delta' division values used in matrix coupling. """
    Mcof = matrix_cofactor(delta_primes)
    assert delta_primes.shape == Mcof.shape, "Cofactor matrix has different shape than input matrix."
    divisors = Mcof.diagonal()
    divisors = jnp.array(divisors).astype(float)
    return jnp.array(divisors)

def extract_variances(delta_primes,delta_prime_errs,debug=False):
    """ Returns propagated errors of delta primes using variance formula 
        (see attached reference doi:10.6028/jres.070c.025 
        to https://en.wikipedia.org/wiki/Propagation_of_uncertainty#cite_note-9)
        Uses jacobians of Delta prime coupling functions calculated using Jax.
    """

    # Check delta_prime_errs is real:
    assert np.isrealobj(delta_prime_errs), "Delta_prime_errs must be real array."
    if not np.isrealobj(delta_primes):
        delta_primes=delta_primes.real
        print("Only the real component of Delta_prime can be used for error propagation.")


    # Jacobians of coupling functions, evaluated at Delta Prime (Delta prime input has shape m, m).
    # The coupling functions output vectors of length k (= m). Thus the jacobians have shape k, m, m (= m, m, m).
    delta_prime_single_helicity_jac = jacfwd(delta_prime_no_couple)(delta_primes)
    delta_prime_eff_jac = jacfwd(delta_prime_full_couple)(delta_primes)
    delta_prime_nn_eff_jac = jacfwd(delta_prime_nn_couple)(delta_primes)
    delta_prime_2nn_eff_jac = jacfwd(delta_prime_2nn_couple)(delta_primes)
    divisors_jac = jacfwd(get_delta_prime_divisors)(delta_primes)

    # Squaring partial derivative values: These values have dimensions k, m, m (= m, m, m).
    delta_prime_single_helicity_jacsq = delta_prime_single_helicity_jac*delta_prime_single_helicity_jac
    delta_prime_eff_jacsq = delta_prime_eff_jac*delta_prime_eff_jac
    delta_prime_nn_eff_jacsq = delta_prime_nn_eff_jac*delta_prime_nn_eff_jac
    delta_prime_2nn_eff_jacsq = delta_prime_2nn_eff_jac*delta_prime_2nn_eff_jac
    divisors_jacsq = divisors_jac*divisors_jac
    assert delta_prime_single_helicity_jacsq.shape == delta_prime_eff_jacsq.shape == delta_prime_nn_eff_jacsq.shape == delta_prime_2nn_eff_jacsq.shape == divisors_jacsq.shape

    # Squaring errs: These values have dimensions m, m
    delta_prime_errs_sq = delta_prime_errs*delta_prime_errs

    # Debug dump
    if debug:
        print("Debugging extract_variances:")
        print("delta_prime_single_helicity_jacsq:", delta_prime_single_helicity_jacsq)
        print("delta_prime_eff_jacsq:", delta_prime_eff_jacsq)
        print("delta_prime_nn_eff_jacsq:", delta_prime_nn_eff_jacsq)
        print("delta_prime_2nn_eff_jacsq:", delta_prime_2nn_eff_jacsq)
        print("divisors_jacsq:", divisors_jacsq)
        print("delta_prime_errs_sq:", delta_prime_errs_sq)

    #Initialise arrays of errs for variance formula output
    delta_prime_single_helicity_err = []
    delta_prime_eff_err = []
    delta_prime_nn_eff_err = []
    delta_prime_2nn_eff_err = []
    divisors_err = []
    for k in range(delta_prime_single_helicity_jacsq.shape[0]):
        # Evaluating variance formula for each output of the coupling functions:
        delta_prime_single_helicity_errk =  jnp.sqrt(jnp.sum(delta_prime_single_helicity_jacsq[k,:,:]* delta_prime_errs_sq))
        delta_prime_eff_errk =              jnp.sqrt(jnp.sum(delta_prime_eff_jacsq[k,:,:]            * delta_prime_errs_sq))
        delta_prime_nn_eff_errk =           jnp.sqrt(jnp.sum(delta_prime_nn_eff_jacsq[k,:,:]         * delta_prime_errs_sq))
        delta_prime_2nn_eff_errk =          jnp.sqrt(jnp.sum(delta_prime_2nn_eff_jacsq[k,:,:]        * delta_prime_errs_sq))
        divisors_errk =                     jnp.sqrt(jnp.sum(divisors_jacsq[k,:,:]                   * delta_prime_errs_sq))

        # Put values inout lists:
        delta_prime_single_helicity_err.append(delta_prime_single_helicity_errk)
        delta_prime_eff_err.append(delta_prime_eff_errk)
        delta_prime_nn_eff_err.append(delta_prime_nn_eff_errk)
        delta_prime_2nn_eff_err.append(delta_prime_2nn_eff_errk)
        divisors_err.append(divisors_errk)

    # Check np.array(delta_prime_single_helicity_err) is close to delta_prime_errs.diagonal():
    if not np.allclose(np.array(delta_prime_single_helicity_err), delta_prime_errs.diagonal()):
        print("Error propagation function isn't working, debug!")

    return np.array(delta_prime_single_helicity_err), np.array(delta_prime_eff_err), np.array(delta_prime_nn_eff_err), np.array(delta_prime_2nn_eff_err), np.array(divisors_err)


""" Deprecated old function:
    #########################################################################################################
    # Do the calculation, using sub functions to enable auto-diff
    #########################################################################################################
    # Extract diagonal elements:
    delta_prime_single_helicity = delta_primes.diagonal()

    delta_prime_eff = delta_prime_full_couple(delta_primes)
    delta_prime_nn_eff = delta_prime_nn_couple(delta_primes)
    delta_prime_2nn_eff = delta_prime_2nn_couple(delta_primes)

    M = Matrix(delta_primes)
    Mcof = matrix_cofactor(delta_primes)
    assert M.shape == Mcof.shape, "Cofactor matrix has different shape than input matrix."
    # Extract coupled elements
    delta_prime_mod = [] # Delta prime modifier due to coupling of all surfaces
    delta_prime_nn_mod = [] # Delta prime modifier due to couplin to nearest neighbour(s) 
    delta_prime_2nn_mod = [] # Delta prime modifier due to coupling to first and second nearest neighbour(s)
    divisors = []
    for i in range(M.shape[0]):
        DP_coupled_i = 0 # Initialise coupled delta_prime to nothing
        DP_coupled_i_nn = 0
        DP_coupled_i_2nn = 0
        divisor = Mcof[i, i]
        for j in range(M.shape[0]):
            if i != j:
                DP_coupled_i += M[i, j]*Mcof[i, j]
                #Check if difference of i and j is less than or equal to 1:
                if abs(i - j) == 1:
                    DP_coupled_i_nn += M[i, j]*Mcof[i, j]
                    DP_coupled_i_2nn += M[i, j]*Mcof[i, j]
                elif abs(i - j) == 2:
                    DP_coupled_i_2nn += M[i, j]*Mcof[i, j]
        delta_prime_mod.append(DP_coupled_i / divisor)
        delta_prime_nn_mod.append(DP_coupled_i_nn / divisor)
        delta_prime_2nn_mod.append(DP_coupled_i_2nn / divisor)
        divisors.append(divisor)

    # To get coupled delta prime values, add modifiers to single helicity values:
    delta_prime_eff = delta_prime_single_helicity + delta_prime_mod    
    delta_prime_nn_eff = delta_prime_single_helicity + delta_prime_nn_mod
    delta_prime_2nn_eff = delta_prime_single_helicity + delta_prime_2nn_mod
"""