# Scripts to calculate effective delta' values using generalisation
# from Brennan & Sugiyama 2006.

import math
import xarray as xr
from sympy import Matrix
import sympy
import numpy as np
import jax.numpy as jnp
from jax import jacfwd

def delta_primes(delta_primes,debug=False, delta_prime_errs=None):
    """
    Calculate delta' values using the outer ideal mode coupling generalisation from Brennan & Sugiyama PoP 2006.
    Applying these delta' values to a singular surface invokes the assumption that there is only one 
    resistive surface in the plasma (the surface of interest). If you use complex values as inputs for delta_primes,
    should automatically provide complex outputs allowing you to evaluate the complex/real ratio of the output delta_primes.

    Parameters
    ----------
    delta_primes : numpy.ndarray
        A square matrix representing delta' values, must be numpy array.
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
        # Check if input is a numpy array:
        if (not isinstance(delta_prime_errs, np.ndarray)):
            raise TypeError("Input must be a numpy array.")
        delta_prime_errs = jnp.array(delta_prime_errs) # Turn input into JAX array

        # Check shape of delta_prime_errs is the same as delta_primes
        if delta_prime_errs.shape != delta_primes.shape:
            raise ValueError("Input delta_prime_errs must have the same shape as delta_primes.")

        run_errs=True

    #########################################################################################################
    # if final few surfaces are nans, we cut them for the coupling calculation (then re-add them at the end)
    #########################################################################################################
    # Check how many nans are present at end of each row and column:
    dp_col1 = delta_primes[:,0]
    nans_in_col1=0
    for i in dp_col1:
        if math.isnan(i):
            nans_in_col1+=1
    dp_row1 = delta_primes[0,:]
    nans_in_row1=0
    for i in dp_row1:
        if math.isnan(i):
            nans_in_row1+=1
    assert nans_in_col1 == nans_in_row1, "Input matrix has inconsistent number of nans in rows and columns."
    #We cut the matrix to remove nans:
    if nans_in_col1 > 0:
        delta_primes = delta_primes[:-nans_in_col1,:-nans_in_row1]
        if run_errs:
            delta_prime_errs = delta_prime_errs[:-nans_in_col1,:-nans_in_row1]
    # Now delta_primes is a square matrix with no nans. Check no nans present:
    if np.isnan(delta_primes).any():
        print(delta_primes)
        raise ValueError("Input matrix has interior nans present. Error somewhere in truncation")
    if run_errs and np.isnan(delta_prime_errs).any():
        print(delta_prime_errs)
        raise ValueError("Input matrix has interior nans present. Error somewhere in truncation")

    #########################################################################################################
    # Do the calculation using sub functions to enable auto-diff
    #########################################################################################################
    # Extract key terms
    delta_prime_single_helicity = delta_prime_no_couple(delta_primes)
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
        delta_prime_single_helicity_err, delta_prime_eff_err, delta_prime_nn_eff_err, delta_prime_2nn_eff_err, divisors_err = extract_variances(delta_primes, delta_prime_errs, debug=debug)

    #########################################################################################################
    # Debug!
    #########################################################################################################
    if debug:
        print("Matrix M:\n",sympy.matrix2numpy(M))
        print("Cofactor matrix Mcof:\n",sympy.matrix2numpy(Mcof))
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
        return M

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

    return delta_prime_single_helicity, delta_prime_eff, delta_prime_nn_eff, delta_prime_2nn_eff, divisors

def delta_prime_no_couple(delta_primes: jnp.array) -> jnp.array:
    """ Uncoupled delta' values. """
    return delta_primes.diagonal()

def delta_prime_full_couple(delta_primes: jnp.array) -> jnp.array:
    """ Delta' values modified by coupling across the whole matrix. """
    delta_prime_single_helicity = delta_primes.diagonal()
    M = Matrix(delta_primes)
    Mcof = M.cofactor_matrix()
    # Extract coupled elements
    delta_prime_mod = [] # Delta prime modifier due to coupling of all surfaces
    for i in range(M.shape[0]):
        divisor = Mcof[i, i]
        DP_coupled_i = 0 # Initialise coupled delta_prime to nothing
        for j in range(M.shape[0]):
            if i != j:
                DP_coupled_i += M[i, j]*Mcof[i, j]
        delta_prime_mod.append(DP_coupled_i / divisor)
    # To get coupled delta prime values, add modifiers to single helicity values:
    delta_prime_eff = delta_prime_single_helicity + delta_prime_mod 
    return jnp.array(delta_prime_eff)

def delta_prime_nn_couple(delta_primes: jnp.array) -> jnp.array:
    """ Delta' values modified by coupling with the first nearest neighbour surfaces. """
    delta_prime_single_helicity = delta_primes.diagonal()
    M = Matrix(delta_primes)
    Mcof = M.cofactor_matrix()
    # Extract coupled elements
    delta_prime_nn_mod = [] # Delta prime modifier due to coupling of all surfaces
    for i in range(M.shape[0]):
        divisor = Mcof[i, i]
        DP_coupled_i_nn = 0
        for j in range(M.shape[0]):
            if abs(i - j) == 1:
                DP_coupled_i_nn += M[i, j]*Mcof[i, j]
        delta_prime_nn_mod.append(DP_coupled_i_nn / divisor)
    # To get coupled delta prime values, add modifiers to single helicity values:
    delta_prime_nn_eff = delta_prime_single_helicity + delta_prime_nn_mod 
    return jnp.array(delta_prime_nn_eff)

def delta_prime_2nn_couple(delta_primes: jnp.array) -> jnp.array:
    """ Delta' values modified by coupling with the first and second nearest neighbour surfaces. """
    delta_prime_single_helicity = delta_primes.diagonal()
    M = Matrix(delta_primes)
    Mcof = M.cofactor_matrix()
    # Extract coupled elements
    delta_prime_2nn_mod = [] # Delta prime modifier due to coupling to first and second nearest neighbour(s)
    for i in range(M.shape[0]):
        divisor = Mcof[i, i]
        DP_coupled_i_2nn = 0
        for j in range(M.shape[0]):
            if abs(i - j) == 1 or abs(i - j) == 2:
                DP_coupled_i_2nn += M[i, j]*Mcof[i, j]
        delta_prime_2nn_mod.append(DP_coupled_i_2nn / divisor)
    # To get coupled delta prime values, add modifiers to single helicity values:
    delta_prime_2nn_eff = delta_prime_single_helicity + delta_prime_2nn_mod 
    return jnp.array(delta_prime_2nn_eff)

def get_delta_prime_divisors(delta_primes: jnp.array) -> jnp.array:
    """ Delta' division values used in matrix coupling. """
    M = Matrix(delta_primes)
    Mcof = M.cofactor_matrix()
    assert M.shape == Mcof.shape, "Cofactor matrix has different shape than input matrix."
    divisors = Mcof.diagonal()
    return jnp.array(divisors)

def extract_variances(delta_primes,delta_prime_errs,debug=False):
    """ Returns propagated errors of delta primes using variance formula 
        (see attached reference doi:10.6028/jres.070c.025 
        to https://en.wikipedia.org/wiki/Propagation_of_uncertainty#cite_note-9)
        Uses jacobians of Delta prime coupling functions calculated using Jax.
    """

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
    for k in range(len(delta_prime_single_helicity_jacsq.shape[0])):
        # Evaluating variance formula for each output of the coupling functions:
        delta_prime_single_helicity_errk =  jnp.sum(delta_prime_single_helicity_jacsq[k,:,:]* delta_prime_errs_sq)
        delta_prime_eff_errk =              jnp.sum(delta_prime_eff_jacsq[k,:,:]            * delta_prime_errs_sq)
        delta_prime_nn_eff_errk =           jnp.sum(delta_prime_nn_eff_jacsq[k,:,:]         * delta_prime_errs_sq)
        delta_prime_2nn_eff_errk =          jnp.sum(delta_prime_2nn_eff_jacsq[k,:,:]        * delta_prime_errs_sq)
        divisors_errk =                     jnp.sum(divisors_jacsq[k,:,:]                   * delta_prime_errs_sq)

        # Put values inout lists:
        delta_prime_single_helicity_err.append(delta_prime_single_helicity_errk)
        delta_prime_eff_err.append(delta_prime_eff_errk)
        delta_prime_nn_eff_err.append(delta_prime_nn_eff_errk)
        delta_prime_2nn_eff_err.append(delta_prime_2nn_eff_errk)
        divisors_err.append(divisors_errk)

    # Check np.array(delta_prime_single_helicity_err) is close to delta_prime_errs.diagonal():
    assert np.allclose(np.array(delta_prime_single_helicity_err), delta_prime_errs.diagonal()), "Error propagation function isn't working, debug!"

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
    Mcof = M.cofactor_matrix()
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