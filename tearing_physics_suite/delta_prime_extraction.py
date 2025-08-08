# Scripts to calculate effective delta' values using generalisation
# from Brennan & Sugiyama 2006.

import math
import xarray as xr
from sympy import Matrix
import sympy
import numpy as np

def delta_primes(delta_prime_xr,debug=False):
    """
    Calculate delta' values using the outer ideal mode coupling generalisation from Brennan & Sugiyama PoP 2006.
    Applying these delta' values to a singular surface invokes the assumption that there is only one 
    resistive surface in the plasma (the surface of interest). 
    Parameters
    ----------
    delta_prime_xr : numpy.ndarray
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
    # check inputs:
    #########################################################################################################

    delta_prime_xr=np.squeeze(delta_prime_xr) # Ensure input is 2D
    # Check if input is a numpy array:
    if (not isinstance(delta_prime_xr, np.ndarray)):
        raise TypeError("Input must be a numpy array.")
    # Check if matrix is square:
    if delta_prime_xr.shape[0] != delta_prime_xr.shape[1]:
        print(delta_prime_xr)
        raise ValueError("Input xarray must be a square matrix.")

    # Check if matrix is symmetric:
    #if not np.allclose(delta_prime_xr, delta_prime_xr.T):
    #    print(delta_prime_xr)
    #    raise ValueError("Input delta prime xarray should be a symmetric matrix.")

    #########################################################################################################
    # if final few surfaces are nans, we cut them for the coupling calculation (then re-add them at the end)
    #########################################################################################################
    # Check how many nans are present at end of each row and column:
    dp_col1 = delta_prime_xr[:,0]
    nans_in_col1=0
    for i in dp_col1:
        if math.isnan(i):
            nans_in_col1+=1
    dp_row1 = delta_prime_xr[0,:]
    nans_in_row1=0
    for i in dp_row1:
        if math.isnan(i):
            nans_in_row1+=1
    assert nans_in_col1 == nans_in_row1, "Input matrix has inconsistent number of nans in rows and columns."
    #We cut the matrix to remove nans:
    if nans_in_col1 > 0:
        delta_prime_xr = delta_prime_xr[:-nans_in_col1,:-nans_in_row1]
    # Now delta_prime_xr is a square matrix with no nans. Check no nans present:
    if np.isnan(delta_prime_xr).any():
        print(delta_prime_xr)
        raise ValueError("Input matrix has interior nans present. Error somewhere in truncation")

    #########################################################################################################
    # Do the calculation
    #########################################################################################################
    # Extract diagonal elements:
    delta_prime_single_helicity = delta_prime_xr.diagonal()
    M = Matrix(delta_prime_xr)
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

    #########################################################################################################
    # Debug!
    #########################################################################################################
    if debug:
        print("Matrix M:\n",sympy.matrix2numpy(M))
        print("Cofactor matrix Mcof:\n",sympy.matrix2numpy(Mcof))
        print("Delta prime single helicity:")
        print(delta_prime_single_helicity)
        print("Delta prime modifier due to coupling of all surfaces:", delta_prime_mod)
        print("Delta prime modifier due to coupling to nearest neighbour(s):", delta_prime_nn_mod)
        print("Delta prime modifier due to coupling to first and second nearest neighbour(s):", delta_prime_2nn_mod)
        print("Divisors:", divisors)
        return M

    #########################################################################################################
    # Append nans nans to the end of each array to make them the same length as the input matrix:
    #########################################################################################################
    if nans_in_col1 > 0:
        delta_prime_single_helicity = np.append(delta_prime_single_helicity, [np.nan]*nans_in_col1)
        delta_prime_eff = np.append(delta_prime_eff, [np.nan]*nans_in_col1)
        delta_prime_nn_eff = np.append(delta_prime_nn_eff, [np.nan]*nans_in_col1)
        delta_prime_2nn_eff = np.append(delta_prime_2nn_eff, [np.nan]*nans_in_col1)
        divisors = np.append(divisors, [np.nan]*nans_in_col1)
    
    return delta_prime_single_helicity.astype(float), delta_prime_eff.astype(float), delta_prime_nn_eff.astype(float), delta_prime_2nn_eff.astype(float), np.array(divisors).astype(float)