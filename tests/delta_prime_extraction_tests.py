# Scripts to test the functions in delta_prime_extraction.py 

import math
import xarray as xr
import sympy
import numpy as np
import jax.numpy as jnp
from jax import jacfwd
import os
import shutil
import pandas as pd
import pickle as pkl

from tearing_physics_suite.environment import home_dir

os.chdir(home_dir)

from tearing_physics_suite.fortran_wrappers import run_resistive_calculation
from tearing_physics_suite.fortran_wrappers import compile_xarrays
from tearing_physics_suite.utils import trim_nans
from tearing_physics_suite.delta_prime_extraction import extract_delta_primes_, extract_delta_primes, delta_prime_no_couple, delta_prime_full_couple, delta_prime_nn_couple, delta_prime_2nn_couple, get_delta_prime_divisors,extract_variances
from tearing_physics_suite.tearing_physics_suite import linear_resistive_calculation

os.chdir(home_dir)

#########################################################################################################
# user settings:
#########################################################################################################

use_default_eq=True
fast=False # <- Don't change this
test_extract_delta_primes_ = True
test_extract_delta_primes  = True
test_surface_coupling=True

#########################################################################################################
# load equilibrium:
#########################################################################################################

# Choose equilibrium file
default_equilibrium =  os.path.join(home_dir, 'submodules/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT')

if use_default_eq:
    eq_filename = default_equilibrium

print(" Getting equilibrium file from ", eq_filename)
eq_filename_short= eq_filename.split('/')[-1]

#########################################################################################################
# Run run_resistive_calculation to get an inputxr with Delta prime
#########################################################################################################
if fast:
    rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict=run_resistive_calculation(
        eq_filename, 1,
        run_stride=False,
        run_pest3=False,
        vac_flag='f',
        ode_flag='f',
        etol=1e-7,
        nx=64,
        mpsi=128,
        mtheta=129
    )
else:
    rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict=run_resistive_calculation(
        eq_filename, 1,
        pest_match_truncation=False,
        run_stride=False,
        run_pest3=True,
        vac_flag='f',
        ode_flag='f',
        etol=1e-7,
        nx=64,
        mpsi=128,
        mtheta=129
    )

if not pest3_xr is None:
    if "Delta_prime_perr" in pest3_xr:
        delta_prime_perr = pest3_xr["Delta_prime_perr"].values
    else:
        delta_prime_perr = None
else:
    delta_prime_perr = None

if fast:
    inputxr = rdcon_xr
else:
    inputxr = pest3_xr

# Check if Delta_prime is in inputxr:
assert "Delta_prime" in inputxr, "Input xarray must contain 'Delta_prime' variable."
delta_prime_real_component = inputxr["Delta_prime"].sel(i=0).values
delta_prime_im_component = inputxr["Delta_prime"].sel(i=1).values
# Make a combined delta_prime with its real and complex values together:
delta_primes = delta_prime_real_component + 1j * delta_prime_im_component 
# Check delta_prime is of type complex
assert np.iscomplexobj(delta_primes), "Delta_prime must be a complex array."

if not fast:
    delta_primes, nans_in_col1, delta_prime_perr = trim_nans(delta_primes,delta_prime_perr)

#########################################################################################################
# Test delta_prime_no_couple, delta_prime_full_couple, delta_prime_nn_couple, delta_prime_2nn_couple, get_delta_prime_divisors
#########################################################################################################

try: 
    if test_surface_coupling:
        dpncR = delta_prime_no_couple(delta_primes.real)
        dpfcR = delta_prime_full_couple(delta_primes.real)
        dpnnR = delta_prime_nn_couple(delta_primes.real)
        dp2nnR = delta_prime_2nn_couple(delta_primes.real)
        dpnc = delta_prime_no_couple(delta_primes)
        dpfc = delta_prime_full_couple(delta_primes)
        dpnn = delta_prime_nn_couple(delta_primes)
        dp2nn = delta_prime_2nn_couple(delta_primes)
        if True:
            print(dpncR[0:2])
            print(dpnnR[0:2])
            print(dp2nnR[0:2])
            print(dpfcR[0:2])
            print(dpnc[0:2])
            print(dpnn[0:2])
            print(dp2nn[0:2])
            print(dpfc[0:2])
        if True:
            print(dpncR[2:4])
            print(dpnnR[2:4])
            print(dp2nnR[2:4])
            print(dpfcR[2:4])
            print(dpnc[2:4])
            print(dpnn[2:4])
            print(dp2nn[2:4])
            print(dpfc[2:4])
        if True:
            print(dpncR[4:6])
            print(dpnnR[4:6])
            print(dp2nnR[4:6])
            print(dpfcR[4:6])
            print(dpnc[4:6])
            print(dpnn[4:6])
            print(dp2nn[4:6])
            print(dpfc[4:6])
    #########################################################################################################
    # Test extract_delta_primes_ without errors:
    #########################################################################################################
    if test_extract_delta_primes_:
        # Try real:
        delta_prime_single_helicity, delta_prime_effR, delta_prime_nn_effR, delta_prime_2nn_effR, divisorsR = extract_delta_primes_(delta_primes)
        delta_prime_single_helicity, delta_prime_eff, delta_prime_nn_eff, delta_prime_2nn_eff, divisors = extract_delta_primes_(delta_primes,couple_reals=False)
    #########################################################################################################
    # Test extract_variances with errors:
    #########################################################################################################   
        dpsherr, dpefferr, dpeffnnerr, dpeff2nnerr, divierr = extract_variances(delta_primes,delta_prime_perr,debug=True)
    #########################################################################################################
    # Test extract_delta_primes_ with errors:
    #########################################################################################################
        debug=True
        delta_prime_single_helicity, delta_prime_eff, delta_prime_nn_eff, delta_prime_2nn_eff, divisors, delta_prime_single_helicity_perr, delta_prime_eff_perr, delta_prime_nn_eff_perr, delta_prime_2nn_eff_perr, divisors_perr = extract_delta_primes_(delta_primes, debug=debug, delta_prime_errs=delta_prime_perr)
    #########################################################################################################
    # Test extract_delta_primes_ with errors:
    #########################################################################################################
    if test_extract_delta_primes:
        rdcon_xr_w_delta_primes = extract_delta_primes(rdcon_xr,couple_reals=True)
        pest3_xr_w_delta_primes = extract_delta_primes(pest3_xr,couple_reals=True)
    print(" Ran through all delta_prime_extraction_tests without throwing an error... ")
except Exception as e:
    print(" Error snagged during delta_prime_extraction_tests: ")
    raise e