# Tests for parallelised multi-run and multi-compile functions

from scipy.interpolate import CubicSpline
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
from tearing_physics_suite.mre_analysis import analyse_with_mre
from tearing_physics_suite.profile_read import read_IDA_lite
from tearing_physics_suite.multi_run import multi_run_, multi_compile

os.chdir(home_dir)

run_test = False
run_compile = True

if __name__ == '__main__':

    #########################################################################################################
    # user settings:
    #########################################################################################################

    fast=False # <- Don't change this
    use_default_eq=True
    run_resist=False

    #########################################################################################################
    # load equilibrium:
    #########################################################################################################

    # Choose equilibrium file
    default_equilibrium =  os.path.join(home_dir, 'submodules/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT')
    eq_filename = default_equilibrium

    print(" Getting equilibrium file from ", eq_filename)
    eq_filename_short= eq_filename.split('/')[-1]

    #########################################################################################################
    # Read rotation_.cdf
    #########################################################################################################

    rotation_cdf_path = os.path.join(home_dir, 'tests', 'rotation_.cdf')

    if os.path.exists(rotation_cdf_path):
        profile_dict = read_IDA_lite(rotation_cdf_path, verbose=True, time_idx=100)

    if os.path.exists(rotation_cdf_path):
        splines_by_time_dict_list = read_IDA_lite(rotation_cdf_path, verbose=False, time_idx=None)

    #########################################################################################################
    # Run nonlinear_resistive_calculation
    #########################################################################################################

    splines_by_time_dict_list_first_20 = splines_by_time_dict_list[100:106]
    eq_filenames = [eq_filename]*len(splines_by_time_dict_list_first_20)
    master_working_dir = os.path.join(home_dir, 'tests', 'test_working_dir_parallel')
    os.makedirs(master_working_dir, exist_ok=True)

    if run_test:
        combined_xr_list, input_dict_list, errors = multi_run_(eq_filenames, splines_by_time_dict_list_first_20,
            master_working_dir,
            Zeff=1.5,
            average_ion_mass=2.5,
            q_surfs_of_interest=[1.5,2.0],
            psi_surfs_of_interest=[0.3,0.95],
            energy_confinement_time=0.12,
            debug_global_mre_quantities=False,
            nvec=[1,2],
            run_stride=True,
            run_pest3=True,
            vac_flag='f',
            ode_flag='f',
            etol=1e-7,
            nx=64,
            mpsi=128,
            mtheta=129,
            fail_fast=True
            )

    if run_compile:
        # Run the bloody compiler:
        multi_compile(eq_filenames, master_working_dir, debug=True)