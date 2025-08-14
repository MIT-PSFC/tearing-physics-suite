# Scripts to test the functions in mre_analysis.py and cross_field_transport.py 

from scipy.interpolate import CubicSpline
import math
import xarray as xr
from sympy import Matrix
import sympy
import numpy as np
import jax.numpy as jnp
from jax import jacfwd
import os
import sys
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
import unittest
import pickle as pkl
import jax.numpy as jnp

from tearing_physics_suite.environment import home_dir

os.chdir(home_dir)

#from tearing_physics_suite.input_scans import scan_1D_input,extract_scanned_xrs
from tearing_physics_suite.fortran_wrappers import run_resistive_calculation
from tearing_physics_suite.fortran_wrappers import compile_xarrays
from tearing_physics_suite.utils import trim_nans
from tearing_physics_suite.delta_prime_extraction import extract_delta_primes_, extract_delta_primes, delta_prime_no_couple, delta_prime_full_couple, delta_prime_nn_couple, delta_prime_2nn_couple, get_delta_prime_divisors,extract_variances
from tearing_physics_suite.tearing_physics_suite import linear_resistive_calculation
from tearing_physics_suite.mre_analysis import analyse_with_mre

os.chdir(home_dir)

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

if use_default_eq:
    eq_filename = default_equilibrium
    profile_filename = eq_filename+'.kin'
    def get_default_profs(profile_filename):
        profile_data_names = pd.read_csv(profile_filename, sep='\s+',header=None,nrows=1)
        profile_data_xr = xr.Dataset(pd.read_csv(profile_filename, skiprows=1, sep='\s+', header=None,names=profile_data_names.iloc[0].values))
        te_keV_spline = CubicSpline(profile_data_xr['psi'].values, profile_data_xr['te(eV)'].values/1000,extrapolate=False)
        ti_keV_spline = CubicSpline(profile_data_xr['psi'].values, profile_data_xr['ti(eV)'].values/1000,extrapolate=False)
        ne_spline = CubicSpline(profile_data_xr['psi'].values, profile_data_xr['ne(m^-3)'].values,extrapolate=False)
        ni_spline = CubicSpline(profile_data_xr['psi'].values, profile_data_xr['ni(m^-3)'].values,extrapolate=False)
        return profile_data_xr, te_keV_spline, ti_keV_spline, ne_spline, ni_spline
    profile_data_xr, te_keV_spline, ti_keV_spline, ne_spline, ni_spline = get_default_profs(profile_filename)
else:
    eq_filename = '/nfs/home/stubenj9/equilibria/ARC/ARC_V3A_kinetic_tokamaker_v11_257_notruncate_scale_jBS=1.05.eqdsk'

print(" Getting equilibrium file from ", eq_filename)
eq_filename_short= eq_filename.split('/')[-1]


#########################################################################################################
# Run big function
#########################################################################################################
try:
    combined_xrb, pest3_xr_outb, input_dictb = analyse_with_mre(eq_filename,1,ni_spline, ne_spline, te_keV_spline, ti_keV_spline,
        run_stride=True,
        run_pest3=True,
        vac_flag='f',
        ode_flag='f',
        etol=1e-7,
        nx=64,
        mpsi=128,
        mtheta=129,
        energy_confinement_time=0.12,
        pest_match_truncation=False,
        wd_static=True,
        debug_mre_terms = False)
    combined_xr, pest3_xr_out, input_dict = analyse_with_mre(eq_filename,1,ni_spline, ne_spline, te_keV_spline, ti_keV_spline,
        run_stride=True,
        run_pest3=True,
        vac_flag='f',
        ode_flag='f',
        etol=1e-7,
        nx=64,
        mpsi=128,
        mtheta=129,
        energy_confinement_time=0.12,
        debug_mre_terms = False)
    print(" Ran analyse_with_mre without throwing an error ")
except Exception as e:
    print(" Error snagged during analyse_with_mre ")
    raise e
