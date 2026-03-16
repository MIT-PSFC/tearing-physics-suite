# Scripts to test the functions in tearing_physics_suite.py

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
from tearing_physics_suite.tearing_physics_suite import linear_resistive_calculation, nonlinear_resistive_calculation,global_mre_quantities
from tearing_physics_suite.mre_analysis import analyse_with_mre

os.chdir(home_dir)

#########################################################################################################
# user settings:
#########################################################################################################

fast=False # <- Don't change this
test_linear_calculation=False
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
# Linear multiple-n calculations:
#########################################################################################################

if test_linear_calculation:
    combined_xrl, input_dict_outl, pest3_xr_vecl, xarray_vecl = linear_resistive_calculation(eq_filename,nvec=[1,2],
        run_stride=True,
        run_pest3=True,
        vac_flag='f',
        ode_flag='f',
        etol=1e-7,
        nx=64,
        mpsi=128,
        mtheta=129
        )

#########################################################################################################
# Nonlinear multiple-n calculations:
#########################################################################################################

combined_xr, input_dict_out, pest3_xr_vec, xarray_vec = nonlinear_resistive_calculation(eq_filename,
    ni_spline,
    ne_spline,
    te_keV_spline,
    ti_keV_spline,
    Zeff=1.5,
    average_ion_mass=2.5,
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
    mtheta=129
    )


##############################################################################################################
# Analysing input_dict_out results:
##############################################################################################################
print(input_dict_out)
##############################################################################################################
# Analysing delta_prime_variability results:
##############################################################################################################
print(combined_xr.Delta_prime_diff_across_GPEC)
print(combined_xr.Delta_prime_reldiff_across_GPEC)
print(combined_xr.Delta_prime_reldiff_across_GPEC_2)
print(combined_xr.Delta_prime_diff_across_GPECvsPEST)
print(combined_xr.Delta_prime_reldiff_across_GPECvsPEST)
print(combined_xr.Delta_prime_reldiff_across_GPECvsPEST_2)
##############################################################################################################
# Analysing global_mre_quantities results:
##############################################################################################################
combined_xr1= global_mre_quantities(combined_xr,psi_pedestal_cutoff=0.8)
combined_xr2= global_mre_quantities(combined_xr,psi_pedestal_cutoff=1.0)
print(combined_xr1.w_marg_surf.isel(Delta_prime_type=3,code=0).values)
print(combined_xr1.min_w_marg_rank.isel(Delta_prime_type=3,code=0).values)
print(combined_xr1.min_w_marg_allsurf.isel(Delta_prime_type=3,code=0).values)
print(combined_xr1.dwdtau_max_surf.isel(Delta_prime_type=3,code=0).values)
print(combined_xr1.psi_n_rational.isel(code=0).values)
print(combined_xr1.max_dwdtau_rank.isel(Delta_prime_type=3,code=0).values)
print(combined_xr1.max_dwdtau_allsurf.isel(Delta_prime_type=3,code=0).values)

print("Got through all calculations without throwing an error")