# Tests for rotation analysis: nonlinear resistive calculation with rotation splines and radial electric field

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
from tearing_physics_suite.profile_read import read_IDA_lite

os.chdir(home_dir)

from tearing_physics_suite.fortran_wrappers import run_resistive_calculation
from tearing_physics_suite.fortran_wrappers import compile_xarrays
from tearing_physics_suite.utils import trim_nans
from tearing_physics_suite.delta_prime_extraction import extract_delta_primes_, extract_delta_primes, delta_prime_no_couple, delta_prime_full_couple, delta_prime_nn_couple, delta_prime_2nn_couple, get_delta_prime_divisors,extract_variances
from tearing_physics_suite.tearing_physics_suite import linear_resistive_calculation, nonlinear_resistive_calculation
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




print(" Getting equilibrium file from ", eq_filename)
eq_filename_short= eq_filename.split('/')[-1]


#########################################################################################################
# Read rotation_.cdf
#########################################################################################################

rotation_cdf_path = os.path.join(home_dir, 'tests', 'rotation_.cdf')

if os.path.exists(rotation_cdf_path):
    print(f"\nReading rotation CDF file: {rotation_cdf_path}")
    try:
        rotation_xr = xr.open_dataset(rotation_cdf_path)
        print("Successfully opened rotation_.cdf")
        print("\nDataset info:")
        print(rotation_xr)
        print("\nData variables:")
        for var in rotation_xr.data_vars:
            print(f"  {var}: {rotation_xr[var].dims} {rotation_xr[var].shape}")
        print("\nCoordinates:")
        for coord in rotation_xr.coords:
            print(f"  {coord}: {rotation_xr[coord].shape}")
        print("\nAttributes:")
        for attr in rotation_xr.attrs:
            print(f"  {attr}: {rotation_xr.attrs[attr]}")
        #########################################################################################################
        # Create cubic splines on psi_n for first time point
        #########################################################################################################
        print("\n\nCreating cubic splines on psi_n for first time point...")
        psi_n_vals = rotation_xr.psi_n.values
        time_idx = 100  # Use first time point
        # Extract data for time point
        n_e_vals = rotation_xr.n_e.isel(time=time_idx).values
        T_e_vals = rotation_xr.T_e.isel(time=time_idx).values
        n_iC12_vals = rotation_xr.n_12C6.isel(time=time_idx).values
        T_iC12_vals = rotation_xr.T_12C6.isel(time=time_idx).values
        n_i_vals = n_e_vals-6*n_iC12_vals  # Assuming carbon is the only impurity, and quasi-neutrality holds
        T_i_vals = T_iC12_vals  # Assuming ion temperature is the same as carbon ion temperature
        omega_tor_vals = rotation_xr.omega_tor_12C6.isel(time=time_idx).values
        v_pol_vals = rotation_xr.v_pol.isel(time=time_idx).values
        E_r_vals = rotation_xr.E_r.isel(time=time_idx).values
        # Create cubic splines
        n_e_spline = CubicSpline(psi_n_vals, n_e_vals, extrapolate=False)
        T_e_spline = CubicSpline(psi_n_vals, T_e_vals, extrapolate=False)
        n_i_spline = CubicSpline(psi_n_vals, n_i_vals, extrapolate=False)
        T_i_spline = CubicSpline(psi_n_vals, T_i_vals, extrapolate=False)
        omega_tor_spline = CubicSpline(psi_n_vals, omega_tor_vals, extrapolate=False)
        v_pol_spline = CubicSpline(psi_n_vals, v_pol_vals, extrapolate=False)
        Er_spline = CubicSpline(psi_n_vals, E_r_vals, extrapolate=False)
        print("✓ Successfully created splines:")
        print(f"  - n_e (electron density)")
        print(f"  - T_e (electron temperature)")
        print(f"  - omega_tor (toroidal rotation)")
        print(f"  - v_pol (poloidal velocity)")
        print(f"  - E_r (radial electric field)")
        # Test evaluation at a point
        test_psi_n = 0.5
        print(f"\nTest evaluation at psi_n = {test_psi_n}:")
        print(f"  n_e = {n_e_spline(test_psi_n):.3e}")
        print(f"  T_e = {T_e_spline(test_psi_n):.3e}")
        print(f"  omega_tor = {omega_tor_spline(test_psi_n):.3e}")
        print(f"  v_pol = {v_pol_spline(test_psi_n):.3e}")
        print(f"  E_r = {Er_spline(test_psi_n):.3e}")
    except Exception as e:
        print(f"Error reading or processing rotation_.cdf: {e}")
        raise e
else:
    print(f"rotation_.cdf not found at {rotation_cdf_path}")


# Create omega_splines dictionary
omega_splines = {
    'omega_tor': omega_tor_spline
}
    
#########################################################################################################
# Run big function
#########################################################################################################
combined_xrb, pest3_xr_outb, input_dictb = nonlinear_resistive_calculation(eq_filename,1,ni_spline, ne_spline, te_keV_spline, ti_keV_spline,
    Zeff=1.5,
    average_ion_mass=2.5,
    q_surfs_of_interest=[1.5,2.0],
    psi_surfs_of_interest=[0.3,0.95],
    Er_spline=Er_spline,
    omega_splines=omega_splines,
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