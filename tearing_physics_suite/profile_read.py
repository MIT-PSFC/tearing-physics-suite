# Set of modular functions that take a filename as input, read the file, and generate kinetic spline inputs for tearing_physics_suite.py

import os
import pandas as pd
import xarray as xr
from scipy.interpolate import Akima1DInterpolator
import h5py
import numpy as np

def read_kin_file(filename):
    """Read a .kin profile file and create cubic splines for kinetic profiles.

    Parameters
    ----------
    filename : str
        Path to the .kin file.

    Returns
    -------
        Returns
    -------
    dict
        Keys include ne_spline, te_keV_spline, ni_spline, ti_keV_spline
    """
    if os.path.exists(filename):
        print(f"\nReading kinetic profile file: {filename}")
        try:
            profile_data_names = pd.read_csv(filename, sep='\s+',header=None,nrows=1)
            profile_data_xr = xr.Dataset(pd.read_csv(filename, skiprows=1, sep='\s+', header=None,names=profile_data_names.iloc[0].values))
            te_keV_spline = Akima1DInterpolator(profile_data_xr['psi'].values, profile_data_xr['te(eV)'].values/1000,extrapolate=False)
            ti_keV_spline = Akima1DInterpolator(profile_data_xr['psi'].values, profile_data_xr['ti(eV)'].values/1000,extrapolate=False)
            ne_spline = Akima1DInterpolator(profile_data_xr['psi'].values, profile_data_xr['ne(m^-3)'].values,extrapolate=False)
            ni_spline = Akima1DInterpolator(profile_data_xr['psi'].values, profile_data_xr['ni(m^-3)'].values,extrapolate=False)
            omega_ExB_spline = Akima1DInterpolator(profile_data_xr['psi'].values, profile_data_xr['wexb(rad/s)'].values,extrapolate=False)
        except Exception as e:
            print(f"Error reading or processing .kin file: {e}")
            raise e
    else:
        print(f".kin file not found at {filename}")
        return None

    return {
        'ne_spline': ne_spline,
        'te_keV_spline': te_keV_spline,
        'ni_spline': ni_spline,
        'ti_keV_spline': ti_keV_spline,
        'omega_splines': {
            'omega_ExB': omega_ExB_spline
        }
    }
        

def read_IDA_lite(filename, verbose=False, time_idx=None, shot_id=None):
    """Read an IDA-lite .cdf file and return kinetic and rotation splines for MRE analysis.

    If time_idx is None, delegates to read_IDA_lite_all_times to process every time slice.

    Parameters
    ----------
    filename : str
        Path to the IDA-lite NetCDF file.
    verbose : bool
        Print diagnostic information during reading.
    time_idx : int or None
        Time index to extract. If None, returns splines for all times.
    shot_id : str or None
        Optional shot identifier stored in the returned dict.

    Returns
    -------
    dict
        Keys include ne_spline, te_keV_spline, ni_spline, ti_keV_spline,
        omega_tor_12C6_spline, v_pol_spline, Er_spline, time, time_idx,
        omega_splines, and optionally shot_id.
        If time_idx is None, returns a list of such dicts (one per time).
    """

    if time_idx is None:
        return read_IDA_lite_all_times(filename, verbose=verbose)

    if os.path.exists(filename):
        if verbose:
            print(f"\nReading rotation CDF file: {filename}")
        try:
            rotation_xr = xr.open_dataset(filename)
            if verbose:
                print("Successfully opened IDA-lite output")
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

            # Extract data for time point
            n_e_vals = rotation_xr.n_e.isel(time=time_idx).values
            T_e_vals = rotation_xr.T_e.isel(time=time_idx).values/1000
            n_iC12_vals = rotation_xr.n_12C6.isel(time=time_idx).values
            T_iC12_vals = rotation_xr.T_12C6.isel(time=time_idx).values/1000
            n_i_vals = n_e_vals-6*n_iC12_vals  # Assuming carbon is the only impurity, and quasi-neutrality holds
            T_i_vals = T_iC12_vals  # Assuming ion temperature is the same as carbon ion temperature
            omega_tor_12C6_vals = rotation_xr.omega_tor_12C6.isel(time=time_idx).values
            v_pol_vals = rotation_xr.v_pol.isel(time=time_idx).values
            E_r_vals = rotation_xr.E_r.isel(time=time_idx).values

            # Create cubic splines
            n_e_spline = Akima1DInterpolator(psi_n_vals, n_e_vals, extrapolate=False)
            T_e_spline = Akima1DInterpolator(psi_n_vals, T_e_vals, extrapolate=False)
            n_i_spline = Akima1DInterpolator(psi_n_vals, n_i_vals, extrapolate=False)
            T_i_spline = Akima1DInterpolator(psi_n_vals, T_i_vals, extrapolate=False)
            omega_tor_spline = Akima1DInterpolator(psi_n_vals, omega_tor_12C6_vals, extrapolate=False)
            v_pol_spline = Akima1DInterpolator(psi_n_vals, v_pol_vals, extrapolate=False)
            Er_spline = Akima1DInterpolator(psi_n_vals, E_r_vals, extrapolate=False)
            print("✓ Successfully created splines:")
            print(f"  - n_e (electron density)")
            print(f"  - T_e (electron temperature)")
            print(f"  - omega_tor_12C6 (toroidal rotation)")
            print(f"  - v_pol (poloidal velocity)")
            print(f"  - E_r (radial electric field)")
            # Test evaluation at a point
            test_psi_n = 0.5
            print(f"\nTest evaluation at psi_n = {test_psi_n}:")
            print(f"  ne (m^-3) = {n_e_spline(test_psi_n):.3e}")
            print(f"  te (keV) = {T_e_spline(test_psi_n):.3e}")
            print(f"  omega_tor_12C6 (rad/s) = {omega_tor_spline(test_psi_n):.3e}")
            print(f"  v_pol (m/s) = {v_pol_spline(test_psi_n):.3e}")
            print(f"  E_r (V/m) = {Er_spline(test_psi_n):.3e}")
        except Exception as e:
            print(f"Error reading or processing IDA-lite output: {e}")
            raise e
    else:
        print(f"IDA-lite output not found at {filename}")

    return_dict = {
        'ne_spline': n_e_spline,
        'te_keV_spline': T_e_spline,
        'ni_spline': n_i_spline,
        'ti_keV_spline': T_i_spline,
        'omega_tor_12C6_spline': omega_tor_spline,
        'v_pol_spline': v_pol_spline,
        'Er_spline': Er_spline,
        'time': rotation_xr.time.values[time_idx],
        'time_idx': time_idx,
        'omega_splines': {
            'omega_tor_12C6': omega_tor_spline
        }
    }

    if shot_id is not None:
        return_dict['shot_id'] = shot_id

    return return_dict

def read_IDA_lite_all_times(filename, verbose=False):
    """Read an IDA-lite .cdf file and return kinetic/rotation splines for every time slice.

    Parameters
    ----------
    filename : str
        Path to the IDA-lite NetCDF file.
    verbose : bool
        Print diagnostic information during reading.

    Returns
    -------
    list of dict
        One dict per time slice, each with the same keys as read_IDA_lite.
    """

    if os.path.exists(filename):
        if verbose:
            print(f"\nReading rotation CDF file: {filename}")
        try:
            rotation_xr = xr.open_dataset(filename)

            times = rotation_xr.time.values
            splines_by_time = []

            for time_idx, time_val in enumerate(times):
                if verbose:
                    print(f"\nProcessing time index {time_idx} (time={time_val})...")

                assert time_idx is not None # Avoid infinite recursion
                splines_by_time.append(read_IDA_lite(filename, verbose=verbose, time_idx=time_idx))

            return splines_by_time
        except Exception as e:
            print(f"Error reading or processing IDA-lite output: {e}")
            raise e
    else:
        print(f"IDA-lite output not found at {filename}")

    return None