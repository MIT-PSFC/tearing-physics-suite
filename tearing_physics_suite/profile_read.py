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

def read_bouquet_output(filename, eqdsk_dir=None, sample_limit=None, selection="all", scan_value=None, 
                            read_omega_ExB=False, Zeff_prof=False):
    """ Read the .h5 output of a bouquet run, unpack its geqdsk files, and 
    return a list of geqdsk filenames and kinetic profile splines.
    
    Parameters
    ----------
    filename : str
        bouquet filename
    sample_limit : bool
        take only on the first 'sample_limit' samples

    Returns
    -------
    eq_filenames : list of str
        Paths to equilibrium files, for input to multi_run_
    profile_list
        One dict per bouquet sample, for input to multi_run_
    """
    
    # Check if filename exists...
    # ---- resolve path ----------------------------------------------------
    if not filename.endswith(".h5"):
        h5path = os.path.abspath(f"{filename}.h5")
    else:
        h5path = os.path.abspath(filename)

    # ---- resolve which draws to show (filter selection) ------------------
    sel_indices = None
    if selection != "all":
        sel_indices = bouquet_select_indices(h5path, scan_value=scan_value,
                                      selection=selection)

    # ---- load data -------------------------------------------------------
    try:
        bl = bouquet_load_baseline_profiles(h5path, scan_value=scan_value)
    except KeyError:
        avail = bouquet_discover_scan_values(h5path)
        msg = (
            f"No data for scan_value={scan_value!r} in {h5path}.\n"
            f"Available scan values: {avail}"
        )
        raise KeyError(msg) from None
    psi_N = bl["psi_N"]
    perturbed_data_list = bouquet_load_all_perturbations(h5path, scan_value=scan_value,
                                        indices=sel_indices)
    
    # Get the kinetic profile information:
    k_keys = ["n_e [m^-3]", "n_i [m^-3]", "T_e [eV]", "T_i [eV]"]
    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list[0:min(n_equils,sample_limit)]): # Stub: check this line please 
            _psi_pert = data.get("psi_N_kinetic", psi_N)
            ne_spline = Akima1DInterpolator(_psi_pert,data["n_e [m^-3]"])
            ni_spline = Akima1DInterpolator(_psi_pert,data["n_i [m^-3]"])
            Te_spline = Akima1DInterpolator(_psi_pert,data["T_e [eV]"]/1000) # KeV
            Ti_spline = Akima1DInterpolator(_psi_pert,data["T_i [eV]"]/1000) # KeV
            profile_dict = {
                'ne_spline': ne_spline,
                'ni_spline': ni_spline,
                'te_keV_spline': Te_spline,
                'ti_keV_spline': Ti_spline,
            }
            if read_omega_ExB:
                profile_dict['omega_splines'] = {'omega_ExB': Akima1DInterpolator(_psi_pert,data["w_ExB [rad/s]"])}
            if Zeff_prof:
                Zeff_spline = Akima1DInterpolator(# STUB _psi_pert if len(Zeff) is len(_psi_pert) else psi_N 
                    ,data["Zeff"]
                )
            else:
                Zeff_spline = np.mean(data["Zeff"])
            ###
            

def bouquet_load_baseline_profiles(h5path, scan_value=None):
    """
    Load the baseline profiles and uncertainties for a given scan value.
    From bouquet - Daniel Burgess.

    Parameters
    ----------
    h5path : str
        Path to the ``.h5`` file.
    scan_value : str, float, or None
        ``None`` for flat-layout files.

    Returns
    -------
    result : dict
        All stored baseline arrays and scalar attributes.
    """
    bkey = _scan_val_key(scan_value)
    if bkey is not None:
        grp_path = f"scan/{bkey}/_baseline"
    else:
        grp_path = "_baseline"

    result = {}
    with h5py.File(h5path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Baseline group '{grp_path}' not found in {h5path}.  "
                f"Was store_baseline_profiles() called?"
            )
        grp = hf[grp_path]
        for key in grp.keys():
            result[key] = np.array(grp[key])
        for attr in grp.attrs:
            result[attr] = grp.attrs[attr]

    return result

def bouquet_load_all_perturbations(h5path, scan_value=None, indices=None):
    """Load all perturbed equilibria for a scan value as a list of dicts.  From bouquet - Daniel Burgess.

    Handles non-contiguous indices (from skipped equilibria) by
    discovering actual stored group names rather than assuming
    sequential 0..N-1.  When ``indices`` is given (an iterable of stored
    draw indices), only those draws are loaded -- used to honour a
    filter selection.
    """
    from .utils import _scan_val_key
    bkey = _scan_val_key(scan_value)
    with h5py.File(h5path, "r") as hf:
        if bkey is not None:
            parent = hf[f"scan/{bkey}"]
        else:
            parent = hf
        # Find all integer-keyed groups (skip _baseline, scan, etc.)
        stored_counts = sorted(
            int(k) for k in parent.keys()
            if k not in ("_baseline", "scan") and k.isdigit()
        )
    if indices is not None:
        keep = set(indices)
        stored_counts = [i for i in stored_counts if i in keep]
    return [
        load_equilibrium_by_path(h5path, count=i, scan_value=scan_value)
        for i in stored_counts
    ]

def bouquet_load_equilibrium_by_path(h5path, count, scan_value=None):
    """
    Load one perturbed equilibrium from an HDF5 file by path. From bouquet - Daniel Burgess.

    Like :func:`load_equilibrium` but takes a file path instead of a
    header string, and uses *scan_value* instead of *baseline*.  Does
    **not** extract the raw eqdsk bytes (use :func:`load_equilibrium`
    if you need those).
    """
    bkey = _scan_val_key(scan_value)
    if bkey is not None:
        grp_path = f"scan/{bkey}/{int(count)}"
    else:
        grp_path = str(int(count))

    result = {}
    with h5py.File(h5path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {h5path}"
            )
        grp = hf[grp_path]

        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure [Pa]" in grp:
            result["pressure [Pa]"] = np.array(grp["pressure [Pa]"])

        if "psi_N_kinetic" in grp:
            result["psi_N_kinetic"] = np.array(grp["psi_N_kinetic"])

        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])

        if "coil_currents [A]" in grp:
            import json
            values = np.array(grp["coil_currents [A]"])
            names = json.loads(grp.attrs.get("coil_names", "[]"))
            result["coil_currents"] = dict(zip(names, values))

        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

    return result

def bouquet_select_indices(h5path_or_header, scan_value=None, selection="selected"):
    """Stored draw indices filtered by ``selection``. From bouquet - Daniel Burgess.

    ``selection`` is one of:

      - ``'all'``      : every stored draw
      - ``'selected'`` : draws with ``selected`` True; if NO filter has
        been applied yet (no ``selected`` attr anywhere), returns all
        draws (so plotting selected on an unfiltered file shows all).
      - ``'excluded'`` : the complement of ``'selected'`` among stored
        draws (empty on an unfiltered file).

    Returns a flat sorted list when a single scan value is in play, or a
    ``{scan_val: [idx, ...]}`` dict when iterating multiple scan values.
    """
    if selection not in ("all", "selected", "excluded"):
        raise ValueError(
            f"selection must be 'all'|'selected'|'excluded', got {selection!r}")
    h5path = _resolve(h5path_or_header)
    svs = _iter_scan_vals(h5path, scan_value)
    result = {}
    with h5py.File(h5path, "r") as hf:
        for sv in svs:
            idxs = list_equilibrium_indices(h5path, scan_value=sv)
            if selection == "all":
                result[sv] = idxs
                continue
            have_any = any("selected" in hf[_group_path(sv, i)].attrs
                           for i in idxs if _group_path(sv, i) in hf)
            if not have_any:
                # unfiltered: 'selected' -> all, 'excluded' -> none
                result[sv] = idxs if selection == "selected" else []
                continue
            sel = [i for i in idxs
                   if bool(hf[_group_path(sv, i)].attrs.get("selected", True))]
            if selection == "selected":
                result[sv] = sel
            else:
                result[sv] = [i for i in idxs if i not in set(sel)]
    if len(svs) == 1:
        return result[svs[0]]
    return result

def bouquet_discover_scan_values(h5path):
    """
    Discover all scan values in an HDF5 equilibrium database. From bouquet - Daniel Burgess.

    Parameters
    ----------
    h5path : str
        Path to the ``.h5`` file.

    Returns
    -------
    scan_values : list[str] or None
        Sorted list of scan-value keys, or ``None`` if the file uses
        the flat layout (no ``scan/`` group).
    """
    with h5py.File(h5path, "r") as hf:
        if "scan" not in hf:
            return None
        keys = list(hf["scan"].keys())

    # Sort numerically when all keys look like numbers, otherwise
    # fall back to lexicographic order.
    try:
        return sorted(keys, key=float)
    except (ValueError, TypeError):
        return sorted(keys)