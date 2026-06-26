# Set of modular functions that take a filename as input, read the file, and generate kinetic spline inputs for tearing_physics_suite.py

import os
import pandas as pd
import xarray as xr
from scipy.interpolate import Akima1DInterpolator
import h5py
import numpy as np
import pickle
from tearing_physics_suite.sampling_DB import GPRProfilePerturber

class _IndexMap:
    """Picklable map_object: ``map_object(idx)`` returns ``flat_list[idx]``."""
    def __init__(self, flat_list):
        self.flat_list = flat_list

    def __call__(self, idx):
        return self.flat_list[idx]

    def __getitem__(self, idx):
        return self.flat_list[idx]

    def __len__(self):
        return len(self.flat_list)

    def __iter__(self):
        return iter(self.flat_list)

class _IndexMapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_IndexMap":          # intercept regardless of original module
            return _IndexMap
        return super().find_class(module, name)   # everything else: normal lookup

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
        
def read_IDA_lite(filename, verbose=False, time_idx=None, shot_id=None, extra_keys=[]):
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

    extra_key_vals = []

    if time_idx is None:
        return read_IDA_lite_all_times(filename, verbose=verbose)

    if os.path.exists(filename):
        if verbose:
            print(f"\nReading rotation CDF file: {filename}")
        try:
            rotation_xr = xr.open_dataset(filename,engine='h5netcdf')
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
            if verbose: print("\n\nCreating cubic splines on psi_n for first time point...")
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

            # Rotation uncertainties
            E_r_err = rotation_xr.E_r_err.isel(time=time_idx).values
            omega_tor_12C6_err = rotation_xr.omega_tor_12C6_err.isel(time=time_idx).values

            # Create cubic splines
            n_e_spline = Akima1DInterpolator(psi_n_vals, n_e_vals, extrapolate=False)
            T_e_spline = Akima1DInterpolator(psi_n_vals, T_e_vals, extrapolate=False)
            n_i_spline = Akima1DInterpolator(psi_n_vals, n_i_vals, extrapolate=False)
            T_i_spline = Akima1DInterpolator(psi_n_vals, T_i_vals, extrapolate=False)
            omega_tor_spline = Akima1DInterpolator(psi_n_vals, omega_tor_12C6_vals, extrapolate=False)
            v_pol_spline = Akima1DInterpolator(psi_n_vals, v_pol_vals, extrapolate=False)
            Er_spline = Akima1DInterpolator(psi_n_vals, E_r_vals, extrapolate=False)

            # extra_keys
            for extra_key in extra_keys:
                extra_key_vals.append(rotation_xr[extra_key].isel(time=time_idx).values)
            # a simple area-averaged ion mass
            rho_n = np.sqrt(psi_n_vals)
            total_ion_mass_radial_integral_spline = Akima1DInterpolator(rho_n,rho_n*(2*n_i_vals+12*n_iC12_vals)) # assuming n_i is deuterium
            total_ion_density_radial_integral_spline = Akima1DInterpolator(rho_n,rho_n*(n_i_vals+n_iC12_vals)) # assuming n_i is deuterium
            average_ion_mass = total_ion_mass_radial_integral_spline.integrate(0,1.0)/total_ion_density_radial_integral_spline.integrate(0,1.0)

            if verbose:
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
        },
        'E_r_err': E_r_err,
        'omega_tor_12C6_err': omega_tor_12C6_err,
        'average_ion_mass': average_ion_mass
    }

    for i,extra_key in enumerate(extra_keys):
        return_dict[extra_key] = extra_key_vals[i]

    if shot_id is not None:
        return_dict['shot_id'] = shot_id

    return return_dict

def read_IDA_lite_all_times(filename, verbose=False, **kwargs):
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
                splines_by_time.append(read_IDA_lite(filename, verbose=verbose, time_idx=time_idx, **kwargs))

            return splines_by_time
        except Exception as e:
            print(f"Error reading or processing IDA-lite output: {e}")
            raise e
    else:
        print(f"IDA-lite output not found at {filename}")

    return None

def read_bouquet_working_dir(directory, rotation_method, verbose=True, quick_exit=None, **kwargs):
    """ Enter a parallel bouquet run directory for a single shot, collate all individual samples
    and generate inputs for tearing physics suite's multi_run function.

    Parameters
    ----------
    directory : str
        bouquet parallel run directory
    rotation_method : function
        takes map_object.pkl from the bouquet directory, and outputs rotation splines

    Returns
    -------
    eq_filenames : list of str
        Paths to equilibrium files, for input to multi_run_
    profile_list
        One dict per bouquet sample, for input to multi_run_
    """

    # check if directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    # check that 'map_object.pkl' lives in the directory, if not raise error
    map_object_path = os.path.join(directory, 'map_object.pkl')
    if not os.path.exists(map_object_path):
        raise ValueError(f"'map_object.pkl' not found in {directory}")
    
    # look for subdirectories of type 'worker_<n>' where n is an integer
    # inside those subdirectories make a subdirectory 'TPS_eqdsks'
    worker_dirs = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith("worker_"):
            try:
                # Try to extract the integer part after 'worker_'
                int(item.split("worker_")[1])
                worker_dirs.append(item_path)
                
                # Create TPS_eqdsks subdirectory
                tps_eqdsks_path = os.path.join(item_path, "TPS_eqdsks")
                os.makedirs(tps_eqdsks_path, exist_ok=True)
            except (ValueError, IndexError):
                # Skip if the suffix is not an integer
                pass
    
    if not worker_dirs:
        raise ValueError(f"No worker_<n> subdirectories found in {directory}")
    
    
    header_list = []
    
    # Inside each worker_dirs, search for all files that have the form str(header+f"_idx{j}.h5") where j is an integer.
    import re
    
    for worker_dir in worker_dirs:
        for filename in os.listdir(worker_dir):
            # Match pattern: something_idxN.h5
            match = re.match(r'(.+)_idx(\d+)\.h5$', filename)
            if match:
                header = match.group(1)
                idx = int(match.group(2))
                header_list.append((header, idx, worker_dir, filename))

    # Check every header in header_list is the same, if not raise NotImplementedError("Can't have multiple bouquet runs in the same directory")
    if len(header_list)>0:
        unique_headers = set(h[0] for h in header_list)
        if len(unique_headers) > 1:
            raise NotImplementedError("Can't have multiple bouquet runs in the same directory")
    else:
        raise ValueError("No H5 files found matching pattern '*_idx*.h5'")
    
    # Verify there are no repeated idx values
    indices = [h[1] for h in header_list]
    if len(indices) != len(set(indices)):
        from collections import Counter
        duplicates = [idx for idx, count in Counter(indices).items() if count > 1]
        raise ValueError(f"Duplicate indices found in bouquet output: {duplicates}")
    
    full_eq_filename_list = []
    full_spline_list = []
    print("Unpacking bouquet equilibria")
    for h in header_list:
        try:
            eq_filenames, splines = read_single_bouquet_output(os.path.join(h[2],h[3]),eqdsk_out_dir=os.path.join(h[2],"TPS_eqdsks"))
        except Exception as e:
            if verbose: print("     idx ",h[1]," - skipped due to load fail ",e)
        if len(eq_filenames)==len(splines)==0:
            if verbose: print("     idx ",h[1]," - skipped due to empty file")
            continue
        else:
            if verbose: print(" idx ",h[1],f": {len(splines)} eqdsks unpacked")
        splines = rotation_method(h, map_object_path, splines, **kwargs)
        full_eq_filename_list.extend(eq_filenames)
        full_spline_list.extend(splines)
        if quick_exit is not None:
            if quick_exit == 0: 
                break
            quick_exit -= 1

    
    return full_eq_filename_list, full_spline_list

def bouquet_ida_rotation(h, map_object_path, splines, resample=False, Er_ls=None, omega_tor_ls=None, extra_keys=['tau_e_basic','tau_th_basic']):
    """Generic rotation_method that adds rotations splines to inout variable splines"""

    (_, idx, worker_dir, _) = h

    with open(map_object_path, 'rb') as f:
        map_object = _IndexMapUnpickler(f).load()

    ida_path, _, time_idx = map_object[idx]

    ida_file = ida_path.split("/")[-1]
    if os.path.isdir(os.path.join(worker_dir,ida_file)):
        ida_path = os.path.join(worker_dir,ida_file) # local version preferrable

    try:
        return_dict = read_IDA_lite(ida_path, time_idx=idx, extra_keys=extra_keys)
        assert time_idx==idx
    except Exception as e:
        print(e)
        raise e

    # Generating samples for Error propagation via resampling
    if resample:
        splnx=return_dict['Er_spline'].x
        splny=return_dict['Er_spline'](return_dict['Er_spline'].x)
        _Er_gpr = GPRProfilePerturber(kernel_func="rbf", length_scale=Er_ls)
        _Er_gpr.precompute_factor(splnx, return_dict['E_r_err']/splny[0]) # This cannot be done in parallel (see parallel.py in bouquet)
        Er_samples = _Er_gpr.draw_from_factor(splny/splny[0], len(splines), np.random.default_rng()) * splny[0]

        splnx=return_dict['omega_tor_12C6_spline'].x
        splny=return_dict['omega_tor_12C6_spline'](return_dict['omega_tor_12C6_spline'].x)
        _omega_tor_gpr = GPRProfilePerturber(kernel_func="rbf", length_scale=omega_tor_ls)
        _omega_tor_gpr.precompute_factor(splnx, return_dict['omega_tor_12C6_err']/splny[0]) # This cannot be done in parallel (see parallel.py in bouquet)
        omega_tor_samples = _omega_tor_gpr.draw_from_factor(splny/splny[0], len(splines), np.random.default_rng()) * splny[0]

        assert np.shape(Er_samples)==(len(splines), len(return_dict['E_r_err']))
        assert np.shape(Er_samples)==(len(splines), len(return_dict['omega_tor_12C6_err']))

    output_splines = []
    for i,spline in enumerate(splines):
        if resample and i>0:
            spline['Er_spline'] = Akima1DInterpolator(return_dict['Er_spline'].x,Er_samples[i])
            spline['omega_splines'] = {
                'omega_tor_12C6': Akima1DInterpolator(return_dict['omega_tor_12C6_spline'].x,omega_tor_samples[i])
            }
        else:
            spline['Er_spline'] = return_dict['Er_spline']
            spline['omega_splines'] = return_dict['omega_splines']
        
        spline['average_ion_mass'] = return_dict['average_ion_mass'] # A scalar for now...
        for extra_key in extra_keys: # no error propagation (yet)
            spline[extra_key] = return_dict[extra_key]

        output_splines.append(spline)

    return splines

def read_single_bouquet_output(filename, eqdsk_out_dir=None, sample_limit=None, selection="all", scan_value=None, 
                            read_omega_ExB=False, debug=False):
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
    sel_indices = bouquet_select_indices(h5path, debug=debug, selection='all')

    # ---- load data -------------------------------------------------------
    try:
        bl = bouquet_load_baseline_profiles(h5path, scan_value=scan_value)
    except KeyError:
        avail = bouquet_discover_scan_values(h5path)
        msg = (
            f"No data for scan_value={scan_value!r} in {h5path}.\n"
            f"Available scan values: {avail}"
        )
        return [], []
    psi_N = bl["psi_N"]
    perturbed_data_list = bouquet_load_all_perturbations(h5path, scan_value=scan_value,
                                        indices=sel_indices)
    
    profile_dict_list = []
    geqdsk_filename_list = []
    assert len(perturbed_data_list) == len(sel_indices)
    if perturbed_data_list:
        if not sample_limit is None:
            raise NotImplementedError("sample_limit not implemented yet")
        for i, data in enumerate(perturbed_data_list): # Stub: check this line please 
    # Get the kinetic profile information:
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
            # Check for Zeff or Zeff_profile in data, then proceed:
            if "Zeff" in data:
                zeff_key = "Zeff"
            elif "Zeff_profile" in data:
                zeff_key = "Zeff_profile"
            else:
                raise ValueError("Neither 'Zeff' nor 'Zeff_profile' found in data")
            
            if len(data[zeff_key]) == len(psi_N):
                Zeff_x_vals = psi_N
            elif len(data[zeff_key]) == len(_psi_pert):
                Zeff_x_vals = _psi_pert
            else:
                raise ValueError(
                    f"{zeff_key} length {len(data[zeff_key])} matches neither "
                    f"psi_N ({len(psi_N)}) nor _psi_pert ({len(_psi_pert)})"
                )
            assert len(Zeff_x_vals)==len(data[zeff_key])
            profile_dict['Zeff']={'x':Zeff_x_vals,'y':data[zeff_key]}
            if read_omega_ExB:
                profile_dict['omega_splines'] = {'omega_ExB': Akima1DInterpolator(_psi_pert,data["w_ExB [rad/s]"])}
                raise NotImplementedError("read_omega_ExB unfinished")
            profile_dict_list.append(profile_dict)
    # Unpack geqdsk from samples:
            if eqdsk_out_dir is None:
                eqdsk_out_dir = os.getcwd()
            result = bouquet_load_equilibrium(h5path.split('.h5')[0], sel_indices[i], eqdsk_out_dir=eqdsk_out_dir) #double check
            geqdsk_filename_list.append(result['eqdsk_filepath'])
            #print(result['eqdsk_filepath'])
    return geqdsk_filename_list, profile_dict_list

def _scan_val_key(scan_val):
    """Convert a scan-value label (float, int, or str) to an HDF5-safe string.

    Returns ``None`` when *scan_val* is ``None`` (flat layout).
    """
    if scan_val is None:
        return None
    return str(scan_val)

def _group_path(scan_val, count):
    """Return the internal HDF5 group path for a given entry."""
    bkey = _scan_val_key(scan_val)
    if bkey is not None:
        return f"scan/{bkey}/{int(count)}"
    return str(int(count))

def _eqdsk_dataset_name(header, scan_val, count):
    """Return the dataset name used for the raw eqdsk bytes."""
    base = os.path.basename(header)
    bkey = _scan_val_key(scan_val)
    if bkey is not None:
        safe_key = bkey.replace("/", "_").replace(" ", "_")
        return f"{base}_{safe_key}_{int(count)}.eqdsk"
    return f"{base}_{int(count)}.eqdsk"

def bouquet_load_equilibrium(header, count, scan_val=None, eqdsk_out_dir=None):
    """
    Retrieve one equilibrium entry from the HDF5 database.
    From bouquet - Daniel Burgess.

    Parameters
    ----------
    header : str
        Base name of the database.
    count : int
        Perturbation index.
    scan_val : str, float, int, or None
        Scan-point label (must match what was used at write time).
    eqdsk_out_dir : str or None, optional
        If given, the raw eqdsk is written to a file in this directory.

    Returns
    -------
    result : dict
        Keys: ``"eqdsk_filepath"``, ``"eqdsk_bytes"``,
        the 1-D array names, ``"l_i(1)"``, ``"l_i(3)"``,
        and optionally ``"pressure [Pa]"``, ``"Zeff"``,
        ``"coil_currents"``, ``"pfile_bytes"``.
    """
    db_path  = os.path.abspath(f"{header}.h5")
    grp_path = _group_path(scan_val, count)
    ds_name  = _eqdsk_dataset_name(header, scan_val, count)

    result = {}

    with h5py.File(db_path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {db_path}"
            )
        grp = hf[grp_path]

        # ---- eqdsk raw bytes -------------------------------------------
        eqdsk_bytes = bytes(grp[ds_name][()])
        result["eqdsk_bytes"] = eqdsk_bytes

        if eqdsk_out_dir is not None:
            os.makedirs(eqdsk_out_dir, exist_ok=True)
            out_path = os.path.join(eqdsk_out_dir, ds_name)
            with open(out_path, "wb") as fh:
                fh.write(eqdsk_bytes)
            result["eqdsk_filepath"] = os.path.abspath(out_path)
        else:
            result["eqdsk_filepath"] = None

        # ---- 1-D arrays ------------------------------------------------
        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure [Pa]" in grp:
            result["pressure [Pa]"] = np.array(grp["pressure [Pa]"])

        # ---- scalars ----------------------------------------------------
        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

        # ---- optional: Zeff -----------------------------------------------
        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])

        # ---- optional: p-file bytes ----------------------------------------
        pf_ds = ds_name.replace(".eqdsk", ".pfile")
        if pf_ds in grp:
            result["pfile_bytes"] = bytes(grp[pf_ds][()])

        # ---- optional: coil currents ---------------------------------------
        if "coil_currents [A]" in grp:
            import json
            values = np.array(grp["coil_currents [A]"])
            names = json.loads(grp.attrs.get("coil_names", "[]"))
            result["coil_currents"] = dict(zip(names, values))

    return result

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
    def _scan_val_key(scan_val):
        """Convert a scan-value label (float, int, or str) to an HDF5-safe string.

        Returns ``None`` when *scan_val* is ``None`` (flat layout).
        """
        if scan_val is None:
            return None
        return str(scan_val)
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
    def _scan_val_key(scan_val):
        """Convert a scan-value label (float, int, or str) to an HDF5-safe string.

        Returns ``None`` when *scan_val* is ``None`` (flat layout).
        """
        if scan_val is None:
            return None
        return str(scan_val)
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
    output = [
        bouquet_load_equilibrium_by_path(h5path, count=i, scan_value=scan_value)
        for i in stored_counts
    ]
    
    # If output[0] has 'Zeff' but none of the other output[i]'s have Zeff, 
    # propagate output[0]['Zeff'] to all output[i]'s
    if output and 'Zeff' in output[0]:
        other_have_zeff = any('Zeff' in output[i] for i in range(1, len(output)))
        if not other_have_zeff:
            for i in range(1, len(output)):
                output[i]['Zeff'] = output[0]['Zeff']
    
    return output

_PROFILE_KEYS = [
        "psi_N",
        "j_phi [A m^-2]",
        "j_BS [A m^-2]",
        "j_BS,edge [A m^-2]",
        "j_inductive [A m^-2]",
        "n_e [m^-3]",
        "T_e [eV]",
        "n_i [m^-3]",
        "T_i [eV]",
        "w_ExB [rad/s]",
    ]

def bouquet_load_equilibrium_by_path(h5path, count, scan_value=None):
    """
    Load one perturbed equilibrium from an HDF5 file by path. From bouquet - Daniel Burgess.

    Like :func:`load_equilibrium` but takes a file path instead of a
    header string, and uses *scan_value* instead of *baseline*.  Does
    **not** extract the raw eqdsk bytes (use :func:`load_equilibrium`
    if you need those).
    """

    def _scan_val_key(scan_val):
        """Convert a scan-value label (float, int, or str) to an HDF5-safe string.

        Returns ``None`` when *scan_val* is ``None`` (flat layout).
        """
        if scan_val is None:
            return None
        return str(scan_val)

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

def bouquet_select_indices(h5path_or_header, scan_value=None, selection="selected", debug=False):
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
    
    Parameters
    ----------
    debug : bool
        If True, print the full HDF5 tree (groups, datasets, and attributes).
    """

    def _print_h5_tree(h5path):
        def _visit(name, obj):
            indent = "  " * name.count("/")
            kind = "GROUP" if isinstance(obj, h5py.Group) else "DATASET"
            shape = obj.shape if hasattr(obj, "shape") else ""
            attrs = dict(obj.attrs)
            print(f"{indent}/{name}  [{kind}] {shape}  attrs={attrs}")
        print(f"\n=== HDF5 tree: {h5path} ===")
        with h5py.File(h5path, "r") as hf:
            print(f"  (root)  attrs={dict(hf.attrs)}")
            print(f"  top-level keys: {list(hf.keys())}")
            hf.visititems(_visit)
        print("=== end tree ===\n")

    def _scan_val_key(scan_val):
        """Convert a scan-value label (float, int, or str) to an HDF5-safe string.

        Returns ``None`` when *scan_val* is ``None`` (flat layout).
        """
        if scan_val is None:
            return None
        return str(scan_val)
    
    def _group_path(scan_val, count):
        """Return the internal HDF5 group path for a given entry."""
        bkey = _scan_val_key(scan_val)
        if bkey is not None:
            return f"scan/{bkey}/{int(count)}"
        return str(int(count))

    def _resolve(h5path_or_header):
        if not h5path_or_header.endswith(".h5"):
            return os.path.abspath(f"{h5path_or_header}.h5")
        return os.path.abspath(h5path_or_header)

    def _iter_scan_vals(h5path, scan_value):
        if scan_value is not None:
            return [scan_value]
        svs = bouquet_discover_scan_values(h5path)
        return svs if svs else [None]
    
    def list_equilibrium_indices(h5path, scan_value=None):
        """Return the sorted list of integer draw indices actually stored.

        Band-rejected / failed draws leave GAPS in the index sequence (e.g.
        ``[0, 1, 2, 3, 4, 5, 7, ...]`` with draw 6 missing), so callers must
        iterate these indices rather than ``range(count_equilibria(...))`` --
        the latter assumes a contiguous ``0..n-1`` and KeyErrors on the gap.

        Parameters
        ----------
        h5path : str
            Path to the ``.h5`` file.
        scan_value : str, float, or None
            Scan-value key.  ``None`` for the flat layout.

        Returns
        -------
        list of int
            Sorted stored draw indices.
        """
        bkey = _scan_val_key(scan_value)
        with h5py.File(h5path, "r") as hf:
            parent = hf[f"scan/{bkey}"] if bkey is not None else hf
            found = sorted(
                int(k) for k in parent.keys()
                if k not in ("_baseline", "scan") and str(k).lstrip("-").isdigit()
            )
            if debug:
                print(f"  [list_equilibrium_indices] scan_value={scan_value!r}, bkey={bkey!r}")
                print(f"  parent keys: {list(parent.keys())}")
                print(f"  integer-keyed indices found: {found}")
            return found

    if selection not in ("all", "selected", "excluded"):
        raise ValueError(
            f"selection must be 'all'|'selected'|'excluded', got {selection!r}")
    h5path = _resolve(h5path_or_header)
    if debug:
        _print_h5_tree(h5path)
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