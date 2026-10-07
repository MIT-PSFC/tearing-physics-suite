# Set of modular functions that take a filename as input, read the file, and generate kinetic spline inputs for drivers/pipeline.py

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import PchipInterpolator


def _average_ion_mass(psi_n, n_main, n_imp, main_mass_amu=2.0, imp_mass_amu=12.0):
    """rho-weighted (area-like) mean thermal ion mass [AMU] over psi_n <= 1."""
    psi_n = np.asarray(psi_n, dtype=float)
    mask = (psi_n >= 0) & (psi_n <= 1.0)
    rho = np.sqrt(psi_n[mask])
    n_main = np.asarray(n_main, dtype=float)[mask]
    n_imp = np.asarray(n_imp, dtype=float)[mask]
    mass = np.trapezoid(rho*(main_mass_amu*n_main + imp_mass_amu*n_imp), rho)
    dens = np.trapezoid(rho*(n_main + n_imp), rho)
    return float(mass/dens)

def read_kin_file(filename):
    """Read a .kin profile file and create splines for kinetic profiles.

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
            profile_data_names = pd.read_csv(filename, sep=r'\s+',header=None,nrows=1)
            profile_data_xr = xr.Dataset(pd.read_csv(filename, skiprows=1, sep=r'\s+', header=None,names=profile_data_names.iloc[0].values))
            te_keV_spline = PchipInterpolator(profile_data_xr['psi'].values, profile_data_xr['te(eV)'].values/1000,extrapolate=False)
            ti_keV_spline = PchipInterpolator(profile_data_xr['psi'].values, profile_data_xr['ti(eV)'].values/1000,extrapolate=False)
            ne_spline = PchipInterpolator(profile_data_xr['psi'].values, profile_data_xr['ne(m^-3)'].values,extrapolate=False)
            ni_spline = PchipInterpolator(profile_data_xr['psi'].values, profile_data_xr['ni(m^-3)'].values,extrapolate=False)
            omega_ExB_spline = PchipInterpolator(profile_data_xr['psi'].values, profile_data_xr['wexb(rad/s)'].values,extrapolate=False)
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


def read_IDA_lite(filename, verbose=False, time_idx=None, shot_id=None, extra_keys=None):
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
    extra_keys : list of str or None
        Extra IDA variables to return at time_idx. Finite radial profiles on psi_n
        are returned as splines, anything else as values.

    Returns
    -------
    dict
        Keys include ne_spline, te_keV_spline, ni_spline, ti_keV_spline,
        omega_tor_12C6_spline, v_pol_spline, Er_spline, time, time_idx,
        omega_splines, average_ion_mass (D + C), extra_keys, and optionally shot_id.
        If time_idx is None, returns a list of such dicts (one per time).
    """

    extra_keys = list(extra_keys) if extra_keys else []
    if time_idx is None:
        return read_IDA_lite_all_times(filename, verbose=verbose, shot_id=shot_id, extra_keys=extra_keys)

    if os.path.exists(filename):
        if verbose:
            print(f"\nReading rotation CDF file: {filename}")
        try:
            rotation_xr = xr.open_dataset(filename, engine='h5netcdf')
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
            # Create splines on psi_n for the selected time point
            #########################################################################################################
            if verbose: print("\n\nCreating splines on psi_n for selected time point...")
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

            # Create splines
            n_e_spline = PchipInterpolator(psi_n_vals, n_e_vals, extrapolate=False)
            T_e_spline = PchipInterpolator(psi_n_vals, T_e_vals, extrapolate=False)
            n_i_spline = PchipInterpolator(psi_n_vals, n_i_vals, extrapolate=False)
            T_i_spline = PchipInterpolator(psi_n_vals, T_i_vals, extrapolate=False)
            omega_tor_spline = PchipInterpolator(psi_n_vals, omega_tor_12C6_vals, extrapolate=False)
            v_pol_spline = PchipInterpolator(psi_n_vals, v_pol_vals, extrapolate=False)
            Er_spline = PchipInterpolator(psi_n_vals, E_r_vals, extrapolate=False)
            average_ion_mass = _average_ion_mass(psi_n_vals, n_i_vals, n_iC12_vals) # n_i assumed deuterium
            extra_key_vals = [rotation_xr[key].isel(time=time_idx).values for key in extra_keys]
            time_val = rotation_xr.time.values[time_idx]

            if verbose:
                print("✓ Successfully created splines:")
                print("  - n_e (electron density)")
                print("  - T_e (electron temperature)")
                print("  - omega_tor_12C6 (toroidal rotation)")
                print("  - v_pol (poloidal velocity)")
                print("  - E_r (radial electric field)")
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
        'time': time_val,
        'time_idx': time_idx,
        'omega_splines': {
            'omega_tor_12C6': omega_tor_spline
        },
        'average_ion_mass': average_ion_mass,
    }

    for key, val in zip(extra_keys, extra_key_vals):
        if np.ndim(val) > 0 and len(val) == len(psi_n_vals) and np.all(np.isfinite(val)):
            return_dict[key] = PchipInterpolator(psi_n_vals, val, extrapolate=False)
        else:
            return_dict[key] = val

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
    **kwargs
        Forwarded to read_IDA_lite (shot_id, extra_keys).

    Returns
    -------
    list of dict
        One dict per time slice, each with the same keys as read_IDA_lite.
    """

    if os.path.exists(filename):
        if verbose:
            print(f"\nReading rotation CDF file: {filename}")
        try:
            rotation_xr = xr.open_dataset(filename, engine='h5netcdf')

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

def read_bouquet_archive(path, scan_keys=None, selection="selected", eqdsk_out_dir=None,
                         ida_path=None, ida_extra_keys=None, ida_time_tol_ms=1.0,
                         meta_data=False, sample_limit=None, impurity_mass_amu=None,
                         eq_source='geqdsk', verbose=False):
    """Unpack a bouquet (schema v2) archive into multi_run_ inputs.

    Requires bouquet (branch bouquet_unified). Per draw: the equilibrium file is written
    to eqdsk_out_dir; ne/ni/Te/Ti are pchip-splined on psi_N_kinetic (as bouquet does); Zeff comes from
    aux_zeff (or Zeff); omega_splines['omega_tor'] / Er_spline from aux_omega_tor /
    aux_e_r when bouquet perturbed them.

    Parameters
    ----------
    path : str
        Bouquet .h5 archive.
    scan_keys : list or None
        Scan keys (time slices, ms) to read. None reads all.
    selection : {'selected', 'all', 'excluded'}
        Which draws to read (bouquet filter flags).
    eqdsk_out_dir : str or None
        Where to write g-files. Defaults to '<archive dir>/TPS_eqdsks'.
    ida_path : str or None
        IDA-lite file for scalar extras (tau_e_basic, tau_th_basic, ida_extra_keys).
        If None, the archive config's ida_path is used when present. Rotation is
        never taken from IDA.
    ida_extra_keys : list of str or None
        Additional IDA variables (stored with an 'IDA_' prefix).
    ida_time_tol_ms : float
        Max |IDA time - scan time| for the IDA join, in ms.
    meta_data : bool
        Also store bouquet per-draw metadata (bq_* keys) for multi_compile_zarr.
    sample_limit : int or None
        Max draws per scan key.
    impurity_mass_amu : float or None
        Impurity mass for average_ion_mass. Default 2*Z_imp.
    eq_source : {'geqdsk', 'ifile'}
        Equilibrium written per draw: the g-file, or the OFT i-file (bouquet
        write_ifile=True runs; GPEC reads it as eq_type 'ldp_i', see
        run_resistive_calculation).

    Returns
    -------
    eq_filenames : list of str
    profile_list : list of dict
    """
    try:
        import bouquet as bq
    except ImportError as e:
        raise ImportError("read_bouquet_archive requires bouquet (branch bouquet_unified) "
                          "installed in this environment.") from e

    ar = bq.BouquetArchive(path)
    header = os.path.splitext(os.path.basename(ar.path))[0]
    if eqdsk_out_dir is None:
        eqdsk_out_dir = os.path.join(os.path.dirname(os.path.abspath(ar.path)), 'TPS_eqdsks')
    os.makedirs(eqdsk_out_dir, exist_ok=True)
    if eq_source not in ('geqdsk', 'ifile'):
        raise ValueError(f"eq_source must be 'geqdsk'|'ifile', got {eq_source!r}")
    if selection not in ('selected', 'all', 'excluded'):
        raise ValueError(f"selection must be 'selected'|'all'|'excluded', got {selection!r}")

    eq_filenames, profile_list = [], []
    for key in (ar.scan_keys if scan_keys is None else scan_keys):
        sc = ar[key]
        cfg = _bouquet_scan_config(bq, ar.path, key)
        draws = getattr(sc, selection)
        if sample_limit is not None:
            draws = draws[:sample_limit]
        ida_extras = _bouquet_ida_extras(cfg, key, ida_path, ida_extra_keys, ida_time_tol_ms, verbose)

        for d in draws:
            prof = d.profiles
            attrs = d.attrs
            if attrs.get('profile_coord', 'psi_n') != 'psi_n':
                raise NotImplementedError(f"Bouquet draw {key}/{d.count} has profile_coord="
                                          f"{attrs['profile_coord']!r}; only psi_n archives are supported.")
            eq_bytes = d.eqdsk_bytes if eq_source == 'geqdsk' else d.ifile_bytes
            if eq_bytes is None:
                raise KeyError(f"Bouquet draw {key}/{d.count} has no stored {eq_source} "
                               f"(i-files need a bouquet run with write_ifile=True).")
            eq_path = os.path.join(eqdsk_out_dir, f"{header}_{key}_{d.count}.{eq_source}")
            with open(eq_path, 'wb') as fh:
                fh.write(eq_bytes)
            eq_filenames.append(eq_path)
            profile_list.append(_bouquet_draw_profiles(prof, attrs, cfg, key, d.count, ida_extras,
                                                       meta_data, impurity_mass_amu))
        if verbose:
            print(f"[read_bouquet_archive] scan {key}: {len(draws)} draws unpacked to {eqdsk_out_dir}")

    return eq_filenames, profile_list

def _bouquet_scan_config(bq, path, key):
    """BouquetConfig for one scan key, or None for archives without stored config."""
    try:
        return bq.utils.load_config(path, scan_key=key)
    except Exception:
        return None

def _scan_time(key):
    """Scan key as a float (bouquet time slices are keyed in ms), else the key itself."""
    try:
        return float(key)
    except (TypeError, ValueError):
        return key

def _bouquet_draw_profiles(prof, attrs, cfg, key, count, ida_extras, meta_data, impurity_mass_amu):
    """Build one multi_run_ profile dict from a bouquet draw."""
    psi_k = prof.get('psi_N_kinetic', prof['psi_N'])
    spl = lambda y: PchipInterpolator(psi_k, y, extrapolate=False)
    out = {
        'ne_spline': spl(prof['n_e']),
        'ni_spline': spl(prof['n_i']),
        'te_keV_spline': spl(prof['T_e']/1000),
        'ti_keV_spline': spl(prof['T_i']/1000),
        'time': _scan_time(key),
        'bq_scan_key': str(key),
        'bq_count': int(count),
    }

    zeff = prof.get('aux_zeff', prof.get('Zeff'))
    if zeff is not None:
        out['Zeff'] = {'x': np.asarray(psi_k).tolist(), 'y': np.asarray(zeff).tolist()}
    if 'aux_omega_tor' in prof:
        out['omega_splines'] = {'omega_tor': spl(prof['aux_omega_tor'])}
    if 'aux_e_r' in prof:
        out['Er_spline'] = spl(prof['aux_e_r'])

    # Single fully-stripped impurity: n_imp = (n_e - n_i)/Z_imp
    Z_imp = attrs.get('Z_imp', getattr(getattr(cfg, 'source', None), 'impurity_Z', 6.0))
    imp_mass = 2.0*Z_imp if impurity_mass_amu is None else impurity_mass_amu
    n_imp = np.clip((prof['n_e'] - prof['n_i'])/Z_imp, 0.0, None)
    out['average_ion_mass'] = _average_ion_mass(psi_k, prof['n_i'], n_imp, imp_mass_amu=imp_mass)

    out.update(ida_extras)

    if meta_data:
        psi_eq = prof['psi_N']
        for name in ('j_phi', 'j_BS', 'j_inductive'):
            if name in prof:
                out[f'bq_{name}'] = PchipInterpolator(psi_eq, prof[name], extrapolate=False)
        for name, val in attrs.items():
            if np.ndim(val) == 0 and not isinstance(val, (bytes, str)):
                out['bq_' + name.replace('(', '').replace(')', '')] = val
    return out

def _bouquet_ida_extras(cfg, key, ida_path, ida_extra_keys, tol_ms, verbose):
    """Scalar/profile extras from the IDA file at the scan's time (rotation excluded)."""
    if ida_path is None and cfg is not None:
        ida_path = (getattr(cfg.source, 'ida_path', None)
                    or getattr(getattr(cfg, 'uncertainty', None), 'ida_path', None))
    if ida_path is None:
        return {}
    if not os.path.exists(ida_path):
        warnings.warn(f"IDA file {ida_path} not found; skipping IDA extras for scan {key}.")
        return {}

    # Scan time [ms]: the source's IDA/scan time if set, else the scan key itself
    t_s = None
    if cfg is not None:
        t_s = getattr(cfg.source, 'ida_time', None) or getattr(cfg.source, 'time', None)
    t_ms = 1e3*float(t_s) if t_s is not None else float(key)

    with xr.open_dataset(ida_path, engine='h5netcdf') as ida:
        times = np.asarray(ida.time.values, dtype=float)
        available = set(ida.variables)
    time_idx = int(np.argmin(np.abs(times - t_ms)))
    if abs(times[time_idx] - t_ms) > tol_ms:
        warnings.warn(f"No IDA time within {tol_ms} ms of scan {key} ({t_ms} ms); skipping IDA extras.")
        return {}

    keys = [k for k in ('tau_e_basic', 'tau_th_basic') if k in available] + list(ida_extra_keys or [])
    ida = read_IDA_lite(ida_path, time_idx=time_idx, extra_keys=keys, verbose=verbose)
    extras = {'IDA_time': ida['time'], 'IDA_time_idx': time_idx}
    for k in keys:
        name = k if k in ('tau_e_basic', 'tau_th_basic') else 'IDA_' + k
        extras[name] = ida[k]
    return extras
