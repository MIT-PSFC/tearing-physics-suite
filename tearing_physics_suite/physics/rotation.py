import xarray as xr
import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator


def add_drift_rotation(rdcon_xarray,Er_spline=None,diamagnetic_rotation_ion_charge=None, dont_override_omega_ExB=True):
    """Compute ion and electron diamagnetic rotation frequencies at all psi_n values.

    Calculates omega_i and omega_e from density and temperature gradients.
    If Er_spline is provided, this function computes E x B rotation frequency, unless omega_ExB is already in rdcon_xarray and dont_override_omega_ExB is True.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Must contain ne_m3, ni_m3, te_keV, ti_keV, avg_nabla_psi, and psio.
    Er_spline : 1DSpline or None
        Radial electric field profile in V/m as a function of normalised poloidal flux.
    diamagnetic_rotation_ion_charge : float or None
        Ion charge state for diamagnetic frequency. If None, inferred from on-axis ne/ni.

    Returns
    -------
    xr.Dataset
        Input dataset with omega_i, omega_e (and omega_ExB if Er_spline provided) added.
    """

    # Necessary kinetic values of interest:
    ne_values = np.array(rdcon_xarray.ne_m3.values)
    ni_values = np.array(rdcon_xarray.ni_m3.values)
    te_values = np.array(rdcon_xarray.te_keV.values)
    ti_values = np.array(rdcon_xarray.ti_keV.values)
    avg_nablapsi_values = np.array(rdcon_xarray.avg_nabla_psi.values) # <|nabla psi|> 
    psio = np.array(rdcon_xarray.psio)
    # Single species approximation of ion charge, assuming on-axis density satisfies quasi-neutrality:
    if diamagnetic_rotation_ion_charge is not None:
        zi = diamagnetic_rotation_ion_charge
    else:
        zi = ne_values[0]/ni_values[0]
    rdcon_xarray = rdcon_xarray.assign(ne_on_ni_axis = zi)

    #Derivatives (from splines):
    ne_spline = PchipInterpolator(rdcon_xarray.psi_n, rdcon_xarray.ne_m3)
    ni_spline = PchipInterpolator(rdcon_xarray.psi_n, rdcon_xarray.ni_m3)
    te_spline = PchipInterpolator(rdcon_xarray.psi_n, rdcon_xarray.te_keV)
    ti_spline = PchipInterpolator(rdcon_xarray.psi_n, rdcon_xarray.ti_keV)
    ne1_values = np.array(ne_spline(rdcon_xarray.psi_n,1))
    ni1_values = np.array(ni_spline(rdcon_xarray.psi_n,1))
    te1_values = np.array(te_spline(rdcon_xarray.psi_n,1))
    ti1_values = np.array(ti_spline(rdcon_xarray.psi_n,1))

    # Check lengths:
    assert len(ne_values) == len(rdcon_xarray.psi_n) == len(avg_nablapsi_values)

    # Diamagnetic drift frequency in radians/s.
    omega_i_values = -ti_values*1e3*ni1_values/(zi*psio*ni_values)-ti1_values*1e3/(zi*psio) # Units rad/s: Ti, Te in this form have units eV*e/e = J/C = V, psio is in Weber/rad = (V*s)/rad (see Eq. 1 of https://doi.org/10.13182/FST48-968)
    omega_e_values =  te_values*1e3*ne1_values/(psio*ne_values)   +te1_values*1e3/(psio)    # Units rad/s

    # Save values onto xarray:
    rdcon_xarray = rdcon_xarray.assign(
        omega_i=omega_i_values+0.0*rdcon_xarray['psi_n'],       # Units rad/s
        omega_e=omega_e_values+0.0*rdcon_xarray['psi_n']        # Units rad/s
    )

    # Calculate ExB rotation if Er_spline is provided, and omega_ExB is not already in rdcon_xarray:
    if (Er_spline is not None) and not ('omega_ExB' in rdcon_xarray and dont_override_omega_ExB):
        Er_values = np.array(Er_spline(rdcon_xarray.psi_n.values))
        omega_ExB_values = Er_values/(psio*avg_nablapsi_values) # Units rad/s: Er units V/m, avg_nablapsi units 1/m, psio units Weber/rad = (V*s)/rad
        rdcon_xarray = rdcon_xarray.assign(
            Er=Er_values+0.0*rdcon_xarray['psi_n'],                 # Units V/m (assuming Er_spline is in V/m)
            omega_ExB=omega_ExB_values+0.0*rdcon_xarray['psi_n']    # Units rad/s
        )

    # Calculate total rotation frequencies if omega_ExB is present:
    if 'omega_ExB' in rdcon_xarray:
        rdcon_xarray = rdcon_xarray.assign(
            omega_ExB_plus_omega_e=rdcon_xarray['omega_ExB']+omega_e_values, # Units rad/s
            omega_ExB_plus_omega_i=rdcon_xarray['omega_ExB']+omega_i_values  # Units rad/s
        )

    rdcon_xarray = put_drift_rotation_on_surfaces(rdcon_xarray)

    return rdcon_xarray


def put_drift_rotation_on_surfaces(rdcon_xarray):
    """Interpolate drift rotation frequencies and their psi_n derivatives onto rational surfaces.

    Must be called after add_drift_rotation. Adds omega_i_surf, omega_e_surf, omega_i1_surf,
    omega_e1_surf (and ExB variants if omega_ExB is present) to rdcon_xarray.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Dataset with omega_i, omega_e on full psi_n grid (from add_drift_rotation).

    Returns
    -------
    xr.Dataset
        Input dataset with rotation _surf and _1_surf variables added.
    """

    # Adding values at surfaces
    rdcon_xarray = rdcon_xarray.assign(
        omega_i_surf = np.array(rdcon_xarray.omega_i.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values)+0.0*rdcon_xarray['psi_n_rational'],
        omega_e_surf = np.array(rdcon_xarray.omega_e.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values)+0.0*rdcon_xarray['psi_n_rational']
    )
    if 'omega_ExB' in rdcon_xarray:
        rdcon_xarray = rdcon_xarray.assign(
            omega_ExB_surf = np.array(rdcon_xarray.omega_ExB.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values)+0.0*rdcon_xarray['psi_n_rational'],
            omega_ExB_plus_omega_e_surf = np.array(rdcon_xarray.omega_ExB_plus_omega_e.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values)+0.0*rdcon_xarray['psi_n_rational'],
            omega_ExB_plus_omega_i_surf = np.array(rdcon_xarray.omega_ExB_plus_omega_i.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values)+0.0*rdcon_xarray['psi_n_rational']
        )
    if 'Er' in rdcon_xarray:
        rdcon_xarray = rdcon_xarray.assign(
            Er_surf = np.array(rdcon_xarray.Er.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values)+0.0*rdcon_xarray['psi_n_rational']
        )

    # Adding derivatives as surfaces:
    omega_i_spline = Akima1DInterpolator(rdcon_xarray.psi_n.values, rdcon_xarray.omega_i.values,extrapolate=False)
    omega_e_spline = Akima1DInterpolator(rdcon_xarray.psi_n.values, rdcon_xarray.omega_e.values,extrapolate=False)

    rdcon_xarray = rdcon_xarray.assign(
        omega_i1_surf = np.array(omega_i_spline(rdcon_xarray.psi_n_rational.values,1))+0.0*rdcon_xarray['psi_n_rational'],
        omega_e1_surf = np.array(omega_e_spline(rdcon_xarray.psi_n_rational.values,1))+0.0*rdcon_xarray['psi_n_rational']
    )

    if 'omega_ExB' in rdcon_xarray:
        omega_ExB_spline = Akima1DInterpolator(rdcon_xarray.psi_n.values, rdcon_xarray.omega_ExB.values,extrapolate=False)
        omega_ExB_plus_omega_e_spline = Akima1DInterpolator(rdcon_xarray.psi_n.values, rdcon_xarray.omega_ExB_plus_omega_e.values,extrapolate=False)
        omega_ExB_plus_omega_i_spline = Akima1DInterpolator(rdcon_xarray.psi_n.values, rdcon_xarray.omega_ExB_plus_omega_i.values,extrapolate=False)
        rdcon_xarray = rdcon_xarray.assign(
            omega_ExB1_surf = np.array(omega_ExB_spline(rdcon_xarray.psi_n_rational.values,1))+0.0*rdcon_xarray['psi_n_rational'],
            omega_ExB_plus_omega_e1_surf = np.array(omega_ExB_plus_omega_e_spline(rdcon_xarray.psi_n_rational.values,1))+0.0*rdcon_xarray['psi_n_rational'],
            omega_ExB_plus_omega_i1_surf = np.array(omega_ExB_plus_omega_i_spline(rdcon_xarray.psi_n_rational.values,1))+0.0*rdcon_xarray['psi_n_rational']
        )

    return rdcon_xarray


def add_rotation(rdcon_xarray,omega_splines=None):
    """Save measured rotation frequencies onto rdcon_xarray at full psi_n grid and rational surfaces.

    Also computes and stores the psi_n derivative of each rotation frequency at rational surfaces.
    Use spline key "omega_ExB" for ExB rotation frequency in rad/s.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Dataset to add rotation data to.
    omega_splines : dict of 1DSplines
        Mapping of rotation name (e.g. 'omega_tor') to a rotation spline (on psi_n) in rad/s.

    Returns
    -------
    xr.Dataset
        Input dataset with {key}, {key}_surf, and {key}1_surf variables added for each spline.
    """

    for key in omega_splines:
        spline = omega_splines[key]
        rdcon_xarray = rdcon_xarray.assign(
            **{key: np.array(spline(rdcon_xarray.psi_n.values))+0.0*rdcon_xarray['psi_n']}
        )
        # Put on surfaces:
        rdcon_xarray = rdcon_xarray.assign(
            **{f"{key}_surf": np.array(spline(rdcon_xarray.psi_n_rational.values))+0.0*rdcon_xarray['psi_n_rational']}
        )
        # Put derivatives on surfaces:
        rdcon_xarray = rdcon_xarray.assign(
            **{f"{key}1_surf": np.array(spline(rdcon_xarray.psi_n_rational.values,1))+0.0*rdcon_xarray['psi_n_rational']}
        )
    
    return rdcon_xarray


def decorrelation_timescales(rdcon_xarray,q_surfs_of_interest=[1],psi_surfs_of_interest=[0.95],omega_splines=None,verbose=True, debug=False):
    """Calculate decorrelation timescales between all m,n surfaces and selected reference surfaces.

    For each rotation quantity, computes 2*pi / delta_omega to get the decorrelation
    timescale (seconds) relative to reference surfaces defined by q value or psi_n.

    Warning: Assumes omega_splines are in units of radians/s.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Dataset with rotation quantities already added (via add_rotation / add_drift_rotation).
    q_surfs_of_interest : list of float
        q values whose largest-psi_n rational surface is used as a reference (default [1]).
    psi_surfs_of_interest : list of float
        psi_n values used as reference surfaces (default [0.95]).
    omega_splines : dict or None
        Same omega_splines passed to add_rotation; keys determine which rotations are included.
    verbose : bool
        Print information about found surfaces.
    debug : bool
        If True, returns the first decorrelation DataArray early for debugging.

    Returns
    -------
    xr.Dataset
        Input dataset with {key}_tdecorr_qsurf and {key}_tdecorr_psisurf variables added,
        plus rotation_keys coordinate.
    """
    #########################################################################################################
    # Defining a short-list of all terms to calculate decorrelation timescales for.
    #########################################################################################################

    # Start with the generalised rotation frequencies from omega_splines:
    rotation_keys = omega_splines.keys() if omega_splines is not None else []
    # Add '_surf' suffix to rotation keys where its missing. These terms should already be added to rdcon_xarray by add_rotation.
    rotation_keys = [key if '_surf' in key else f"{key}_surf" for key in rotation_keys]

    # Add the drift rotation frequencies to the list of rotation keys:
    rotation_keys = rotation_keys + ['omega_i_surf']
    rotation_keys = rotation_keys + ['omega_e_surf']

    # Add ExB rotation frequencies to the list of rotation keys if Er_spline is provided, or omega_
    if 'omega_ExB' in rdcon_xarray:
        rotation_keys = rotation_keys + ['omega_ExB_surf']
        rotation_keys = rotation_keys + ['omega_ExB_plus_omega_e_surf']
        rotation_keys = rotation_keys + ['omega_ExB_plus_omega_i_surf']

    # Sanity check that all these keys are in rdcon_xarray:
    for key in rotation_keys:
        # Confirm that key ends in '_surf':
        if not key.endswith('_surf'):
            no_suffix_key = None
            raise ValueError(f"Key {key} does not end in '_surf'. Debug function decorrelation_timescales")
        else:
            no_suffix_key = key[:-5] # Remove '_surf' suffix to get the key without it

        if key not in rdcon_xarray:
            print(f"Available keys in rdcon_xarray: {rdcon_xarray.data_vars.keys()}")
            print(f"Missing key: {key}")
            raise ValueError(f"Key {key} not found in rdcon_xarray. Please check that add_rotation and add_drift_rotation have been run, and that omega_splines keys match the keys in rdcon_xarray.")

        if no_suffix_key not in rdcon_xarray:
            print(f"Available keys in rdcon_xarray: {rdcon_xarray.data_vars.keys()}")
            print(f"Missing key: {no_suffix_key}")
            raise ValueError(f"Key {no_suffix_key} not found in rdcon_xarray. Please check that add_rotation and add_drift_rotation have been run, and that omega_splines keys match the keys in rdcon_xarray.")

    #########################################################################################################
    # Convert q_surfs_of_interest to psi_n values
    #########################################################################################################

    psi_n_at_q_surfs_of_interest = [] # List to store the psi_n values at the rational surfaces of interest
    for q_surf in q_surfs_of_interest:
        #########################################################################################################
        # Find largest psi_n value where the q profile crosses the q value of interest.
        #########################################################################################################
        shifted_q_spline = Akima1DInterpolator(rdcon_xarray.psi_n.values, np.array(rdcon_xarray.q.values-q_surf),extrapolate=False)
        # Count how many times q_spline crosses this q_surf, and get the corresponding psi_n values:
        q_roots = shifted_q_spline.roots()
        if len(q_roots) == 0:
            if verbose: print(f"No rational surface found for q={q_surf}.")
            psi_n_at_q_root = np.nan
        elif len(q_roots) > 1:
            if verbose: print(f"Multiple rational surfaces found for q={q_surf}. Using the largest root at psi_n={q_roots.max()}.")
            psi_n_at_q_root = q_roots.max()
        else:
            if verbose: print(f"One rational surface found for q={q_surf} at psi_n={q_roots[0]}.")
            psi_n_at_q_root = q_roots[0]

        psi_n_at_q_surfs_of_interest.append(psi_n_at_q_root)

    #########################################################################################################
    # Cycle through keys and calculate decorrelation timescales
    #########################################################################################################
    for key in rotation_keys:
        key_no_suffix = key[:-5] # Remove '_surf' suffix to get the key without it
        decorellation_data_arrays_qsurf = []    # List to store the decorrelation data arrays for each q surface of interest for this key
        decorellation_data_arrays_psisurf = []  # List to store the decorrelation data arrays for each psi surface of interest for this key

        #########################################################################################################
        # q_surfs_of_interest loop:
        #########################################################################################################
        for psi_n_at_q_root in psi_n_at_q_surfs_of_interest:
            # If nans, just make an array of nans for this surface of interest and move on to the next one:
            # Important because we want the dimension of q_surfs_of_interest to be consistent across all timeslices in a shot...
            if np.isnan(psi_n_at_q_root):
                decorellation_data_array = xr.full_like(rdcon_xarray[key], np.nan)
                decorellation_data_arrays_qsurf.append(decorellation_data_array)
                continue

            rotation_freq_at_surf_of_interest = rdcon_xarray[key_no_suffix].interp(psi_n=psi_n_at_q_root,method="cubic").values

            # Double check rotation_freq_at_surf_of_interest is a scalar:
            if np.ndim(rotation_freq_at_surf_of_interest) != 0:
                print(f"Rotation frequency at surface of interest for key {key} is not a scalar. Value: {rotation_freq_at_surf_of_interest}")
                raise ValueError(f"Rotation frequency at surface of interest for key {key} is not a scalar. Please debug decorrelation_timescales.")

            # Differences for rotation key:
            decorellation_data_array = rdcon_xarray[key]-rotation_freq_at_surf_of_interest # Frequency difference (in rad/s)
            decorellation_data_array = np.pi*2/decorellation_data_array # Convert to decorrelation timescale in seconds (assuming an entire 2pi rotation = decorrelation)
            decorellation_data_arrays_qsurf.append(decorellation_data_array)

        # Stack the decorrelation data arrays for each surface of interest into a single xarray DataArray with a new dimension 'q_surfs_of_interest':
        if len(psi_n_at_q_surfs_of_interest) > 0:
            decorellation_data_arrays_qsurf = xr.concat(decorellation_data_arrays_qsurf, dim='q_surfs_of_interest')
            decorellation_data_arrays_qsurf = decorellation_data_arrays_qsurf.assign_coords(q_surfs_of_interest=q_surfs_of_interest) # Assign the q values of interest as coordinates for this new dimension

        # Check you get what you're expecting:
        if debug:
            return decorellation_data_arrays_qsurf

        #########################################################################################################
        # psi_surfs_of_interest loop:
        #########################################################################################################
        for psi_n_of_interest in psi_surfs_of_interest:
            # Check if psi_n_of_interest is in the range of psi_n values in rdcon_xarray:
            if psi_n_of_interest < rdcon_xarray.psi_n.min() or psi_n_of_interest > rdcon_xarray.psi_n.max():
                print(f"psi_n_of_interest {psi_n_of_interest} is out of bounds. Must be between {rdcon_xarray.psi_n.min()} and {rdcon_xarray.psi_n.max()}.")
                raise ValueError(f"psi_n_of_interest {psi_n_of_interest} is out of bounds. Please check your input to decorrelation_timescales.")

            rotation_freq_at_surf_of_interest = rdcon_xarray[key_no_suffix].interp(psi_n=psi_n_of_interest,method="cubic").values

            # Double check rotation_freq_at_surf_of_interest is a scalar:
            if np.ndim(rotation_freq_at_surf_of_interest) != 0:
                print(f"Rotation frequency at surface of interest for key {key} is not a scalar. Value: {rotation_freq_at_surf_of_interest}")
                raise ValueError(f"Rotation frequency at surface of interest for key {key} is not a scalar. Please debug decorrelation_timescales.")

            decorellation_data_array = rdcon_xarray[key]-rotation_freq_at_surf_of_interest # Frequency difference (in rad/s)
            decorellation_data_array = np.pi*2/decorellation_data_array # Convert to decorrelation timescale in seconds (assuming an entire 2pi rotation = decorrelation)
            decorellation_data_arrays_psisurf.append(decorellation_data_array)

        # Stack the decorrelation data arrays for each surface of interest into a single xarray DataArray with a new dimension 'psi_surf_of_interest':
        if len(psi_surfs_of_interest) > 0:
            decorellation_data_arrays_psisurf = xr.concat(decorellation_data_arrays_psisurf, dim='psi_surf_of_interest')
            decorellation_data_arrays_psisurf = decorellation_data_arrays_psisurf.assign_coords(psi_surf_of_interest=psi_surfs_of_interest) # Assign the psi_n values of interest as coordinates for this new dimension

        # Now we have two xarray DataArrays for this key: one with decorrelation timescales to rational surfaces of interest, and one with decorrelation timescales to psi surfaces of interest. We can save these onto rdcon_xarray with new keys:
        if len(psi_n_at_q_surfs_of_interest) > 0:
            rdcon_xarray = rdcon_xarray.assign(**{f"{key_no_suffix}_tdecorr_qsurf": decorellation_data_arrays_qsurf})  
        if len(psi_surfs_of_interest) > 0:
            rdcon_xarray = rdcon_xarray.assign(**{f"{key_no_suffix}_tdecorr_psisurf": decorellation_data_arrays_psisurf})

    # Save rotation keys onto xarray for later reference:
    rdcon_xarray = rdcon_xarray.assign(rotation_keys=rotation_keys)

    return rdcon_xarray


def decorrelation_ratios(rdcon_xarray):
    """Calculate ratios of decorrelation timescales to physics-relevant timescales.

    For each rotation key, computes the decorrelation time normalised by taua_surf, taur_surf,
    and Q0_surf. Must be called after decorrelation_timescales.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Dataset with _tdecorr_qsurf and _tdecorr_psisurf variables.

    Returns
    -------
    xr.Dataset
        Input dataset with _on_taua, _on_taur, and _Q0 ratio variables added.
    """
    
    for key in rdcon_xarray.rotation_keys.values:
        key_no_suffix = key[:-5] # Remove '_surf' suffix to get the key without it

        if f"{key_no_suffix}_tdecorr_qsurf" in rdcon_xarray:
            taua_surf___ = rdcon_xarray['taua_surf'].broadcast_like(rdcon_xarray[f"{key_no_suffix}_tdecorr_qsurf"])
            taur_surf___ = rdcon_xarray['taur_surf'].broadcast_like(rdcon_xarray[f"{key_no_suffix}_tdecorr_qsurf"])
            Q0_surf___ = rdcon_xarray['Q0_surf'].broadcast_like(rdcon_xarray[f"{key_no_suffix}_tdecorr_qsurf"])

            rdcon_xarray = rdcon_xarray.assign(**{f"{key_no_suffix}_tdecorr_qsurf_on_taua": rdcon_xarray[f"{key_no_suffix}_tdecorr_qsurf"]/taua_surf___})
            rdcon_xarray = rdcon_xarray.assign(**{f"{key_no_suffix}_tdecorr_qsurf_on_taur": rdcon_xarray[f"{key_no_suffix}_tdecorr_qsurf"]/taur_surf___})
            rdcon_xarray = rdcon_xarray.assign(**{f"{key_no_suffix}_tdecorr_qsurf_Q0": rdcon_xarray[f"{key_no_suffix}_tdecorr_qsurf"]*Q0_surf___})

        if f"{key_no_suffix}_tdecorr_psisurf" in rdcon_xarray:
            taua_surf____ = rdcon_xarray['taua_surf'].broadcast_like(rdcon_xarray[f"{key_no_suffix}_tdecorr_psisurf"])
            taur_surf____ = rdcon_xarray['taur_surf'].broadcast_like(rdcon_xarray[f"{key_no_suffix}_tdecorr_psisurf"])
            Q0_surf____ = rdcon_xarray['Q0_surf'].broadcast_like(rdcon_xarray[f"{key_no_suffix}_tdecorr_psisurf"])

    
            rdcon_xarray = rdcon_xarray.assign(**{f"{key_no_suffix}_tdecorr_psisurf_on_taua": rdcon_xarray[f"{key_no_suffix}_tdecorr_psisurf"]/taua_surf____})
            rdcon_xarray = rdcon_xarray.assign(**{f"{key_no_suffix}_tdecorr_psisurf_on_taur": rdcon_xarray[f"{key_no_suffix}_tdecorr_psisurf"]/taur_surf____})
            rdcon_xarray = rdcon_xarray.assign(**{f"{key_no_suffix}_tdecorr_psisurf_Q0": rdcon_xarray[f"{key_no_suffix}_tdecorr_psisurf"]*Q0_surf____})

    return rdcon_xarray
