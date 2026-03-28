# Python functions to construct cross-field transport terms for the modified Rutherford equation

import xarray as xr
import numpy as np
from scipy.interpolate import CubicSpline

# Checked
def chi_para_lmfp_no_w_on_modes(rdcon_xarray):
    """Long-mean-free-path parallel thermal diffusivity at rational surfaces, without island width factor. This calculation assumes the 
    island connection length is setting the parallel transport timescale, not the electron-ion collision time.

    Implements Fitzpatrick 2023 Eq. 14.206, converted to normalised poloidal flux space
    (Rosenburg PoP 2002 Eq. 33). Divide by island width to get chi_parallel_lmfp.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Must contain v_te_surf, psi_n_rational, flux_shear_s_surf, ro, n.

    Returns
    -------
    xr.Dataset
        Input dataset with chi_para_lmfp_no_w_surf [m^2/s] added.
    """
    R0 = rdcon_xarray.ro
    n = rdcon_xarray.n

    # Fitzpatrick 2023 14.206 equation (converted into normalised poloidal flux space as per Rosenburg PoP 2002 eq. 33)
    rdcon_xarray = rdcon_xarray.assign(
        chi_para_lmfp_no_w_surf= 2*R0*rdcon_xarray['v_te_surf']*rdcon_xarray['psi_n_rational']/(np.sqrt(np.pi)*n*rdcon_xarray['flux_shear_s_surf']) #Divide by island width in normalised flux space to get chi_parallel_lmfp
    )
    
    return rdcon_xarray

def chi_para_lmfp_noisland_on_modes(rdcon_xarray):
    """Long-mean-free-path parallel thermal diffusivity with no island, using (half) the helical field line connection length instead of 
    electron-ion collision time (Fitzpatrick 1995 Eq. 132, divided by electron density). Appropriate when no island is present.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Must contain avg_R_surf, avg_r_surf, q_rational, v_te_surf, n.

    Returns
    -------
    xr.Dataset
        Input dataset with helical_correction_length_surf [m] and
        chi_para_lmfp_noisland_surf [m^2/s] added.
    """
    n = rdcon_xarray.n
    m_ints = np.round(n*rdcon_xarray['q_rational'].values).astype(int)

    # We remove any common factors of m and n to get the reduced m and n for each surface:
    import math
    m_reds = np.array([int(np.round(m_ints[i]/math.gcd(m_ints[i],n))) for i in range(len(m_ints))])
    n_reds = np.array([int(n/math.gcd(m_ints[i],n)) for i in range(len(m_ints))])

    # Toroidal distance around the tokamak:
    major_radii_at_modes = rdcon_xarray['avg_R_surf'].values
    Lc_tors = m_reds*(2*np.pi*major_radii_at_modes) #[m]
    # Poloidal distance around the tokamak:
    minor_radii_at_modes = rdcon_xarray['avg_r_surf'].values
    Lc_pols = n_reds*(2*np.pi*minor_radii_at_modes) #[m]

    # Connection length is approximated by the square root of the sum of the squares of the toroidal and poloidal distances:
    Lc_surf = np.array(np.sqrt(Lc_tors*Lc_tors + Lc_pols*Lc_pols)) #[m]

    # Add connection length to rdcon_xarray:
    rdcon_xarray = rdcon_xarray.assign(
        helical_correction_length_surf = Lc_surf+0.0*rdcon_xarray['psi_n_rational']
    )

    # Using Fitzpatrick 1995 equation 132, and dividing by electron density.
    # We use half the connection length, as this is the largest physical distance between two points on a closed field 
    # line over which convective heat transport can occur (making no distinction between co and counter-passing electrons).
    rdcon_xarray = rdcon_xarray.assign(
        chi_para_lmfp_noisland_surf = 0.5*rdcon_xarray['helical_correction_length_surf']*rdcon_xarray['v_te_surf'] #[m^2/s]
    )
    
    return rdcon_xarray

# Checked
def chi_para_smfp_on_modes(rdcon_xarray,Zeff):
    """Short-mean-free-path parallel thermal diffusivity at rational surfaces.

    Implements Fitzpatrick 2023 Eq. 14.205: chi_para set by assuming the mean free path is set by the electron-ion collision time.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Must contain taue_surf and v_te_surf.
    Zeff : float
        Effective ion charge.

    Returns
    -------
    xr.Dataset
        Input dataset with chi_para_smfp_surf [m^2/s] added.
    """

    # Fitzpatrick 2023 14.205
    rdcon_xarray = rdcon_xarray.assign(
        chi_para_smfp_surf = 1.581*rdcon_xarray['taue_surf']*(rdcon_xarray['v_te_surf']**2)/(1+0.2535*Zeff)
    )

    return rdcon_xarray

# Checked
def chi_perp_on_modes(rdcon_xarray, 
        areal_elongation=None, 
        minor_radius=None, 
        energy_confinement_time=None, 
        chi_perp_spline=None):
    """
    Calculate the perpendicular thermal diffusivity in units m^2/s.
    Will use chi_perp_spline if provided. Otherwise, energy_confinement_time (alone) is required.
    Default operation with energy_confinement_time assumes all energy goes through each surface (see **). 
    However if minor_radius and areal_elongation are provided, we apply the Fitzpatrick 1995 formula (see ***).

    Parameters
    ----------
    rdcon_xarray : xarray.DataArray
        The xarray containing the radial coordinate data.
    areal_elongations : float, optional
        The areal_elongation of the plasma, used if chi_perp_spline is not provided.
    minor_radii : float, optional
        The minor radius of the plasma, used if chi_perp_spline is not provided.
    energy_confinement_time : float, optional
        The energy confinement time of the plasma, used if chi_perp_spline is not provided.
    chi_perp_spline : CubicSpline, optional
        A precomputed spline for the perpendicular thermal diffusivity, used if provided.

    Returns
    ----------
    xr.Dataset
        Input dataset with chi_perp_surf [m^2/s] added.
    """

    if not (chi_perp_spline is None): # Use chi_perp_spline
        chi_perp_on_modes = np.array(chi_perp_spline(rdcon_xarray.psi_n_rational.values))

    elif energy_confinement_time is None: # If no spline, need energy_confinement_time to continue
        raise ValueError("Must provide either chi_perp spline, or areal_elongation, minor_radius, and energy_confinement_time to calculate chi_perp.")

    elif minor_radius is None: # **Assume all energy goes through each surface. This is the default option!
        minor_radii_squared = rdcon_xarray.avg_r_surf.values*rdcon_xarray.avg_r_surf.values
        chi_perp_on_modes = np.array(minor_radii_squared / (6*energy_confinement_time)) # Assumes all energy put into the plasma deposits in core (overestimates chi_perp inner surfaces)
    
    elif isinstance(minor_radius, float) and isinstance(areal_elongation, float): # ***Uses Fitzpatrick 1995 formula:
        chi_perp = areal_elongation * minor_radius**2 / (6*energy_confinement_time) # One value for all surfaces
        chi_perp_on_modes = np.array([chi_perp]*len(rdcon_xarray.psi_n_rational.values)) 

    else:
        raise ValueError("Incorrect entries to chi_perp function.") 

    rdcon_xarray = rdcon_xarray.assign(chi_perp_surf=chi_perp_on_modes+0.0*rdcon_xarray['psi_n_rational'])

    if chi_perp_spline is None:
        # Add energy_confinement_time to rdcon_xarray:
        rdcon_xarray = rdcon_xarray.assign(energy_confinement_time=energy_confinement_time)

    return rdcon_xarray