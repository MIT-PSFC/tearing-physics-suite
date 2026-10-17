# Python functions to construct cross-field transport terms for the modified Rutherford equation

import numpy as np

import tearing_physics_suite.physics.constants as gv
from tearing_physics_suite.physics.xr_utils import like

# chi_perp / chi_par models for the transport-threshold width (chi_model):
#   'tau_E'    chi_perp from energy_confinement_time (total thermal energy); chi_par averaged over electrons and ions
#   'electron' chi_perp from chi_perp_spline (electron profile); electron chi_par
#   'total'    chi_perp and chi_par both averaged over electrons and ions (chi_perp_spline and chi_perp_i_spline);
#              'electron' on surfaces where the averaged chi_perp is <= 0
# Averages use the ion share f_i of chi_ion_weight_on_modes.
CHI_MODELS = ('tau_E', 'electron', 'total')


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
        helical_correction_length_surf = like(Lc_surf, rdcon_xarray['psi_n_rational'])
    )

    # Using Fitzpatrick 1995 equation 132, and dividing by electron density.
    # We use half the connection length, as this is the largest physical distance between two points on a closed field
    # line over which convective heat transport can occur (making no distinction between co and counter-passing electrons).
    rdcon_xarray = rdcon_xarray.assign(
        chi_para_lmfp_noisland_surf = 0.5*rdcon_xarray['helical_correction_length_surf']*rdcon_xarray['v_te_surf'] #[m^2/s]
    )

    return rdcon_xarray

# Checked
def chi_para_smfp_on_modes(rdcon_xarray):
    """Short-mean-free-path parallel thermal diffusivity at rational surfaces.

    Implements Fitzpatrick 2023 Eq. 14.205: chi_para set by assuming the mean free path is set by the electron-ion collision time.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Must contain taue_surf and v_te_surf.

    Returns
    -------
    xr.Dataset
        Input dataset with chi_para_smfp_surf [m^2/s] added.
    """

    # Fitzpatrick 2023 14.205
    rdcon_xarray = rdcon_xarray.assign(
        chi_para_smfp_surf = 1.581*rdcon_xarray['taue_surf']*(rdcon_xarray['v_te_surf']**2)/(1+0.2535*rdcon_xarray['Zeff_surf'])
    )

    return rdcon_xarray

def chi_para_ion_on_modes(rdcon_xarray, main_ion_mass_amu=2.0):
    """Main-ion parallel thermal diffusivity at rational surfaces [m^2/s], short and long mean free path.

    SMFP: Braginskii chi_par,i = 3.9 T_i tau_i/m_i, with tau_i from Wesson (Tokamaks p. 729) for main ions colliding
    with all ions (field density sum_j n_j Z_j^2 = Zeff n_e in place of n_i) and Wesson's ion Coulomb logarithm.
    LMFP: chi_para_lmfp_no_w_surf with the main-ion thermal speed in place of v_te.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Must contain ti_keV_surf, ni_m3_surf, ne_m3_surf, Zeff_surf, v_te_surf, chi_para_lmfp_no_w_surf.
    main_ion_mass_amu : float
        Main-ion mass (deuterium by default; not the impurity-weighted average_ion_mass).

    Returns
    -------
    xr.Dataset
        Input dataset with taui_surf [s], v_ti_main_surf [m/s], chi_para_smfp_i_surf and
        chi_para_lmfp_no_w_i_surf [m^2/s] added.
    """
    ti = rdcon_xarray['ti_keV_surf']
    m_i = main_ion_mass_amu*gv.amu
    lnLamb_i = 17.3 - 0.5*np.log(rdcon_xarray['ni_m3_surf']/1e20) + 1.5*np.log(ti)
    taui = 6.60e17*np.sqrt(m_i/gv.mp)*ti**1.5/(rdcon_xarray['Zeff_surf']*rdcon_xarray['ne_m3_surf']*lnLamb_i)
    v_ti = np.sqrt(2*gv.e*1e3*ti/m_i)
    return rdcon_xarray.assign(
        taui_surf=taui,
        v_ti_main_surf=v_ti,
        chi_para_smfp_i_surf=3.9*gv.e*1e3*ti*taui/m_i,
        chi_para_lmfp_no_w_i_surf=rdcon_xarray['chi_para_lmfp_no_w_surf']*v_ti/rdcon_xarray['v_te_surf'])


def chi_ion_weight_on_modes(rdcon_xarray, chi_model='tau_E'):
    """Ion share f_i of the single-fluid diffusivities: chi = (1 - f_i) chi_e + f_i chi_i, for chi_perp and chi_par alike.

    With a common island temperature shape for both species (delta T_s = T_s' f), the electron and ion heat equations
    sum to one equation for f, in which each chi_s is weighted by n_s T_s'. This is the single-fluid equation of
    Schlutt & Hegna 2012 (Eqs. 35-38), so one W_c follows. 'tau_E' and 'total' use f_i = n_i T_i'/(n_e T_e' + n_i T_i');
    'electron' uses f_i = 0.

    Returns
    -------
    xr.Dataset
        Input dataset with chi_ion_weight_surf added.
    """
    if chi_model not in CHI_MODELS:
        raise ValueError(f"chi_model must be one of {CHI_MODELS}, got {chi_model!r}")
    if chi_model in ('tau_E', 'total'):
        g_e = rdcon_xarray['ne_m3_surf']*rdcon_xarray['te1_keV_surf']
        g_i = rdcon_xarray['ni_m3_surf']*rdcon_xarray['ti1_keV_surf']
        f_i = g_i/(g_e + g_i)
    else:
        f_i = 0.0*rdcon_xarray['psi_n_rational']
    return rdcon_xarray.assign(chi_ion_weight_surf=f_i)


# Checked
def chi_perp_on_modes(rdcon_xarray,
        areal_elongation=None,
        minor_radius=None,
        energy_confinement_time=None,
        chi_perp_spline=None,
        chi_perp_i_spline=None,
        chi_model='tau_E'):
    """
    Calculate the perpendicular thermal diffusivity in units m^2/s; chi_model (see CHI_MODELS) picks the source:
    'tau_E' energy_confinement_time, 'electron' chi_perp_spline, 'total' (1 - f_i) chi_perp_spline + f_i chi_perp_i_spline
    (needs chi_ion_weight_surf). Where that is <= 0, 'total' falls back to 'electron' (chi_perp_spline, f_i = 0).
    'tau_E' assumes all energy goes through each surface (see **).
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
    chi_perp_spline : 1DSpline, optional
        A precomputed spline for the perpendicular thermal diffusivity, used if provided.
    chi_perp_i_spline : 1DSpline, optional
        Ion perpendicular thermal diffusivity, for chi_model='total'.
    chi_model : {'tau_E', 'electron', 'total'}

    Returns
    ----------
    xr.Dataset
        Input dataset with chi_perp_surf [m^2/s] added (and chi_ion_weight_surf zeroed where 'total' falls back).
    """

    if chi_model not in CHI_MODELS:
        raise ValueError(f"chi_model must be one of {CHI_MODELS}, got {chi_model!r}")
    if chi_model == 'tau_E':
        chi_perp_spline = None
    elif chi_model in ('electron', 'total') and chi_perp_spline is None:
        raise ValueError(f"chi_model={chi_model!r} needs chi_perp_spline.")
    if chi_model == 'total' and chi_perp_i_spline is None:
        raise ValueError("chi_model='total' needs chi_perp_i_spline.")

    if chi_perp_spline is not None: # Use chi_perp_spline
        chi_perp_on_modes = np.array(chi_perp_spline(rdcon_xarray.psi_n_rational.values))
        if chi_model == 'total':
            f_i = rdcon_xarray['chi_ion_weight_surf']
            total = (1 - f_i.values)*chi_perp_on_modes + f_i.values*np.array(chi_perp_i_spline(rdcon_xarray.psi_n_rational.values))
            ok = total > 0
            chi_perp_on_modes = np.where(ok, total, chi_perp_on_modes)
            rdcon_xarray = rdcon_xarray.assign(chi_ion_weight_surf=f_i.where(ok, 0.0))

    elif energy_confinement_time is None:
        raise ValueError("chi_model='tau_E' needs energy_confinement_time.")

    elif minor_radius is None: # **Assume all energy goes through each surface. This is the default option!
        minor_radii_squared = rdcon_xarray.avg_r_surf.values*rdcon_xarray.avg_r_surf.values
        chi_perp_on_modes = np.array(minor_radii_squared / (6*energy_confinement_time)) # Assumes all energy put into the plasma deposits in core (overestimates chi_perp inner surfaces)

    elif isinstance(minor_radius, float) and isinstance(areal_elongation, float): # ***Uses Fitzpatrick 1995 formula:
        chi_perp = areal_elongation * minor_radius**2 / (6*energy_confinement_time) # One value for all surfaces
        chi_perp_on_modes = np.array([chi_perp]*len(rdcon_xarray.psi_n_rational.values))

    else:
        raise ValueError("Incorrect entries to chi_perp function.")

    rdcon_xarray = rdcon_xarray.assign(chi_perp_surf=like(chi_perp_on_modes, rdcon_xarray['psi_n_rational']))

    # Recorded whichever source was used (NaN: none given)
    return rdcon_xarray.assign(energy_confinement_time=np.nan if energy_confinement_time is None else energy_confinement_time)
