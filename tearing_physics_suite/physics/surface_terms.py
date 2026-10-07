import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

import tearing_physics_suite.physics.constants as gv
from tearing_physics_suite.physics.rotation import (
    add_drift_rotation,
    add_rotation,
    decorrelation_ratios,
    decorrelation_timescales,
)
from tearing_physics_suite.physics.xr_utils import interp_to_surfaces, like

# RDCON profile (on psi_n) -> its value on the rational surfaces (mre_raw_interp)
_RAW_TO_SURF = {
    'dvdpsi': 'dvdpsi_n_surf', 'di': 'Di_surf', 'dr': 'Dr_surf', 'h': 'H_surf', 'Hbs_prefac': 'Hbs_prefac_surf',
    'tau_a': 'taua_prefac_surf', 'tau_r': 'taur_prefac_surf', 'ftr': 'ftr_surf', 'mufrac': 'mufrac_surf',
    'avg_nabla_psi': 'avg_nabla_psi_surf', 'Dnc': 'Dnc_surf', 'Wc': 'Wc_prefac_surf',
    'avg_mu0Jpara': 'avg_mu0Jpara_surf', 'avg_B': 'avg_B_surf', 'avg_Bt': 'avg_Bt_surf', 'avg_Bp': 'avg_Bp_surf',
    'avg_r': 'avg_r_surf', 'avg_R': 'avg_R_surf', 'avg_inv_R': 'avg_inv_R_surf', 'overbar_Rsq': 'overbar_Rsq_surf',
    'avg_Rsq': 'avg_Rsq_surf', 'avg_1': 'avg_Bsq_on_nabla_psisq_surf', 'avg_5': 'avg_Bsq_surf',
    'avg_7': 'avg_dpsisq_surf',
}


def mre_raw_interp(rdcon_xarray):
    """Interpolate MRE terms onto rational surfaces using cubic splines.

    Requires mre_flag & geom_flag='t' (as per default) when running RDCON.
    Interpolates Di, Dr, H, Dnc, tau_a, tau_r, ftr, mufrac, Wc, avg_B, avg_Bp, avg_Bt,
    avg_r, avg_R and other geometric/MRE quantities from the full psi grid onto psi_n_rational.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Dataset from RDCON with full-grid MRE and geometry variables.

    Returns
    -------
    xr.Dataset
        Input dataset with _surf variables added at rational surface locations.
    """
    #########################################################################################################
    # Put mre terms onto surfaces:
    #########################################################################################################
    surf = interp_to_surfaces(rdcon_xarray, _RAW_TO_SURF)
    # Zeff gets special treatment:
    # Handle Zeff - it may be a DataArray or already a numpy array
    psi_N_Zeff_vals = rdcon_xarray.psi_N_Zeff.values if hasattr(rdcon_xarray.psi_N_Zeff, 'values') else rdcon_xarray.psi_N_Zeff
    Zeff_vals = rdcon_xarray.Zeff.values if hasattr(rdcon_xarray.Zeff, 'values') else rdcon_xarray.Zeff
    Zeff_spline = PchipInterpolator(psi_N_Zeff_vals, Zeff_vals, extrapolate=False) # pchip, as GPEC/rdcon/mercier.f (spline_fit_pchip)
    Zeff_surf = Zeff_spline(rdcon_xarray['psi_n_rational'].values)
    #########################################################################################################
    # Load these surface values into the xarray:
    #########################################################################################################
    rdcon_xarray = rdcon_xarray.assign(**{_RAW_TO_SURF[k]: v for k, v in surf.items()},
                                       Zeff_surf=like(Zeff_surf, rdcon_xarray['psi_n_rational']))
    rdcon_xarray = rdcon_xarray.assign(fc_surf = 1-rdcon_xarray['ftr_surf'])
    return rdcon_xarray


def get_X0s_and_DeltaPrime_crit(eta,mass_densities,n,
                                taur_prefac_surf,taua_prefac_surf,DeltaPrime_crits_no_X0,H_surf):
    """Calculate X0, S, taua, taur, and Delta_prime_crit for comparison with resistive MHD simulations.

    Takes simulation values (resistivity, mass density) and pre-calculated equilibrium terms
    at rational surfaces to compute dimensionless MRE parameters.

    Parameters
    ----------
    eta : array-like
        Spitzer resistivity in Ohm*m at each rational surface.
    mass_densities : array-like
        Mass density in kg/m^3 at each rational surface.
    n : int
        Toroidal mode number.
    taur_prefac_surf, taua_prefac_surf : array-like
        Resistive and Alfven time prefactors at rational surfaces (from RDCON).
    DeltaPrime_crits_no_X0 : array-like
        Critical Delta' values before X0 correction.
    H_surf : array-like
        H parameter (curvature/pressure gradient term) at rational surfaces.

    Returns
    -------
    X0s, Ss, tauas, taurs, DeltaPrime_crits : np.ndarray
        Dimensionless parameters and critical Delta' at each rational surface.
    """
    assert len(eta) == len(mass_densities) == len(taur_prefac_surf) == len(taua_prefac_surf) == len(DeltaPrime_crits_no_X0) == len(H_surf), "All input arrays must be the same length."
    X0s = np.zeros(len(eta))
    Ss = np.zeros(len(eta))
    tauas = np.zeros(len(eta))
    taurs = np.zeros(len(eta))
    DeltaPrime_crits = np.zeros(len(eta))
    for i in range(len(eta)):
        tauas[i] = taua_prefac_surf[i]*np.sqrt(mass_densities[i])/n
        taurs[i] = taur_prefac_surf[i]/eta[i]
        Ss[i] = taurs[i]/tauas[i]
        X0s[i] = Ss[i]**(-1/3)
        DeltaPrime_crits[i] = DeltaPrime_crits_no_X0[i]*(1.0/X0s[i])**(1-2*H_surf[i])
    return X0s, Ss, tauas, taurs, DeltaPrime_crits


def res_func(rdcon_xarray, eta_fac=1.0, Coulomb_logarithm=None):
    """Compute Spitzer resistivity and Coulomb logarithm on rational surfaces and full psi grid.

    Adds variables eta_spitz, eta_spitz_surf, lnLamb_ei, lnLamb_ei_surf (and ee variants)
    to rdcon_xarray using Wesson Tokamaks formulae.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Must contain ne_m3, te_keV (full grid) and ne_m3_surf, te_keV_surf (on rational surfaces).
    eta_fac : float
        Multiplicative factor applied to Spitzer resistivity (e.g. to match simulation values).
    Coulomb_logarithm : float or None
        If provided, overrides the Wesson formula with a fixed value.

    Returns
    -------
    xr.Dataset
        Input dataset with resistivity variables added.
    """
    # Keep up to date with res_func in equilibrium_helper
    if Coulomb_logarithm is None:
        # Coulomb Logarithm using Wesson Tokamaks page 727:
        rdcon_xarray = rdcon_xarray.assign(
            lnLamb_ee_surf = 14.9-0.5*np.log(rdcon_xarray['ne_m3_surf']/1e20)+np.log(rdcon_xarray['te_keV_surf']),  # Dimless
            lnLamb_ei_surf = 15.2-0.5*np.log(rdcon_xarray['ne_m3_surf']/1e20)+np.log(rdcon_xarray['te_keV_surf']),  # Dimless
            lnLamb_ee = 14.9-0.5*np.log(rdcon_xarray['ne_m3']/1e20)+np.log(rdcon_xarray['te_keV']),                  # Dimless
            lnLamb_ei = 15.2-0.5*np.log(rdcon_xarray['ne_m3']/1e20)+np.log(rdcon_xarray['te_keV'])                  # Dimless
        )
    else:
        rdcon_xarray = rdcon_xarray.assign(
            lnLamb_ei_surf = like(Coulomb_logarithm, rdcon_xarray['psi_n_rational']), # Dimless
            lnLamb_ei = like(Coulomb_logarithm, rdcon_xarray['psi_n'])                # Dimless
        )
    # Resistivity in Ohm m from Wesson Tokamaks
    rdcon_xarray = rdcon_xarray.assign(
        eta_spitz_surf = eta_fac*1.65*1e-9*rdcon_xarray['lnLamb_ei_surf']*(rdcon_xarray['te_keV_surf']**(-3/2)),    # Ohm m
        eta_spitz = eta_fac*1.65*1e-9*rdcon_xarray['lnLamb_ei']*(rdcon_xarray['te_keV']**(-3/2))                    # Ohm m
    )
    return rdcon_xarray


def mre_terms_on_modes(rdcon_xarray,ni_spline,ne_spline,ti_spline,te_spline,average_ion_mass=2.5,Coulomb_logarithm=None,eta_fac=1.0,Er_spline=None,omega_splines=None,q_surfs_of_interest=None,psi_surfs_of_interest=None,diamagnetic_rotation_ion_charge=None):
    """
    Calculate the MRE terms on modes using the provided xarray data and splines. This just
    deals with values out of rdcon_xarray, and natural flux coordinates. Requires mre_flag & geom_flag='t' (as per default)
    when running RDCON.

    Parameters:
    rdcon_xarray : xarray.DataArray
        The xarray containing the radial coordinate data.
    ni_spline : 1DSpline
        Spline for ion density in m^(-3) on normalised poloidal flux.
    ne_spline : 1DSpline
        Spline for electron density in m^(-3) on normalised poloidal flux.
    ti_spline : 1DSpline
        Spline for ion temperature in KeV on normalised poloidal flux.
    te_spline : 1DSpline
        Spline for electron temperature in KeV on normalised poloidal flux.

    Returns:
    rdcon_xarray : xarray.DataArray
        The updated xarray with MRE terms calculated.
    """
    if psi_surfs_of_interest is None:
        psi_surfs_of_interest = [0.95]
    if q_surfs_of_interest is None:
        q_surfs_of_interest = [1.0]

    rdcon_xarray = mre_raw_interp(rdcon_xarray)
    rdcon_xarray = mre_flux_gradients(rdcon_xarray)

    # Unit sanity checks:
    if ni_spline(0.1) < 1e11 or ni_spline(0.1) > 1e25:
        raise ValueError("Ion density spline must be in m^(-3).")
    if ne_spline(0.1) < 1e11 or ne_spline(0.1) > 1e25:
        raise ValueError("Electron density spline must be in m^(-3).")
    if ti_spline(0.1) < 0.05 or ti_spline(0.1) > 100:
        raise ValueError("Ion temperature spline must be in KeV.")
    if te_spline(0.1) < 0.05 or te_spline(0.1) > 100:
        raise ValueError("Electron temperature spline must be in KeV.")

    # Put kinetic information into rdcon_xarray in it's full form:
    ni_m3 = ni_spline(rdcon_xarray['psi_n'].values) # Ion density in m^(-3)
    ne_m3 = ne_spline(rdcon_xarray['psi_n'].values) # Electron density in m^(-3)
    ti_keV = ti_spline(rdcon_xarray['psi_n'].values) # Ion temperature in KeV
    te_keV = te_spline(rdcon_xarray['psi_n'].values) # Electron temperature in KeV

    rdcon_xarray = rdcon_xarray.assign(
        ni_m3=like(ni_m3, rdcon_xarray['psi_n']),
        ne_m3=like(ne_m3, rdcon_xarray['psi_n']),
        ti_keV=like(ti_keV, rdcon_xarray['psi_n']),
        te_keV=like(te_keV, rdcon_xarray['psi_n'])
    )

    # Put kinetic information onto surfaces:
    rdcon_xarray = rdcon_xarray.assign(ni_m3_surf =like(np.array(ni_spline(rdcon_xarray['psi_n_rational'].values)), rdcon_xarray['psi_n_rational']),
                                        ne_m3_surf =like(np.array(ne_spline(rdcon_xarray['psi_n_rational'].values)), rdcon_xarray['psi_n_rational']),
                                        ti_keV_surf =like(np.array(ti_spline(rdcon_xarray['psi_n_rational'].values)), rdcon_xarray['psi_n_rational']),
                                        te_keV_surf =like(np.array(te_spline(rdcon_xarray['psi_n_rational'].values)), rdcon_xarray['psi_n_rational']))

    # Put gradients of kinetic information onto surfaces:
    rdcon_xarray = rdcon_xarray.assign(ni1_m3_surf = like(np.array(ni_spline(rdcon_xarray['psi_n_rational'].values,1)), rdcon_xarray['psi_n_rational']),
                                        ne1_m3_surf = like(np.array(ne_spline(rdcon_xarray['psi_n_rational'].values,1)), rdcon_xarray['psi_n_rational']),
                                        ti1_keV_surf = like(np.array(ti_spline(rdcon_xarray['psi_n_rational'].values,1)), rdcon_xarray['psi_n_rational']),
                                        te1_keV_surf = like(np.array(te_spline(rdcon_xarray['psi_n_rational'].values,1)), rdcon_xarray['psi_n_rational']))

    # Check if average_ion_mass is in rdcon_xarray:
    if 'average_ion_mass' not in rdcon_xarray:
        rdcon_xarray = rdcon_xarray.assign(average_ion_mass=average_ion_mass) # Mass in units amu

    # Thermal velocities in m/s. Note eV*e = joules, using Fitzpatrick 2023 Eq. 1.71-1.72 definition of thermal velocities
    rdcon_xarray = rdcon_xarray.assign(
        v_te_surf = np.sqrt(2*
                            gv.e*(1e3*rdcon_xarray['te_keV_surf']) #Electron temp in joules
                            / gv.me), # Electron mass in kg
        v_ti_surf = np.sqrt(2*
                            gv.e*(1e3*rdcon_xarray['ti_keV_surf']) #Ion temp in joules
                            / (rdcon_xarray.average_ion_mass.values*gv.amu))) # Average ion mass in kg

    rdcon_xarray = res_func(rdcon_xarray, eta_fac=eta_fac, Coulomb_logarithm=Coulomb_logarithm)

    # Electron-ion collision time in seconds using Wesson Tokamaks page 729 assuming singly charged ions:
    rdcon_xarray = rdcon_xarray.assign(
        taue_surf = 1.09*(10**16)*(rdcon_xarray['te_keV_surf']**(3/2))*(1/rdcon_xarray['ne_m3_surf'])*(1/rdcon_xarray['lnLamb_ei_surf'])) # seconds

    # mu_e_on_nu_e from Callen, 2010 UW-CPTC 09-6R, taking banana limit of eq. B17 (& B14).
    rdcon_xarray = rdcon_xarray.assign(
        mu_e_on_nu_e_surf = (rdcon_xarray['ftr_surf']/rdcon_xarray['fc_surf'])*(1+0.533/rdcon_xarray['Zeff_surf'])) #Dimless

    # Neoclassical resistivity from Spitzer resistivity:
    rdcon_xarray = rdcon_xarray.assign(
        eta_nc_surf = rdcon_xarray['eta_spitz_surf']*(1+rdcon_xarray['mu_e_on_nu_e_surf']), # Ohm m
        eta_nc2_surf = rdcon_xarray['eta_spitz_surf']/rdcon_xarray['fc_surf'] # Ohm m
    )

    # Neoclassical resistive diffusivity in flux space (Hegna 1999) (units psi_p_norm^2/s):
    rdcon_xarray = rdcon_xarray.assign(
        eta_star_surf = (1/gv.mu0)*rdcon_xarray['eta_nc_surf'] # resistive diffusivity in m^2/s
                    *rdcon_xarray['avg_Bsq_surf']/rdcon_xarray['avg_Bsq_on_nabla_psisq_surf'] # psi_p_norm^2/m^2
    )

    # Mass density in kg/m^3
    rdcon_xarray = rdcon_xarray.assign(
        rho_surf = rdcon_xarray['ni_m3_surf']*rdcon_xarray['average_ion_mass']*gv.amu + rdcon_xarray['ne_m3_surf']*gv.me) # kg/m^3

    # The following are from Glasser et al. 2016 Appendix A12-A16:
    #  Resistive diffusion time in seconds
    rdcon_xarray = rdcon_xarray.assign(
        taur_surf = rdcon_xarray['taur_prefac_surf']/rdcon_xarray['eta_spitz_surf']) #mu0 is included in taur_prefac_surf
    #  Alven time in seconds
    rdcon_xarray = rdcon_xarray.assign(
        taua_surf = rdcon_xarray['taua_prefac_surf']*np.sqrt(rdcon_xarray['rho_surf'])/rdcon_xarray.n) # seconds
    #  Lundquist number
    rdcon_xarray = rdcon_xarray.assign(
        S_surf = rdcon_xarray['taur_surf']/rdcon_xarray['taua_surf']) # dimless
    #  Characteristic resistive linear layer width in normalised flux space
    rdcon_xarray = rdcon_xarray.assign(
        X0_surf = rdcon_xarray['S_surf']**(-1/3)) # dimless
    #  Characteristic timescale of linear resistive mode growth
    rdcon_xarray = rdcon_xarray.assign(
        Q0_surf = rdcon_xarray['X0_surf']/rdcon_xarray['taua_surf']) # 1 / seconds

    # Add mode number m for rational surfaces:
    m_ints = np.round(rdcon_xarray.n*rdcon_xarray['q_rational'].values).astype(int)
    rdcon_xarray = rdcon_xarray.assign(m_rational = like(m_ints, rdcon_xarray['psi_n_rational']))

    # Add mode number m to Wc_prefacs
    # To get wd, just multiply Wc_prefac_m_surf by chi frac, then take to power (1/4):
    rdcon_xarray = rdcon_xarray.assign(
        Wc_prefac_m_surf = rdcon_xarray['Wc_prefac_surf']/(rdcon_xarray['m_rational']**2)
    )

    """ Sanity check comparing the formulas in GPEC/rdcon/resist.f to the Wesson formulas used above:
        Glasser values:
            we assume ne = 1e14 is actually 1e20 m^(-3) based on rho=ne*mi*1e6, and we assume te = 3e3 = 3KeV
            e=1.6021917e-19
            mp=1.672614e-27
            me=9.1091e-31
            mi=2*mp
            ne=1e14
            te=3e3
            lambd = (24-0.5*np.log(ne)+np.log(te))
            taue=3.44e5*te**1.5/(ne*lambd)
            eta=me/(ne*1e6*e**2*taue*1.96)
            rho=ne*mi*1e6
            taur=
        Wesson values:
            we use using ne = 1e20, and te = 3e3 = 3 KeV
            lambdW = 15.2-0.5*np.log(1e20/1e20)+np.log(3)
            taueW=1.09*(10**16)*(3**(3/2))/(1e20*lambdW) # seconds
            mu0=4*np.pi*1e-7
            etaW=1.65*1e-9*lambdW*(3**(-3/2))
            rhoW=1e20*mi
    """

    if omega_splines is not None:
        rdcon_xarray = add_rotation(rdcon_xarray,omega_splines=omega_splines)

    rdcon_xarray = add_drift_rotation(rdcon_xarray,diamagnetic_rotation_ion_charge=diamagnetic_rotation_ion_charge,Er_spline=Er_spline)
    rdcon_xarray = decorrelation_timescales(rdcon_xarray,
                        q_surfs_of_interest=q_surfs_of_interest,
                        psi_surfs_of_interest=psi_surfs_of_interest,
                        omega_splines=omega_splines)
    rdcon_xarray = decorrelation_ratios(rdcon_xarray)

    return rdcon_xarray


def mre_flux_gradients(rdcon_xarray):
    """Compute psi_n gradients of q and mu0*p, and the dimensionless flux shear factor s.

    Also integrates dV/dpsi to get enclosed plasma volume at each rational surface.

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Dataset with q, mu0p, dvdpsi on the full psi_n grid.

    Returns
    -------
    xr.Dataset
        Input dataset with dq_dpsi_n_surf, dmu0p_dpsi_n_surf, flux_shear_s_surf,
        and V_surf variables added.
    """

    # Make cubic splines of terms I want to differentiate:
    q_spline = Akima1DInterpolator(rdcon_xarray.psi_n.values, rdcon_xarray.q.values,extrapolate=False)
    mu0p_spline = Akima1DInterpolator(rdcon_xarray.psi_n.values, rdcon_xarray.mu0p.values,extrapolate=False)

    # Calculate the gradients of these terms:
    dq_dpsi_n = q_spline.derivative()(rdcon_xarray.psi_n.values)
    dmu0p_dpsi_n = mu0p_spline.derivative()(rdcon_xarray.psi_n.values)

    # Calculate the gradients using finite differences
    # dq_dpsi_n_finite_diff = np.gradient(rdcon_xarray.q.values, rdcon_xarray.psi_n.values)
    # dmu0p_dpsi_n_finite_diff = np.gradient(rdcon_xarray.mu0p.values, rdcon_xarray.psi_n.values)

    # Put these gradients into the xarray:
    rdcon_xarray = rdcon_xarray.assign(
        dq_dpsi_n=like(dq_dpsi_n, rdcon_xarray['psi_n']),
        dmu0p_dpsi_n=like(dmu0p_dpsi_n, rdcon_xarray['psi_n'])
    )

    # Put these gradients onto surfaces:
    dq_dpsi_n_surf = rdcon_xarray.dq_dpsi_n.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    dmu0p_dpsi_n_surf = rdcon_xarray.dmu0p_dpsi_n.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values

    rdcon_xarray = rdcon_xarray.assign(
        dq_dpsi_n_surf =like(dq_dpsi_n_surf, rdcon_xarray['psi_n_rational']),
        dmu0p_dpsi_n_surf =like(dmu0p_dpsi_n_surf, rdcon_xarray['psi_n_rational'])
    )

    # Calculate the dimensionless flux shear factor (s in Fitz.)
    rdcon_xarray = rdcon_xarray.assign(
        flux_shear_s_surf = rdcon_xarray['psi_n_rational']*rdcon_xarray['dq_dpsi_n_surf']/rdcon_xarray['q_rational']
    )

    # Get plasma volumes on surfaces by integrating dVdpsi:
    dVdpsi_spline = Akima1DInterpolator(rdcon_xarray.psi_n.values, rdcon_xarray.dvdpsi.values, extrapolate=False)
    min_psi_n = rdcon_xarray.psi_n.values.min()
    V_surf = [dVdpsi_spline.integrate(min_psi_n, i) for i in rdcon_xarray.psi_n_rational.values]
    rdcon_xarray = rdcon_xarray.assign(
        V_surf = like(np.array(V_surf), rdcon_xarray['psi_n_rational'])
    )
    return rdcon_xarray


def deltaprime_crit_on_modes(rdcon_xarray, force_lmfp=False):
    """Calculate the linear critical Delta' for tearing instability onset on each rational surface.

    Implements two formulations:
    1. Glasser, Greene & Johnson, Phys. Fluids 1975, Eq. 111
    2. Connor, Hastie & Helander, PPCF 2015, Eq. 59

    Parameters
    ----------
    rdcon_xarray : xr.Dataset
        Dataset with H_surf, Dr_surf, X0_surf, and chi_para variables.
    force_lmfp : bool
        If True, always use the long-mean-free-path chi_para formulation.

    Returns
    -------
    xr.Dataset
        Input dataset with DeltaPrime_crit_GGJ_surf, DeltaPrime_crit_CHH_surf,
        and related variables added.
    """
    import math

    #########################################################################################################
    # Flux surface quantities needed:
    #########################################################################################################
    Hs = rdcon_xarray['H_surf'].values
    Drs = rdcon_xarray['Dr_surf'].values
    X0s = rdcon_xarray['X0_surf'].values
    v_rationals = rdcon_xarray['V_surf'].values
    v1_rationals = rdcon_xarray['dvdpsi_n_surf'].values
    q1_rationals = rdcon_xarray['dq_dpsi_n_surf'].values
    # For Connor et al. 2015:
    avg_dpsisq_surf = rdcon_xarray['avg_dpsisq_surf'].values
    avg_Bsq_surf = rdcon_xarray['avg_Bsq_surf'].values
    chi_perp_surf = rdcon_xarray['chi_perp_surf'].values
    chi_para_smfp_surf = rdcon_xarray['chi_para_smfp_surf'].values
    chi_para_lmfp_noisland_surf = rdcon_xarray['chi_para_lmfp_noisland_surf'].values
    psio = rdcon_xarray.psio
    n = rdcon_xarray.n

    #########################################################################################################
    # Starting loop:
    #########################################################################################################
    Qcrits = np.full_like(Hs, np.nan)
    DeltaPrimeCrits75 = np.full_like(Hs, np.nan)
    DeltaPrimeCrits75_no_X0 = np.full_like(Hs, np.nan)
    DeltaPrimeCrits15 = np.full_like(Hs, np.nan)
    DeltaPrimeCrits15_no_chifrac = np.full_like(Hs, np.nan)

    for i in range(len(Hs)):
        if not (Hs[i] < 0.5 or Hs[i] > -5/2): # Condition for validity for these formulas
            continue
    #########################################################################################################
    # Glasser et al. Phys. Fluids 1975, Eq 111:
    #########################################################################################################
        Qcrit=abs(math.gamma(3/4)*math.gamma(1/2-Hs[i]/4)**2*math.gamma(1/4-Hs[i]/2)*math.sin((1-2*Hs[i])*math.pi/8)*Drs[i]/
                (math.gamma(1/4)*math.gamma(1-Hs[i]/4)**2*math.gamma(3/4-Hs[i]/2)*math.sin((5+2*Hs[i])*math.pi/8)*4))**(2.0/3.0)
        surface_factor = 2*v_rationals[i]/(X0s[i]*v1_rationals[i])
        DeltaPrimeCrits75[i]=math.pi*surface_factor**(1-2*Hs[i])*math.gamma(1/4)*math.gamma(1-Hs[i]/4)**2*math.gamma(3/4-Hs[i]/2)*Qcrit**((2*Hs[i]+5)/4)/((np.sqrt(2)*(1-2*Hs[i])*math.sin((1-2*Hs[i])*math.pi/8))*(1-Hs[i]/2)*(math.cos(Hs[i]*math.pi/2)*math.gamma((1+Hs[i])/4)*math.gamma(1-Hs[i]))**2)
        DeltaPrimeCrits75_no_X0[i] = DeltaPrimeCrits75[i]*(X0s[i]**(1-2*Hs[i]))
        Qcrits[i] = Qcrit
    #########################################################################################################
    # Connor et al. PPCF 2015, Eq 59, many terms defined in Glasser 1975:
    #########################################################################################################
        Lambda = -4*np.pi**2*psio**2*q1_rationals[i]/(v1_rationals[i]**3)
        alpha = v1_rationals[i]*n/psio
        avg_dVsq = avg_dpsisq_surf[i]*(v1_rationals[i]**2)
        # Find the minimum of chi_para_smfp_surf and chi_para_lmfp_noisland_surf
        if not force_lmfp:
            chi_para = min(chi_para_smfp_surf[i], chi_para_lmfp_noisland_surf[i]) # Choose the smaller mean free path (either set by electron-ion collisions, or helical connection length at rational surface)
        else:
            chi_para = chi_para_lmfp_noisland_surf[i]
        chi_frac_noisland = chi_para / chi_perp_surf[i]
        DeltaPrimeCrits15[i] = (1/2)*np.pi**(3/2)*chi_frac_noisland**(1/4)*v_rationals[i]*(-Drs[i])*(alpha*alpha*Lambda*Lambda/(avg_Bsq_surf[i]*avg_dVsq))**(1/4)
        DeltaPrimeCrits15_no_chifrac[i] = DeltaPrimeCrits15[i]/(chi_frac_noisland**(1/4))

    rdcon_xarray = rdcon_xarray.assign(
        Qcrit_surf = like(Qcrits, rdcon_xarray['psi_n_rational']), # Glasser et al. Phys. Fluids 1975, Eq 110.
        Delta_prime_crit = like(DeltaPrimeCrits75, rdcon_xarray['psi_n_rational']), # Glasser et al. Phys. Fluids 1975, Eq 111.
        DeltaPrime_crit_no_X0 = like(DeltaPrimeCrits75_no_X0, rdcon_xarray['psi_n_rational']), # Multiply by (1/X0)^(1-2Hs) to get Delta_prime_crit if you are modifying resistivity and/or mass density.
        Delta_prime_tcrit = like(DeltaPrimeCrits15, rdcon_xarray['psi_n_rational']),  # Connor et al. PPCF 2015, Eq 59. Requires small Dr, small H assumption to be valid (generally true, see Benjamin et al., NF 2025).
        Delta_prime_tcrit_no_chifrac = like(DeltaPrimeCrits15_no_chifrac, rdcon_xarray['psi_n_rational'])  # Multiply by (chi_para/chi_perp)^(1/4) to get DeltaPrime_crit2 if you are modifying transport coefficients.
    )

    # All we need for S, X0, and Delta_prime_crit, in m3dc1 is n, eta(spitz or otherwise), and mass density (ni, ion mass, ne - see mre_terms_on_modes for formula.)
    # All we need for Delta_prime_tcrit is chi_frac. Note chi_frac in theory depends on Zeff, but if chifrac is being artificially set by M3DC1, we don't need to worry about it for Delta_prime_tcrit.

    return rdcon_xarray
