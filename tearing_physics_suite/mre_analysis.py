# Python functions to call GPEC and PEST3 fortran codes for delta prime calculations

import os
import sys
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import CubicSpline
import tearing_physics_suite.global_vars as gv

# To do:
# Add pressure check (kinetic vs equilibrium)

def mre_raw_interp(rdcon_xarray):
    """
    Interpolate MRE terms onto rational surfaces using cubic splines. Requires mre_flag & geom_flag='t' (as per default) 
    when running RDCON.
    """
    #########################################################################################################
    # Put mre terms onto surfaces:
    #########################################################################################################
    Di_surf = rdcon_xarray.di.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    Dr_surf = rdcon_xarray.dr.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    H_surf = rdcon_xarray.h.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    Hbs_prefac_surf = rdcon_xarray.Hbs_prefac.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    taua_prefac_surf = rdcon_xarray.tau_a.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    taur_prefac_surf = rdcon_xarray.tau_r.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    ftr_surf = rdcon_xarray.ftr.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    mufrac_surf = rdcon_xarray.mufrac.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    #avg_mu0Jbs_dot_B_surf = rdcon_xarray.avg_mu0Jbs_dot_B.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    Dnc_surf = rdcon_xarray.Dnc.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    Wc_prefac_surf = rdcon_xarray.Wc.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_mu0Jpara_surf = rdcon_xarray.avg_mu0Jpara.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_B_surf = rdcon_xarray.avg_B.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_Bt_surf = rdcon_xarray.avg_Bt.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_Bp_surf = rdcon_xarray.avg_Bp.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_r_surf = rdcon_xarray.avg_r.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_R_surf = rdcon_xarray.avg_R.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_inv_R_surf = rdcon_xarray.avg_inv_R.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    overbar_Rsq_surf = rdcon_xarray.overbar_Rsq.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_Rsq_surf = rdcon_xarray.avg_Rsq.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_Bsq_on_nabla_psisq_surf = rdcon_xarray.avg_1.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    avg_Bsq_surf = rdcon_xarray.avg_5.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    #########################################################################################################
    # Load these surface values into the xarray:
    #########################################################################################################
    rdcon_xarray = rdcon_xarray.assign(
        Di_surf =Di_surf+0.0*rdcon_xarray['psi_n_rational'],
        Dr_surf =Dr_surf+0.0*rdcon_xarray['psi_n_rational'],
        H_surf =H_surf+0.0*rdcon_xarray['psi_n_rational'],
        Hbs_prefac_surf =Hbs_prefac_surf+0.0*rdcon_xarray['psi_n_rational'],
        taua_prefac_surf =taua_prefac_surf+0.0*rdcon_xarray['psi_n_rational'],
        taur_prefac_surf =taur_prefac_surf+0.0*rdcon_xarray['psi_n_rational'],
        ftr_surf =ftr_surf+0.0*rdcon_xarray['psi_n_rational'],
        mufrac_surf =mufrac_surf+0.0*rdcon_xarray['psi_n_rational'],
        #avg_mu0Jbs_dot_B_surf =avg_mu0Jbs_dot_B_surf+0.0*rdcon_xarray['psi_n_rational'], removed because broken!
        Dnc_surf =Dnc_surf+0.0*rdcon_xarray['psi_n_rational'],
        Wc_prefac_surf =Wc_prefac_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_mu0Jpara_surf =avg_mu0Jpara_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_B_surf =avg_B_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_Bt_surf =avg_Bt_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_Bp_surf =avg_Bp_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_r_surf =avg_r_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_R_surf =avg_R_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_inv_R_surf =avg_inv_R_surf+0.0*rdcon_xarray['psi_n_rational'],
        overbar_Rsq_surf =overbar_Rsq_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_Rsq_surf =avg_Rsq_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_Bsq_on_nabla_psisq_surf = avg_Bsq_on_nabla_psisq_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_Bsq_surf = avg_Bsq_surf+0.0*rdcon_xarray['psi_n_rational']
    )
    rdcon_xarray = rdcon_xarray.assign(fc_surf = 1-rdcon_xarray['ftr_surf'])
    return rdcon_xarray

def mre_terms_on_modes(rdcon_xarray,ni_spline,ne_spline,ti_spline,te_spline,average_ion_mass=2.5):
    """
    Calculate the MRE terms on modes using the provided xarray data and splines. This just 
    deals with values out of rdcon_xarray, and natural flux coordinates. Requires mre_flag & geom_flag='t' (as per default) 
    when running RDCON.
    
    Parameters:
    rdcon_xarray : xarray.DataArray
        The xarray containing the radial coordinate data.
    ni_spline : CubicSpline
        Spline for ion density in m^(-3) on normalised poloidal flux.
    ne_spline : CubicSpline
        Spline for electron density in m^(-3) on normalised poloidal flux.
    ti_spline : CubicSpline
        Spline for ion temperature in KeV on normalised poloidal flux.
    te_spline : CubicSpline
        Spline for electron temperature in KeV on normalised poloidal flux.
    
    Returns:
    rdcon_xarray : xarray.DataArray
        The updated xarray with MRE terms calculated.
    """

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
        ni_m3=ni_m3+0.0*rdcon_xarray['psi_n'],
        ne_m3=ne_m3+0.0*rdcon_xarray['psi_n'],
        ti_keV=ti_keV+0.0*rdcon_xarray['psi_n'],
        te_keV=te_keV+0.0*rdcon_xarray['psi_n']
    )

    # Put kinetic information onto surfaces:
    rdcon_xarray = rdcon_xarray.assign(ni_m3_surf =np.array(ni_spline(rdcon_xarray['psi_n_rational'].values))+0.0*rdcon_xarray['psi_n_rational'],
                                        ne_m3_surf =np.array(ne_spline(rdcon_xarray['psi_n_rational'].values))+0.0*rdcon_xarray['psi_n_rational'],
                                        ti_keV_surf =np.array(ti_spline(rdcon_xarray['psi_n_rational'].values))+0.0*rdcon_xarray['psi_n_rational'],
                                        te_keV_surf =np.array(te_spline(rdcon_xarray['psi_n_rational'].values))+0.0*rdcon_xarray['psi_n_rational'])
    
    # Check if average_ion_mass is in rdcon_xarray:
    if not 'average_ion_mass' in rdcon_xarray:
        rdcon_xarray = rdcon_xarray.assign(average_ion_mass=average_ion_mass) # Mass in units amu 

    # Thermal velocities in m/s. Note eV*e = joules, using Fitzpatrick 2023 Eq. 1.71-1.72 definition of thermal velocities
    rdcon_xarray = rdcon_xarray.assign(
        v_te_surf = np.sqrt(2* 
                            gv.e*(1e3*rdcon_xarray['te_keV_surf']) #Electron temp in joules
                            / gv.me), # Electron mass in kg
        v_ti_surf = np.sqrt(2* 
                            gv.e*(1e3*rdcon_xarray['ti_keV_surf']) #Ion temp in joules
                            / (rdcon_xarray.average_ion_mass.values*gv.amu))) # Average ion mass in kg

    # Coulomb Logarithm using Wesson Tokamaks page 727:
    rdcon_xarray = rdcon_xarray.assign(
        lnLamb_ee_surf = 14.9-0.5*np.log(rdcon_xarray['ne_m3_surf']/1e20)+np.log(rdcon_xarray['te_keV_surf']),  # Dimless
        lnLamb_ei_surf = 15.2-0.5*np.log(rdcon_xarray['ne_m3_surf']/1e20)+np.log(rdcon_xarray['te_keV_surf'])) # Dimless

    # Electron-ion collision time in seconds using Wesson Tokamaks page 729 assuming singly charged ions:
    rdcon_xarray = rdcon_xarray.assign(
        taue_surf = 1.09*(10**16)*(rdcon_xarray['te_keV_surf']**(3/2))*(1/rdcon_xarray['ne_m3_surf'])*(1/rdcon_xarray['lnLamb_ei_surf'])) # seconds
        
    # Resistivity in Ohm m from Wesson Tokamaks
    rdcon_xarray = rdcon_xarray.assign(
        eta_spitz_surf = 1.65*1e-9*rdcon_xarray['lnLamb_ei_surf']*(rdcon_xarray['te_keV_surf']**(-3/2))) # Ohm m

    # mu_e_on_nu_e from Callen, 2010 UW-CPTC 09-6R, taking banana limit of eq. B17 (& B14).
    Zeff = rdcon_xarray.Zeff
    rdcon_xarray = rdcon_xarray.assign(
        mu_e_on_nu_e_surf = (rdcon_xarray['ftr_surf']/rdcon_xarray['fc_surf'])*(1+0.533/Zeff)) #Dimless

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
        taur_surf = rdcon_xarray['taur_prefac_surf']/rdcon_xarray['eta_spitz_surf']) # seconds SEEMS TOO LONG...
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
        Q0_surf = rdcon_xarray['X0_surf']/rdcon_xarray['taua_surf']) # seconds 

    # Add mode number m for rational surfaces:
    m_ints = np.round(rdcon_xarray.n*rdcon_xarray['q_rational'].values).astype(int)
    rdcon_xarray = rdcon_xarray.assign(m_rational = m_ints + 0.0*rdcon_xarray['psi_n_rational'])

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

    return rdcon_xarray

def mre_flux_gradients(rdcon_xarray):
    """
    Calculate the gradient of certain values with respect to magnetic flux coordinate,
    on modes, using Cubic splines.
    """

    # Make cubic splines of terms I want to differentiate:
    q_spline = CubicSpline(rdcon_xarray.psi_n.values, rdcon_xarray.q.values,extrapolate=False)
    mu0p_spline = CubicSpline(rdcon_xarray.psi_n.values, rdcon_xarray.mu0p.values,extrapolate=False)

    # Calculate the gradients of these terms:
    dq_dpsi_n = q_spline.derivative()(rdcon_xarray.psi_n.values)
    dmu0p_dpsi_n = mu0p_spline.derivative()(rdcon_xarray.psi_n.values)

    # Calculate the gradients using finite differences
    # dq_dpsi_n_finite_diff = np.gradient(rdcon_xarray.q.values, rdcon_xarray.psi_n.values)
    # dmu0p_dpsi_n_finite_diff = np.gradient(rdcon_xarray.mu0p.values, rdcon_xarray.psi_n.values)

    # Put these gradients into the xarray:
    rdcon_xarray = rdcon_xarray.assign(
        dq_dpsi_n=dq_dpsi_n+0.0*rdcon_xarray['psi_n'],
        dmu0p_dpsi_n=dmu0p_dpsi_n+0.0*rdcon_xarray['psi_n']
    )

    # Put these gradients onto surfaces:
    dq_dpsi_n_surf = rdcon_xarray.dq_dpsi_n.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    dmu0p_dpsi_n_surf = rdcon_xarray.dmu0p_dpsi_n.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values

    rdcon_xarray = rdcon_xarray.assign(
        dq_dpsi_n_surf =dq_dpsi_n_surf+0.0*rdcon_xarray['psi_n_rational'],
        dmu0p_dpsi_n_surf =dmu0p_dpsi_n_surf+0.0*rdcon_xarray['psi_n_rational']
    )

    # Calculate the dimensionless flux shear factor (s in Fitz.)
    rdcon_xarray = rdcon_xarray.assign(
        flux_shear_s_surf = rdcon_xarray['psi_n_rational']*rdcon_xarray['dq_dpsi_n_surf']/rdcon_xarray['q_rational']
    )

    return rdcon_xarray
    
def extract_critical_mre_factors_on_modes(rdcon_xarray,Delta_prime_vec,use_cylindrical_terms=False, k0=0.8227, k1=1.7, C0=0.6):
    """
    For each rational surface, extract critical MRE factors including maximum island width, 
    location of maximum island width, and minimum marginally stable island width.
    Parameters:
    rdcon_xarray : xarray.DataSet after it has gone through mre_terms_on_modes.
    Delta_prime_vec : numpy array that is a 1D vector of Delta' values on rational surfaces.
    use_cylindrical_terms : bool, optional
        Whether to use cylindrical terms in the calculation.
    k0 = 0.8227 comes from private communication w. Eric Howell, but is near identical to LaHaye 2017 10.1051/epjconf/201715703027 Eq. 1.
    k1 = 1.7 comes from Chang et al. PRL 1995 
    C0 = 0.6 comes from Schlutt and Hegna PoP 2012
    Returns:
    rdcon_xarray : xarray.DataSet
    The updated xarray with critical MRE terms calculated.
    """
    # Check Delta_prime_vec is the right shape:
    if not (len(Delta_prime_vec) == len(rdcon_xarray.psi_n_rational.values)):
        raise ValueError("Delta_prime_vec must be a 1D numpy array with length equal to the number of rational surfaces in rdcon_xarray.")
    # This code should create a vector of w values, then construct the MRE for each w, pulling the minimum for diffusion etc...
    w_vec = np.logspace(-8,0,num=1000)
    w_vec_lowres = np.logspace(-5,0,num=200)
    # Make outputs numpy arrays to store the results:
    w_margs = np.full(len(rdcon_xarray.psi_n_rational.values), np.nan)
    w_sats = np.full(len(rdcon_xarray.psi_n_rational.values), np.nan)
    w_max_locs = np.full(len(rdcon_xarray.psi_n_rational.values), np.nan)
    dwdtau_maxs = np.full(len(rdcon_xarray.psi_n_rational.values), np.nan)
    prefacs = np.full(len(rdcon_xarray.psi_n_rational.values), np.nan)
    lowres_vecs = np.full((len(rdcon_xarray.psi_n_rational.values), len(w_vec_lowres)), np.nan)
    wd_at_margs = np.full(len(rdcon_xarray.psi_n_rational.values), np.nan)
    wd_at_X0s = np.full(len(rdcon_xarray.psi_n_rational.values), np.nan)
    for i in range(len(rdcon_xarray.psi_n_rational.values)):
        # Get the current surface's data
        rdcon_surf = rdcon_xarray.isel(r=i)
        wd_function = generate_wd_function(rdcon_surf)
        if not np.isnan(Delta_prime_vec[i]):
            dwdtau_loc = lambda w_in: dwdtau(w_in, wd_function, Delta_prime_vec[i], 
                                    rdcon_surf['Dr_surf'].values, rdcon_surf['Di_surf'].values, 
                                    rdcon_surf['Dnc_surf'].values, rdcon_surf['H_surf'].values, 
                                    k1, C0)
            dwdtau_vec = dwdtau_loc(w_vec)
            dwdtau_vec_low_res = dwdtau_loc(w_vec_lowres)
            prefac = rdcon_surf['eta_star_surf'].values/k0
            w_marg, w_sat, w_max_loc, dwdtau_max = extract_mre_factors(dwdtau_vec, w_vec)
            if not np.isnan(w_marg):
                wd_at_margs[i] = wd_function(w_marg)    
            wd_at_X0s[i] = wd_function(rdcon_surf['X0_surf'].values)
        else:
            # If Delta' is NaN, set all values to NaN:
            dwdtau_vec = np.full_like(w_vec, np.nan)
            dwdtau_vec_low_res = np.full_like(w_vec_lowres, np.nan)
            w_marg = np.nan
            w_sat = np.nan
            w_max_loc = np.nan
            dwdtau_max = np.nan
            prefac = np.nan
        # Store the results in the output arrays:
        w_margs[i] = w_marg
        w_sats[i] = w_sat
        w_max_locs[i] = w_max_loc
        dwdtau_maxs[i] = dwdtau_max
        prefacs[i] = prefac
        lowres_vecs[i,:] = prefac*dwdtau_vec_low_res # These have units psi_p_norm/s now.
    rdcon_xarray = rdcon_xarray.assign(
        w_marg_surf=w_margs+0.0*rdcon_xarray['psi_n_rational'],
        w_sat_surf=w_sats+0.0*rdcon_xarray['psi_n_rational'],
        w_max_loc_surf=w_max_locs+0.0*rdcon_xarray['psi_n_rational'],
        dwdtau_max_surf=dwdtau_maxs+0.0*rdcon_xarray['psi_n_rational'],
        prefac_surf=prefacs+0.0*rdcon_xarray['psi_n_rational'],
        wd_at_marg_surf = wd_at_margs+0.0*rdcon_xarray['psi_n_rational'],
        wd_at_X0s_surf = wd_at_X0s+0.0*rdcon_xarray['psi_n_rational']
    )
    rdcon_xarray = rdcon_xarray.assign(
        lowres_dwdt_surf=(
            ('r', 'w_bar'),
            lowres_vecs
        )
    )
    return rdcon_xarray

def extract_mre_factors(dwdtau_vec, w_vec): #Update with cubic spline?
    """
    Extracts the critical MRE factors from the dwdtau_vec and w_vec.
    Returns the marginally stable island width, saturated island width, 
    location of maximum island width, and the maximum dwdtau value.
    """
    # Find where dwdtau crosses zero:
    zero_crossings = np.where(np.diff(np.sign(dwdtau_vec)))[0]
    
    w_marg = np.nan
    w_sat = np.nan

    if len(zero_crossings) != 0:
        if dwdtau_vec[0] < 0:
            # Marginally stable island width is the first zero crossing:
            w_marg = w_vec[zero_crossings[0]]
        if dwdtau_vec[-1] < 0:
            # Saturated island width is the last zero crossing:
            w_sat = w_vec[zero_crossings[-1]]

    # Maximum island width is where dwdtau is maximum:
    max_index = np.argmax(dwdtau_vec)
    w_max_loc = w_vec[max_index]
    dwdtau_max = dwdtau_vec[max_index]

    return w_marg, w_sat, w_max_loc, dwdtau_max

def generate_wd_function(rdcon_xarray_surf,use_lmfp=False,iterator=False,use_Fitz_formula=False):
    """
    Generates for a particular surface, a function that takes in w_bar 
    (island width in normalised poloidal flux), and returns wd_bar
    (Fitzpatrick island width in normalised poloidal flux). 
    By default, takes the minimum of chi_para_lmfp and chi_para_smfp, 
    but can be set to use only chi_para_lmfp. Assumes that rdcon_xarray
    has already been through cross_field_transport.py.

    If use_Fitz_formula is True, then we apply the Fitzpatrick 2023 formula 14.209. Default is no, 
    since if the chi_para_smfp and chi_para_lmfp are equal, this formula cuts chi_para in half, which I don't agree with.

    If iterator is False, we calculate the ratio of chi_perp/chi_para at the specific island size being evaluated (I think this is more correct).
    """

    chi_perp = rdcon_xarray_surf['chi_perp_surf'].values
    chi_para_smfp = rdcon_xarray_surf['chi_para_smfp_surf'].values
    chi_para_lmfp_no_w = rdcon_xarray_surf['chi_para_lmfp_no_w_surf'].values
    Wc_prefac_m = rdcon_xarray_surf['Wc_prefac_m_surf'].values
    X0 = rdcon_xarray_surf['X0_surf'].values
    
    if not iterator: # I think this is more correct
        def wd_function(w_bar: float):
            """
            Takes in w_bar (island width in normalised poloidal flux) and returns wd_bar
            (Fitzpatrick island width in normalised poloidal flux).
            """
            chi_para_lmfp = chi_para_lmfp_no_w/w_bar
            if use_lmfp:
                chi_para=chi_para_lmfp
            else:
                if not use_Fitz_formula:
                    chi_para = np.minimum(chi_para_lmfp, chi_para_smfp)
                else:
                    chi_para = chi_para_lmfp*chi_para_smfp/(chi_para_lmfp+chi_para_smfp) # If the two are equal, it cuts chi_para in half, which I don't agree with.
            chifrac = chi_perp/chi_para
            return (chifrac*Wc_prefac_m)**(1/4)
    else:
        def wd_function_iterator(w_bar: float):
            """ !!! IGNORES w_bar !!!
            Takes in w_bar (island width in normalised poloidal flux) and returns wd_bar
            (Fitzpatrick island width in normalised poloidal flux). 
            """
            wd_bar4=X0**4
            for i in range(10): #Iterate to convergence
                chi_para_lmfp = chi_para_lmfp_no_w/(wd_bar4**(1/4))
                if not use_Fitz_formula:
                    chi_para = np.minimum(chi_para_lmfp, chi_para_smfp)
                else:
                    chi_para = chi_para_lmfp*chi_para_smfp/(chi_para_lmfp+chi_para_smfp)
                chifrac = chi_perp/chi_para
                wd_bar4 = Wc_prefac_m*chifrac
            return wd_bar4**(1/4)
    return wd_function

def dwdtau(w_bar: float, wd_function: 'function', DeltaPrimeGPEC: float, Dr: float, Di: float, Dnc: float, H: float, k1: float, C0: float):
    """
    Returns the right hand side of the MRE from Schlutt and Hegna PoP 2012 & Hegna 1999, using normalised
    poloidal flux space as per Rosenburg PoP 2002. Units are psi_p_norm^(-1)
    """
    wd_bar=wd_function(w_bar)
    return DeltaPrime_bar(w_bar, DeltaPrimeGPEC, Di) + Delta_GGJ(w_bar, wd_bar, Dr, Di, H, k1, C0) + Delta_nc(w_bar, wd_bar, Dnc, k1, C0)
        
def Delta_nc(w_bar: float, wd_bar: float, Dnc: float, k1: float, C0: float):
    """
    Calculates the neoclassical bootstrap drive terms in the MRE from Schlutt and Hegna PoP 2012,
    converted to normalised poloidal flux space. This whole term has units psi_norm^(-1).
    """
    return k1*Dnc*w_bar/(w_bar**2+(wd_bar**2)*k1/(C0*0.81))

def Delta_GGJ(w_bar: float, wd_bar: float, Dr: float, Di: float, H: float, k1: float, C0: float):
    """
    Calculates the curvature stabilisation term in the MRE from Schlutt and Hegna PoP 2012,
    converted to normalised poloidal flux space. Note typo in that paper; to agree with Hegna 1999 in 
    the toroidal limit, we use k1 instead of k0. This whole term has units psi_norm^(-1).
    """
    alpha_l=0.5-np.sqrt(-Di)
    alpha_s=0.5+np.sqrt(-Di)
    Dh = Dr/(alpha_s-H)
    denom = w_bar + (2*k1*wd_bar)/(C0*(1+alpha_s))
    return k1*Dh/denom

def DeltaPrime_bar(w_bar: float, DeltaPrimeGPEC: float, Di: float):
    """
    Calculates the delta prime term for the MRE from Hegna PoP 1999 & Schlutt and Hegna PoP 2012
    (note the typo in the latter), converted to normalised poloidal flux space as per Rosenburg PoP 2002. 
    Note DeltaPrimeGPEC has units psi_norm^{-2sqrt(-Di)}. This whole term has units psi_norm^(-1).
    """
    alpha_l=0.5-np.sqrt(-Di)
    return DeltaPrimeGPEC*(w_bar/2)**(-2*alpha_l)*np.sqrt(-4*Di)

