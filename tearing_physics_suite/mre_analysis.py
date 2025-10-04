# Python functions to construct and analyse the modified Rutherford equation on modes

import os
import sys
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import CubicSpline #, make_interp_spline
from scipy.signal import find_peaks
import tearing_physics_suite.global_vars as gv
from tearing_physics_suite.fortran_wrappers import run_resistive_calculation
from tearing_physics_suite.cross_field_transport import chi_para_lmfp_no_w_on_modes, chi_para_smfp_on_modes, chi_para_lmfp_noisland_on_modes, chi_perp_on_modes
from tearing_physics_suite.delta_prime_extraction import extract_delta_primes
# To do:
# Add pressure check (kinetic vs equilibrium)

def analyse_with_mre(eq_filename, nn, ni_spline, ne_spline, te_keV_spline, ti_keV_spline,
        energy_confinement_time = None,
        chi_perp_spline=None,
        k0=0.8227,
        k1=1.7,
        C0=0.6,
        wd_static=False, # Set true to ignore the variation in the ratio of perpendicular to parallel transport across the island, as island width varies
        debug_mre_terms=False,
        debug=False,
        delete_attrs=True,
        **kwargs):
    """ 
    Big function that calculates Delta primes with run_resistive_calculation, then runs analysis on output deltaprimes, returning
    a fully combined xarray.
    """ 

    # Check that either energy_confinement_time or chi_perp_spline is defined:
    assert not (chi_perp_spline is None and energy_confinement_time is None), "Must either define energy_confinement_time or chi_perp_spline for MRE analysis."

    #########################################################################################################
    # Run resistive delta prime calculation: 
    #########################################################################################################
    rdcon_xr, stride_xr, pest3_xr, rdcon_ran, stride_ran, pest3_ran, rdcon_stride_input_dict, pest3_input_dict = run_resistive_calculation(eq_filename,nn,**kwargs)

    # Combine input dictionaries:
    if not (rdcon_stride_input_dict is None): #RDCON dict present
        if not (pest3_input_dict is None): # PEST3 dict present
            rdcon_stride_input_dict.update(pest3_input_dict)
        input_dict = rdcon_stride_input_dict
    elif not (pest3_input_dict is None):
        input_dict = pest3_input_dict
    else:
        input_dict = {}

    # Add wd_static, energy_confinement_time, k0, k1, C0, wd_static to input_dict:
    input_dict['wd_static'] = wd_static
    input_dict['energy_confinement_time'] = energy_confinement_time
    input_dict['k0'] = k0
    input_dict['k1'] = k1
    input_dict['C0'] = C0

    #########################################################################################################
    # Fill out rdcon_xr with important MRE terms:
    #########################################################################################################
    rdcon_xr = mre_terms_on_modes(rdcon_xr, ni_spline, ne_spline, te_keV_spline, ti_keV_spline)
    rdcon_xr = chi_para_lmfp_no_w_on_modes(rdcon_xr)
    rdcon_xr = chi_para_lmfp_noisland_on_modes(rdcon_xr, rdcon_xr.Zeff)
    rdcon_xr = chi_para_smfp_on_modes(rdcon_xr, rdcon_xr.Zeff)
    rdcon_xr = chi_perp_on_modes(rdcon_xr, energy_confinement_time=energy_confinement_time, chi_perp_spline=chi_perp_spline)
    rdcon_xr = deltaprime_crit_on_modes(rdcon_xr)

    if debug_mre_terms:
        return rdcon_xr, None, None

    # Combine xarrays:
    xarrays = []

    #########################################################################################################
    # RDCON delta xarray, delta prime and MRE calculation
    #########################################################################################################
    if not (rdcon_xr is None): 
        # Turn all attributes into variables:
        for attr_key in rdcon_xr.attrs.keys():
            rdcon_xr[attr_key] = rdcon_xr.attrs[attr_key]
        if delete_attrs:
            rdcon_xr.attrs = {}
        # Add new dimension for code to rdcon_xr
        rdcon_xr_expanded = rdcon_xr.expand_dims(dim='code', axis=0)
        rdcon_xr_expanded['code'] = ['rdcon']
        if 'Delta_prime' in rdcon_xr_expanded:
            rdcon_xr_expanded = extract_delta_primes(rdcon_xr_expanded)
            rdcon_xr_expanded = extract_critical_mre_factors_on_modes(rdcon_xr_expanded,rdcon_xr_expanded,k0=k0,k1=k1,C0=C0,iterator=wd_static)
        # Add to xarrays list
        xarrays.append(rdcon_xr_expanded)

    #########################################################################################################
    # STRIDE delta xarray, delta prime and MRE calculation
    #########################################################################################################
    if not (stride_xr is None):
        # Turn all attributes into variables:
        for attr_key in stride_xr.attrs.keys():
            stride_xr[attr_key] = stride_xr.attrs[attr_key]
        if delete_attrs:
            stride_xr.attrs = {}
        # Add new dimension for code to stride_xr
        stride_xr_expanded = stride_xr.expand_dims(dim='code', axis=0)
        stride_xr_expanded['code'] = ['stride']
        if 'Delta_prime' in stride_xr_expanded:
            # Calculate delta' values for stride_xr
            stride_xr_expanded = extract_delta_primes(stride_xr_expanded)
            stride_xr_expanded = extract_critical_mre_factors_on_modes(stride_xr_expanded,rdcon_xr_expanded,k0=k0,k1=k1,C0=C0,iterator=wd_static)
        xarrays.append(stride_xr_expanded)
            
    #########################################################################################################
    # PEST3 delta xarray and delta prime calculation
    #########################################################################################################
    pest3_xr_expanded = None
    if not (pest3_xr is None):
        # Turn all attributes into variables:
        for attr_key in pest3_xr.attrs.keys():
            pest3_xr[attr_key] = pest3_xr.attrs[attr_key]
        if delete_attrs:
            pest3_xr.attrs = {}
        # Add new dimension for code to pest3_xr
        pest3_xr_expanded = pest3_xr.expand_dims(dim='code', axis=0)
        pest3_xr_expanded['code'] = ['pest3']
        if 'Delta_prime' in pest3_xr_expanded:
            assert 'Delta_prime_perr' in pest3_xr_expanded, "Current version of extract_delta_primes assumes this."
            pest3_xr_expanded = extract_delta_primes(pest3_xr_expanded)
            pest3_xr_expanded = extract_critical_mre_factors_on_modes(pest3_xr_expanded,rdcon_xr_expanded,k0=k0,k1=k1,C0=C0,iterator=wd_static)
        xarrays.append(pest3_xr_expanded)

    # Combine all xarrays into one xarray:
    # Breaks if different number of rational surfaces across different codes at the axis
    #   - beware psilow =/= 0 while also running pest3 (pest3 has no psilow truncation)
    #   - for this reason, we also output pest3_xr_out separately if something goes wrong
    pest3_xr_out = None
    combined_xr = None

    #########################################################################################################
    # Concatenating xarrays
    #########################################################################################################
    if len(xarrays) > 0:
        pest3_xr_out = pest3_xr_expanded
        try:
            combined_xr = xr.concat(xarrays, dim='code', coords='all')
            pest3_xr_out = None
        except Exception as e:
            if not (pest3_xr is None): #We remove pest3_xr_expanded from xarrays and retry
                xarrays = xarrays[:-1]  # Remove the last element (pest3_xr_expanded)
                combined_xr = xr.concat(xarrays, dim='code', coords='all')
            print("Error combining xarrays:", e)
            if debug:
                raise e

    #########################################################################################################
    # Adding nn to combined_xr:
    #########################################################################################################
    if combined_xr is not None:
        # Check nn isn't already defined:
        assert not 'nn' in combined_xr, 'nn already defined, debug this function.'
        # We expand dims to add nn:
        combined_xr = combined_xr.expand_dims(dim='nn', axis=0)
        combined_xr['nn'] = [input_dict['nn']]

    return combined_xr, pest3_xr_out, input_dict

def mre_raw_interp(rdcon_xarray):
    """
    Interpolate MRE terms onto rational surfaces using cubic splines. Requires mre_flag & geom_flag='t' (as per default) 
    when running RDCON.
    """
    #########################################################################################################
    # Put mre terms onto surfaces:
    #########################################################################################################
    dvdpsi_n_surf = rdcon_xarray.dvdpsi.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
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
    avg_dpsisq_surf = rdcon_xarray.avg_7.interp(psi_n=rdcon_xarray.psi_n_rational.values,method="cubic").values
    #########################################################################################################
    # Load these surface values into the xarray:
    #########################################################################################################
    rdcon_xarray = rdcon_xarray.assign(dvdpsi_n_surf = dvdpsi_n_surf+0.0*rdcon_xarray['psi_n_rational'],
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
        avg_Bsq_surf = avg_Bsq_surf+0.0*rdcon_xarray['psi_n_rational'],
        avg_dpsisq_surf = avg_dpsisq_surf+0.0*rdcon_xarray['psi_n_rational']
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

    # Get plasma volumes on surfaces by integrating dVdpsi:
    dVdpsi_spline = CubicSpline(rdcon_xarray.psi_n.values, rdcon_xarray.dvdpsi.values, extrapolate=False)
    min_psi_n = rdcon_xarray.psi_n.values.min()
    V_surf = [dVdpsi_spline.integrate(min_psi_n, i) for i in rdcon_xarray.psi_n_rational.values]
    rdcon_xarray = rdcon_xarray.assign(
        V_surf = np.array(V_surf)+0.0*rdcon_xarray['psi_n_rational']
    )
    return rdcon_xarray

def deltaprime_crit_on_modes(rdcon_xarray):
    """
    Calculate the linear, critical (single helicity) Delta' for an instability on each surface,
    using a couple of different formulations.
    The first formulation is Glasser et al. Phys. Fluids 1975, Eq 111.
    The second formulation is from Connor et al. PPCF 2015, Eq 59.
    """
    import math

    #########################################################################################################
    # Flux surface quantities needed:
    #########################################################################################################
    Hs = rdcon_xarray['H_surf'].values
    Drs = rdcon_xarray['Dr_surf'].values
    X0s = rdcon_xarray['X0_surf'].values
    psi_rationals = rdcon_xarray['psi_n_rational'].values
    v_rationals = rdcon_xarray['V_surf'].values
    v1_rationals = rdcon_xarray['dvdpsi_n_surf'].values
    q1_rationals = rdcon_xarray['dq_dpsi_n_surf'].values
    # For Connor et al. 2015:
    avg_dpsisq_surf = rdcon_xarray['avg_dpsisq_surf'].values
    avg_Bsq_surf = rdcon_xarray['avg_Bsq_surf'].values
    chi_perp_surf = rdcon_xarray['chi_perp_surf'].values
    chi_para_smfp_surf = rdcon_xarray['chi_para_smfp_surf'].values
    chi_para_lmfp_noisland_surf = rdcon_xarray['chi_para_lmfp_noisland_surf'].values
    psio = rdcon_xarray['psio']
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
        if (Hs[i] < 0.5 or Hs[i] > -5/2): # Condition for validity for these formulas 
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
        chi_para = min(chi_para_smfp_surf[i], chi_para_lmfp_noisland_surf[i]) # Choose the smaller mean free path (either set by electron-ion collisions, or helical connection length at rational surface)
        chi_frac_noisland = chi_para / chi_perp_surf[i]
        DeltaPrimeCrits15[i] = (1/2)*np.pi**(3/2)*chi_frac_noisland**(1/4)*v_rationals[i]*(-Drs[i])*(alpha*alpha*Lambda*Lambda/(avg_Bsq_surf[i]*avg_dVsq))**(1/4) 
        DeltaPrimeCrits15_no_chifrac[i] = DeltaPrimeCrits15[i]/(chi_frac_noisland**(1/4))

    rdcon_xarray = rdcon_xarray.assign(
        Qcrit_surf = Qcrits+0.0*rdcon_xarray['psi_n_rational'], # Glasser et al. Phys. Fluids 1975, Eq 110.
        DeltaPrime_crit = DeltaPrimeCrits75+0.0*rdcon_xarray['psi_n_rational'], # Glasser et al. Phys. Fluids 1975, Eq 111.
        DeltaPrime_crit_no_X0 = DeltaPrimeCrits75_no_X0+0.0*rdcon_xarray['psi_n_rational'], # Multiply by (1/X0)^(1-2Hs) to get DeltaPrime_crit if you are modifying resistivity and/or mass density.
        DeltaPrime_tcrit = DeltaPrimeCrits15+0.0*rdcon_xarray['psi_n_rational'],  # Connor et al. PPCF 2015, Eq 59. Requires small Dr, small H assumption to be valid (generally true, see Benjamin et al., NF 2025). 
        DeltaPrime_tcrit_no_chifrac = DeltaPrimeCrits15_no_chifrac+0.0*rdcon_xarray['psi_n_rational']  # Multiply by (chi_para/chi_perp)^(1/4) to get DeltaPrime_crit2 if you are modifying transport coefficients.
    )
    
    # All we need for S, X0, and DeltaPrime_crit, in m3dc1 is n, eta(spitz or otherwise), and mass density (ni, ion mass, ne - see mre_terms_on_modes for formula.) 
    # All we need for DeltaPrime_tcrit is chi_frac. Note chi_frac in theory depends on Zeff, but if chifrac is being artificially set by M3DC1, we don't need to worry about it for DeltaPrime_tcrit.
    
    return rdcon_xarray

# If I want: make extra dimension for different versions of generate_wd_function [should probably do this, right now not sure...]
def extract_critical_mre_factors_on_modes(
        code_xarray, # Xarray providing the Delta primes, which will be updated with the MRE analysis
        rdcon_xarray, # Xarray providing surface MRE information, will not be updated unless code_xarray = rdcon_xarray
        k0=0.8227,
        k1=1.7,
        C0=0.6,
        **kwargs):
    """
    Takes in code_xarray with delta prime values already computed, and rdcon_xarray with key mre surface terms computed.
    Returns code_xarray with critical MRE factors including maximum island width, 
    location of maximum island width, and minimum marginally stable island width computed using the delta prime values.
    Parameters:
    k0 = 0.8227 comes from private communication w. Eric Howell, but is near identical to LaHaye 2017 10.1051/epjconf/201715703027 Eq. 1.
    k1 = 1.7 comes from Chang et al. PRL 1995 
    C0 = 0.6 comes from Schlutt and Hegna PoP 2012
    Returns:
    code_xarray : xarray.DataSet
    The updated xarray with critical MRE terms calculated.
    """
    # Check if Delta_prime_varname is in code_xarray:
    assert 'Delta_prime_surf' in code_xarray.data_vars, "Can't run MRE analysis on code_xarray if Delta_prime_surf isn't present..."
    DP_da = code_xarray.Delta_prime_surf

    # Next steps: we check that the r coordinates are the same:
    r1 = code_xarray.Delta_prime_surf.r
    r2 = rdcon_xarray.psi_n_rational.r
    assert np.allclose(r1, r2), "r coordinate does not match between code_xarray and rdcon_xarray"

    # This code should create a vector of w values, then construct the MRE for each w, pulling the minimum for diffusion etc...
    w_vec = np.logspace(-8,0,num=1000)
    w_vec_lowres = np.logspace(-5,0,num=200)

    # Defining output structures:
    tempda = xr.full_like(code_xarray.Delta_prime_surf, np.nan)
    tempda2 = xr.full_like(rdcon_xarray.psi_n_rational, np.nan)
    tempda3 = xr.DataArray(
        np.full(code_xarray.Delta_prime_surf.shape + (len(w_vec_lowres),), np.nan),
        dims=code_xarray.Delta_prime_surf.dims + ('w_bar',),
        coords={**code_xarray.Delta_prime_surf.coords, 'w_bar': w_vec_lowres}
    )
    
    w_margs = tempda.copy(deep=True)
    w_sats = tempda.copy(deep=True)
    w_max_locs = tempda.copy(deep=True)
    dwdtau_maxs = tempda.copy(deep=True)
    wd_at_margs = tempda.copy(deep=True)

    prefacs = tempda2.copy(deep=True)
    wd_at_X0s = tempda2.copy(deep=True)
    dwdt_lowres_da = tempda3.copy(deep=True)

    # Start surface by surface
    for rloc in rdcon_xarray.psi_n_rational.r:
        # Necessity of going surface by surface = defining wd_function:
        rdcon_surf = rdcon_xarray.sel(r=rloc)
        wd_function = generate_wd_function(rdcon_surf,**kwargs)
                
        # Things needed for calculating MRE data
        Dr = rdcon_surf['Dr_surf'].values
        Di = rdcon_surf['Di_surf'].values
        Dnc = rdcon_surf['Dnc_surf'].values
        H = rdcon_surf['H_surf'].values
        
        # Things that we will output using the structure: tempda2 
        prefac = rdcon_surf['eta_star_surf'].values/k0
        wd_at_X0 = wd_function(rdcon_surf['X0_surf'].values)

        # Update prefacs and wd_at_X0s:
        prefacs.loc[dict(r=rloc)] = prefac
        wd_at_X0s.loc[dict(r=rloc)] = wd_at_X0

        # Generate DP_to_MRE function
        DP_to_MRE = mre_combination_wrap(wd_function, Dr, Di, Dnc, H, k1, C0, prefac, w_vec, w_vec_lowres)

        # Apply DP_to_MRE across all delta prime types, record results
        for Dp_type in DP_da.Delta_prime_type:
            DP_val = DP_da.sel(r=rloc,Delta_prime_type=Dp_type).values
            dwdt_low_res, w_marg, w_sat, w_max_loc, dwdtau_max, wd_at_marg = DP_to_MRE(DP_val)

            w_margs.loc[dict(r=rloc, Delta_prime_type=Dp_type)] = w_marg
            w_sats.loc[dict(r=rloc, Delta_prime_type=Dp_type)] = w_sat
            w_max_locs.loc[dict(r=rloc, Delta_prime_type=Dp_type)] = w_max_loc
            dwdtau_maxs.loc[dict(r=rloc, Delta_prime_type=Dp_type)] = dwdtau_max
            wd_at_margs.loc[dict(r=rloc, Delta_prime_type=Dp_type)] = wd_at_marg  

            # Update dwdt_lowres_da (will remain to be seen if this works)
            dwdt_lowres_da.loc[dict(r=rloc, Delta_prime_type=Dp_type)] = dwdt_low_res

    # Now we store these data arrays in code_xarray
    code_xarray = code_xarray.assign(
        w_marg_surf = w_margs,
        w_sat_surf = w_sats,
        w_max_loc_surf = w_max_locs,
        dwdtau_max_surf = dwdtau_maxs,
        prefac_surf = prefacs,
        wd_at_marg_surf = wd_at_margs,
        wd_at_X0_surf = wd_at_X0s,
        dwdt_surf = dwdt_lowres_da
    )

    # Want X0_on_w_marg_surf and X0_on_wd_at_marg_surf to be less than 0 for MRE analysis to be valid!
    code_xarray = code_xarray.assign( 
        X0_on_w_marg_surf = rdcon_xarray['X0_surf']/code_xarray['w_marg_surf'],
        X0_on_wd_at_marg_surf = rdcon_xarray['X0_surf']/code_xarray['wd_at_marg_surf'],
        X0_on_wd_at_X0_surf = rdcon_xarray['X0_surf']/code_xarray['wd_at_X0_surf'])

    return code_xarray

def mre_combination_wrap(wd_function, Dr, Di, Dnc, H, k1, C0, prefac, w_vec, w_vec_lowres):
    """
        Defines for a particular surface (where wd_function and surface quantities are set)
        a function that takes in a delta prime and outputs delta prime dependent MRE values.
    """
    def DP_to_MRE(delta_prime_surf):
        dwdt_vec_low_res = np.full_like(w_vec_lowres, np.nan)
        w_marg = np.nan
        w_sat = np.nan
        w_max_loc = np.nan
        dwdtau_max = np.nan
        wd_at_marg = np.nan
        if not np.isnan(delta_prime_surf):
            dwdtau_loc = lambda w_in: dwdtau(w_in, wd_function, delta_prime_surf, 
                                            Dr, Di, 
                                            Dnc, H, 
                                            k1, C0)
            dwdtau_vec = dwdtau_loc(w_vec)
            dwdt_vec_low_res = prefac*dwdtau_loc(w_vec_lowres)
            w_marg, w_sat, w_max_loc, dwdtau_max = extract_mre_factors(dwdtau_vec, w_vec)
            if not np.isnan(w_marg):
                wd_at_marg = wd_function(w_marg)    
            else:
                wd_at_marg = np.nan
        return dwdt_vec_low_res, w_marg, w_sat, w_max_loc, dwdtau_max, wd_at_marg
    return DP_to_MRE

def extract_critical_mre_factors_on_modes_DEPRECATED(rdcon_xarray,Delta_prime_vec,use_cylindrical_terms=False, k0=0.8227, k1=1.7, C0=0.6,Dprim_name=None):
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
        wd_at_X0_surf = wd_at_X0s+0.0*rdcon_xarray['psi_n_rational']
    )
    rdcon_xarray = rdcon_xarray.assign(
        lowres_dwdt_surf=(
            ('r', 'w_bar'),
            lowres_vecs
        )
    )
    return rdcon_xarray

def extract_mre_factors_old(dwdtau_vec, w_vec): #Update with cubic spline?
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
    # Check if max_index is start or end of vector:
    if max_index==0 or max_index==len(dwdtau_vec)-1:
        # If this is the case, we aren't at a local max. Want a local max
        w_max_loc, dwdtau_max = get_local_max(w_vec,dwdtau_vec)

    return w_marg, w_sat, w_max_loc, dwdtau_max


def extract_mre_factors(dwdtau_vec, w_vec): #Updated with cubic spline
    """
    Extracts the critical MRE factors from the dwdtau_vec and w_vec.
    Returns the marginally stable island width, saturated island width, 
    location of maximum island width, and the maximum dwdtau value.
    """
    dwdtau_spln=CubicSpline(w_vec,dwdtau_vec,extrapolate=False)
    dwdtau_deriv_spln=CubicSpline(w_vec,dwdtau_spln(w_vec,1),extrapolate=False) 

    # Find where dwdtau crosses zero:
    zero_crossings = dwdtau_spln.roots(extrapolate=False)
    
    w_marg = np.nan
    w_sat = np.nan

    if len(zero_crossings) != 0:
        if dwdtau_vec[0] < 0:
            # Marginally stable island width is the first zero crossing:
            w_marg = zero_crossings[0]
        if dwdtau_vec[-1] < 0:
            # Saturated island width is the last zero crossing:
            w_sat = zero_crossings[-1]

    # Maximum island width is where dwdtau is maximum:
    max_index = np.argmax(dwdtau_vec)
    w_max_loc_temp = w_vec[max_index]
    # Check if max_index is start or end of vector:
    if max_index==0 or max_index==len(dwdtau_vec)-1:
        # If this is the case, we aren't at a local max. Want a local max
        w_max_loc_temp,_ = get_local_max(w_vec,dwdtau_vec)
    
    if np.isnan(w_max_loc_temp):
        return w_marg, w_sat, np.nan, np.nan

    extremum_points = dwdtau_deriv_spln.roots(extrapolate=False)
    special_ind = np.argmin(np.abs(extremum_points-w_max_loc_temp))
    w_max_loc = extremum_points[special_ind]
    dwdtau_max = dwdtau_spln(w_max_loc)

    return w_marg, w_sat, w_max_loc, dwdtau_max

def get_local_max(xvec,yvec):
    peak_inds = find_peaks(yvec)[0]
    if len(peak_inds) == 0:
        return np.nan, np.nan
    elif len(peak_inds) > 1:
        print(" Warning, more than one peak in dw/dt, using peak with largest w value.")
        return xvec[peak_inds[-1]], yvec[peak_inds[-1]]
    return xvec[peak_inds[0]], yvec[peak_inds[0]]

# When you artificially set chifrac in M3DC1, make another generate_wd_function that just uses that chifrac. 
# How to implement this within the island is another question
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
        def wd_function(w_bar: float):
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

