"""Collision, seed-island and beam-ion timescales at the rational surfaces, from a TPS store's variables (no code
reruns). Sources: eq_stab claude_findings/fast_ion_times (fast_ion_times.md, tps_timescales_plan.md); tau_p_seed_*_surf:
meta_plan derivations/TPS__seed-pressure-time.md. Times in s; the surface inputs are the store's (SI, T_e in keV,
widths in psi_N). The electron bootstrap response is the store's taue_surf (García-Perciante et al. 2005, Eq. 41).
"""
import numpy as np
import xarray as xr
from scipy.special import gamma

FKR = (np.pi * gamma(0.75) / gamma(0.25)) ** 0.8    # Hegna, Callen & La Haye 1999, Eq. 27
BEAM_Z = 1.0                                         # D beams (FUSE nbi species z_n)

SURFACE_INPUTS = ('taua_surf', 'taur_surf', 'taue_surf', 'taui_surf', 'mu_e_on_nu_e_surf', 'm_rational',
                  'avg_R_surf', 'avg_r_surf', 'v_te_surf', 'v_ti_surf', 'te_keV_surf', 'ne_m3_surf', 'ni_m3_surf',
                  'nimp_m3_surf', 'lnLamb_ei_surf', 'chi_perp_surf', 'avg_Bsq_surf', 'avg_Bsq_on_nabla_psisq_surf',
                  'eta_star_surf', 'w_marg_surf', 'wd_at_marg_surf')
INPUTS = SURFACE_INPUTS + ('main_ion_mass', 'impurity_Z', 'impurity_mass')


def surface_timescales(ds, e_beam_keV=None, beam_mass=2.0):
    """The timescales on ds's surface grid (run_idx, nn, r; w_marg's dims for the width-dependent ones).
    e_beam_keV: the beam injection energy per run (on run_idx, NaN where unknown); None gives NaN beam-ion
    thermalisation and scattering times. beam_mass: beam ion mass in amu."""
    s = {v: ds[v].astype('f8') for v in INPUTS}
    taua, taur, taue, taui = s['taua_surf'], s['taur_surf'], s['taue_surf'], s['taui_surf']
    q = s['m_rational'] / ds['nn']
    eps = s['avg_r_surf'] / s['avg_R_surf']
    # Gianakon, Hegna & Callen 1996, App. A Eqs. 18-20 (v_T = sqrt(2T/m), as the store's v_te_surf, v_ti_surf)
    nus_e = eps ** -1.5 * s['avg_R_surf'] * q / (s['v_te_surf'] * taue)
    nus_i = eps ** -1.5 * s['avg_R_surf'] * q / (s['v_ti_surf'] * taui)
    mu_e = 2.3 * eps ** 0.5 / taue / ((1 + 1.07 * nus_e ** 0.5 + 1.02 * nus_e) * (1 + 1.07 * nus_e * eps ** 1.5))
    mu_i = 0.66 * eps ** 0.5 / taui / ((1 + 1.03 * nus_i ** 0.5 + 0.31 * nus_i) * (1 + 0.66 * nus_i * eps ** 1.5))
    # Seed pressure settling time (TPS__seed-pressure-time.md); G = <B^2>/<B^2/|grad psi_N|^2> [psi_N^2/m^2]. Two
    # conventions for the island scales: Schlutt & Hegna 2012 (k_perp = 1/w; parallel/perpendicular = (w/w_d)^4) and
    # Fitzpatrick 2023 (k_perp = 4/w, Eq. 8.84; ratio (w/W_T)^4 with W_T = sqrt(8) w_d, Eq. 14.203)
    w, wd = s['w_marg_surf'], s['wd_at_marg_surf']
    chi_G = s['chi_perp_surf'] * s['avg_Bsq_surf'] / s['avg_Bsq_on_nabla_psisq_surf']
    # Beam ions: Stix slowing down, critical energy, thermalisation (as IMAS.jl physics/fast.jl); NRL Formulary p. 32
    # 90-degree scattering (fast on ions, slow on electrons). T_e and energies in eV, densities in cm^-3.
    te, ne, lnl = 1e3 * s['te_keV_surf'], 1e-6 * s['ne_m3_surf'], s['lnLamb_ei_surf']
    z2n = s['ni_m3_surf'] + s['impurity_Z'] ** 2 * s['nimp_m3_surf']
    tau_s = 6.27e8 * beam_mass * te ** 1.5 / (BEAM_Z ** 2 * ne * lnl)
    z2_on_a = s['ni_m3_surf'] / s['main_ion_mass'] + s['impurity_Z'] ** 2 * s['nimp_m3_surf'] / s['impurity_mass']
    e_c = 14.8 * beam_mass * te * (z2_on_a / s['ne_m3_surf']) ** (2 / 3)
    e_b = xr.full_like(ds['run_idx'], np.nan, dtype='f8') if e_beam_keV is None else 1e3 * e_beam_keV
    nu_perp = (1.8e-7 * beam_mass ** -0.5 * e_b ** -1.5 * 1e-6 * z2n * lnl
               + 3.2e-9 / beam_mass * te ** -0.5 / e_b * ne * lnl)
    out = dict(
        tau_AS13_surf=taua ** (2 / 3) * taur ** (1 / 3),                       # Hegna et al. Eqs. 23-24
        tau_SP_surf=(taua * taur) ** 0.5,                                       # Hegna et al. Eq. 35
        tau_FKR_surf=taur ** 0.6 * taua ** 0.4 * s['m_rational'] ** -1.2 * FKR,  # Hegna et al. Eq. 27
        nustar_e_surf=nus_e, nustar_i_surf=nus_i,
        inv_mu_e_banana_surf=taue / s['mu_e_on_nu_e_surf'], inv_mu_e_surf=1 / mu_e, inv_mu_i_surf=1 / mu_i,
        tau_eta_wmarg_surf=w ** 2 / s['eta_star_surf'],
        tau_p_seed_SH_surf=1.5 * w ** 2 / chi_G / (1 + (w / wd) ** 4),
        tau_p_seed_Fitz_surf=1.5 * (w / 4) ** 2 / chi_G / (1 + (w / wd) ** 4 / 64),
        tau_s_fast_surf=tau_s, E_c_surf=e_c / 1e3,
        tau_th_fast_surf=tau_s / 3 * np.log1p((e_b / e_c) ** 1.5), tau_perp_fast_surf=1 / nu_perp,
        e_beam_keV=e_b / 1e3)
    return xr.Dataset({k: v.astype('f4') for k, v in out.items()})
