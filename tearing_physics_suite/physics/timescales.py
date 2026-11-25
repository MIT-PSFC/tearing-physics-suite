"""Collision, seed-island and beam-ion timescales at the rational surfaces, from a TPS store's variables (no code
reruns). Sources: eq_stab claude_findings/fast_ion_times (fast_ion_times.md, tps_timescales_plan.md); tau_p_seed_*_surf:
meta_plan derivations/TPS__seed-pressure-time.md; collisions: derivations/TPS__collision-times-callen.md. Times in s;
the surface inputs are the store's (SI, T_e in keV, widths in psi_N), and eps_surf the geometric inverse aspect ratio
(R_max - R_min)/(R_max + R_min) of analysis.shot_data.open_shot. The electron bootstrap response is ~1/<nu_bar>
(inv_nubar_e_surf; García-Perciante et al. 2005, Eqs. 8, 37, 41).
"""
import numpy as np
import xarray as xr
from scipy.special import gamma

C_FKR = gamma(0.25) / (2 * np.pi * gamma(0.75))     # constant-psi layer in GGJ variables: Delta' X0 = Q^(5/4) / C_FKR
BEAM_Z = 1.0                                         # D beams (FUSE nbi species z_n)
AMU_EV = 931.494e6                                   # m_u c^2 [eV]
K_B0 = 2 ** 0.5 - np.log(1 + 2 ** 0.5)               # Callen 2010 Table I: K^B_00 = (Z + K_B0)/(nu tau), K_B0 ~ 0.533

SURFACE_INPUTS = ('taua_surf', 'taur_surf', 'taue_surf', 'taui_surf', 'ftr_surf', 'fc_surf', 'Zeff_surf', 'm_rational',
                  'avg_R_surf', 'eps_surf', 'v_te_surf', 'v_ti_main_surf', 'te_keV_surf', 'ne_m3_surf', 'ni_m3_surf',
                  'nimp_m3_surf', 'lnLamb_ei_surf', 'chi_perp_surf', 'avg_Bsq_surf', 'avg_dpsisq_surf',
                  'eta_star_surf', 'w_marg_surf', 'wd_at_marg_surf', 'Delta_prime_surf', 'Di_surf', 'avg_r_surf',
                  'avg_nabla_psi_surf')
INPUTS = SURFACE_INPUTS + ('main_ion_mass', 'impurity_Z', 'impurity_mass')
OPTIONAL_INPUTS = ('avg_bdotgradB_sq_surf',)      # RDCON's <(b.grad B)^2> (GPEC 355b87c8); Callen's form where absent


def callen_mu(nu, nus, ft_on_fc, z, nu_tau, wt):
    """Poloidal-flow damping rate [1/s] of one species, mu_s/tau_ss of Callen 2010 (UW-CPTC 09-6R) Eqs. B13-B14 with
    Table I's coefficients (B17 and B19 are its sqrt(eps) << 1 forms): nu its collision frequency, nus its general
    collisionality (B12), wt its transit frequency v_T/(R q) (B10), z = Z_eff (electrons) or Z* (hydrogenic ions),
    nu_tau = nu tau_ss (Z_eff or Z* + 1/sqrt(2))."""
    kb = (z + K_B0) / nu_tau                                                    # K^B_00
    kp = np.pi ** 0.5                                                           # K^P_00
    kps = (17 * z / 4 + 205 / (48 * 2 ** 0.5)) / (1.2 * (2 * z ** 2 + 301 * z / (48 * 2 ** 0.5) + 89 / 48))  # K^PS_00
    return nu * ft_on_fc * kb / ((1 + nus ** 0.5 + 2.92 * nus * kb / kp) * (1 + 2 * kp * nu / (3 * wt * nu_tau * kps)))


def surface_timescales(ds, e_beam_keV=None, beam_mass=2.0):
    """The timescales on ds's surface grid (run_idx, nn, r; w_marg's dims for the width-dependent ones).
    e_beam_keV: the beam injection energy per run (on run_idx, NaN where unknown); None gives NaN beam-ion
    thermalisation and scattering times. beam_mass: beam ion mass in amu."""
    s = {v: ds[v].astype('f8') for v in INPUTS}
    taua, taur, taue, taui = s['taua_surf'], s['taur_surf'], s['taue_surf'], s['taui_surf']
    q = s['m_rational'] / ds['nn']
    eps, rq = s['eps_surf'], s['avg_R_surf'] * q
    # Callen 2010: nu_e = Z_eff/tau_ee (A9; taue_surf is tau_ee), nu_i (A14; taui_surf's field density is Z_eff n_e).
    # nustar_*: the regime parameter nu R q/(eps^3/2 v_T), v_T = sqrt(2T/m) (B7). mu: the general nu* (B12) with the
    # same f_t/f_c as mu's numerator, so that it cancels in the plateau and Pfirsch-Schlueter limits, and RDCON's
    # <(b.grad B)^2>/<B^2>, else Callen's eps^2/(2 R^2 q^2) (p. 19)
    zeff, zstar = s['Zeff_surf'], s['impurity_Z'] ** 2 * s['nimp_m3_surf'] / s['ni_m3_surf']
    nu_e = zeff / taue
    nu_i = (s['ni_m3_surf'] + 2 ** 0.5 * s['impurity_Z'] ** 2 * s['nimp_m3_surf']) / (zeff * s['ne_m3_surf'] * taui)
    v_te, v_ti = s['v_te_surf'], s['v_ti_main_surf']
    ft_on_fc = s['ftr_surf'] / s['fc_surf']
    bgradb = eps ** 2 / (2 * rq ** 2)                                          # <(b.grad B)^2>/<B^2>
    if 'avg_bdotgradB_sq_surf' in ds:
        bgradb = (ds['avg_bdotgradB_sq_surf'].astype('f8') / s['avg_Bsq_surf']).fillna(bgradb)
    def nus_b12(nu, v_t):
        return ft_on_fc / 2.92 * nu / (v_t * rq) / bgradb
    # Seed pressure settling time (TPS__seed-pressure-time.md), with <|grad psi_N|^2> [1/m^2], the flux-surface average of
    # the cross-field heat flux q.grad psi_N = -n chi |grad psi_N|^2 dT/dpsi_N (as Schlutt & Hegna's operator). Two
    # conventions for the island scales: Schlutt & Hegna 2012 (k_perp = 1/w; parallel/perpendicular = (w/w_d)^4) and
    # Fitzpatrick 2023 (k_perp = 4/w, Eq. 8.84; ratio (w/W_T)^4 with W_T = sqrt(8) w_d, Eq. 14.203)
    w, wd = s['w_marg_surf'], s['wd_at_marg_surf']
    chi_g2 = s['chi_perp_surf'] * s['avg_dpsisq_surf']
    # FKR time (TPS__fkr-time.md), GGJ variables: X0 = S^-1/3, Q = gamma tau_A / X0, Delta' X0 = Q^(5/4)/C_FKR. GPEC's
    # Delta' (units psi_N^(-2 p1), p1 = sqrt(-D_I)) is taken at the layer width X0 Q^(1/4) as the MRE takes it at w
    # (Rosenberg 2002), so Q = K^(4/(5 - a)), a = 2 p1 - 1. Proxy: Hegna et al. 1999's cylindrical |Delta'| r_s = 2m,
    # with r_s = <r> and d/dr = <|grad psi_N|> d/dpsi_N, at p1 = 1/2
    x0 = (taua / taur) ** (1 / 3)
    p1 = np.sqrt(-s['Di_surf'])
    a = 2 * p1 - 1
    dp = abs(s['Delta_prime_surf'].sel(code='stride', Delta_prime_type='single helicity'))
    q_fkr = (C_FKR * dp * 2 * p1 * x0 * (x0 / 2) ** a) ** (4 / (5 - a))
    q_fkr_proxy = (C_FKR * 2 * s['m_rational'] / (s['avg_r_surf'] * s['avg_nabla_psi_surf']) * x0) ** 0.8
    # Beam ions: Stix slowing down, critical energy, thermalisation (as IMAS.jl physics/fast.jl); NRL Formulary p. 32
    # 90-degree scattering (fast on ions, slow on electrons), with the p. 35 beam-ion Coulomb logarithm
    # (counterstreaming ions, T_i/m_i < v_b^2 < T_e/m_e). T_e and energies in eV, densities in cm^-3.
    te, ne, lnl = 1e3 * s['te_keV_surf'], 1e-6 * s['ne_m3_surf'], s['lnLamb_ei_surf']
    tau_s = 6.27e8 * beam_mass * te ** 1.5 / (BEAM_Z ** 2 * ne * lnl)
    z2_on_a = s['ni_m3_surf'] / s['main_ion_mass'] + s['impurity_Z'] ** 2 * s['nimp_m3_surf'] / s['impurity_mass']
    e_c = 14.8 * beam_mass * te * (z2_on_a / s['ne_m3_surf']) ** (2 / 3)
    e_b = xr.full_like(ds['run_idx'], np.nan, dtype='f8') if e_beam_keV is None else 1e3 * e_beam_keV
    beta2 = 2 * e_b / (beam_mass * AMU_EV)                                     # (v_b/c)^2
    def lam_b(z, a):
        return 43 - np.log(BEAM_Z * z * (beam_mass + a) / (beam_mass * a * beta2) * (ne / te) ** 0.5)
    z2n_lam = 1e-6 * (s['ni_m3_surf'] * lam_b(1, s['main_ion_mass'])
                      + s['impurity_Z'] ** 2 * s['nimp_m3_surf'] * lam_b(s['impurity_Z'], s['impurity_mass']))
    nu_perp = (1.8e-7 * BEAM_Z ** 2 * beam_mass ** -0.5 * e_b ** -1.5 * z2n_lam
               + 3.2e-9 * BEAM_Z ** 2 / beam_mass * te ** -0.5 / e_b * ne * lnl)
    out = dict(
        tau_FKR_surf=taua / (x0 * q_fkr), tau_FKR_proxy_surf=taua / (x0 * q_fkr_proxy),
        nustar_e_surf=eps ** -1.5 * rq * nu_e / v_te, nustar_i_surf=eps ** -1.5 * rq * nu_i / v_ti,
        inv_nubar_e_surf=taue / (zeff + K_B0),                        # García-Perciante's nu_bar averaged as in their Eq. 37
        inv_mu_e_banana_surf=1 / (nu_e * ft_on_fc * (1 + K_B0 / zeff)),
        inv_mu_e_surf=1 / callen_mu(nu_e, nus_b12(nu_e, v_te), ft_on_fc, zeff, zeff, v_te / rq),
        inv_mu_i_surf=1 / callen_mu(nu_i, nus_b12(nu_i, v_ti), ft_on_fc, zstar, zstar + 2 ** -0.5, v_ti / rq),
        tau_eta_wmarg_surf=w ** 2 / s['eta_star_surf'],
        tau_p_seed_SH_surf=1.5 * w ** 2 / chi_g2 / (1 + (w / wd) ** 4),
        tau_p_seed_Fitz_surf=1.5 * (w / 4) ** 2 / chi_g2 / (1 + (w / wd) ** 4 / 64),
        tau_s_fast_surf=tau_s, E_c_surf=e_c / 1e3,
        tau_th_fast_surf=tau_s / 3 * np.log1p((e_b / e_c) ** 1.5), tau_perp_fast_surf=1 / nu_perp,
        e_beam_keV=e_b / 1e3)
    return xr.Dataset({k: v.astype('f4') for k, v in out.items()})
