import numpy as np

import tearing_physics_suite.physics.constants as gv
from tearing_physics_suite.drivers.profile_read import read_kin_file
from tearing_physics_suite.physics.surface_terms import mre_terms_on_modes, wc_prefac_m


def test_wc_cylinder_heat_balance_width():
    """Large-aspect-ratio cylinder: RDCON's Wc_prefac (rdcon/mercier.f) through wc_prefac_m gives the
    heat-balance width x_c = (chi_perp/chi_para)^(1/4) (R0 r/(n s))^(1/2) (Schlutt & Hegna W_C)."""
    R0, B0, r, q, s, n, psio, chi = 3.0, 2.0, 0.4, 2.0, 1.5, 1, 0.7, 1e-8
    dPsi_dr = r*B0/q                    # poloidal flux per radian
    v1 = 4*np.pi**2*R0*r/dPsi_dr*psio   # dV/dpsi_n
    q1 = (s*q/r)/dPsi_dr*psio           # dq/dpsi_n
    avg5, avg7 = B0**2, (dPsi_dr/psio)**2   # <B^2>, <|grad psi_n|^2>
    Wc_prefac = (v1*avg5/(q*psio))*v1*avg7*(q*psio)*(q**6/(q1/psio)**2)*(1/(q*psio))**4
    w_r = (chi*wc_prefac_m(Wc_prefac, n*q))**0.25*psio/dPsi_dr
    np.testing.assert_allclose(w_r, chi**0.25*np.sqrt(R0*r/(n*s)), rtol=1e-12)


def test_single_impurity_species(code_fixture, kin_file):
    """species: rho = m_D n_i + m_C n_C + m_e n_e with n_C = (n_e - n_i)/6, v_ti and omega*_i of the main ions (Z = 1)."""
    p = read_kin_file(kin_file)
    kw = dict(ni_spline=p['ni_spline'], ne_spline=p['ne_spline'], ti_spline=p['ti_keV_spline'], te_spline=p['te_keV_spline'])
    ni_scaled = lambda x, nu=0: 0.7*p['ne_spline'](x, nu)              # n_i = 0.7 n_e: carbon makes up the rest
    ds = mre_terms_on_modes(code_fixture('rdcon'), **dict(kw, ni_spline=ni_scaled),
                            species=dict(main_mass=2.0, impurity_Z=6.0, impurity_mass=12.0))
    ne, ni = ds.ne_m3_surf.values, ds.ni_m3_surf.values
    nc = (ne - ni)/6
    np.testing.assert_allclose(ds.nimp_m3_surf.values, nc)
    np.testing.assert_allclose(ds.rho_surf.values, (2*ni + 12*nc)*gv.amu + ne*gv.me)
    np.testing.assert_allclose(ds.Zeff_species_surf.values, (ni + 36*nc)/ne)
    np.testing.assert_allclose(ds.v_ti_surf.values, np.sqrt(2*gv.e*1e3*ds.ti_keV_surf.values/(2*gv.amu)))
    assert float(ds.ne_on_ni_axis) == 1.0
    old = mre_terms_on_modes(code_fixture('rdcon'), **kw, average_ion_mass=2.5)   # one ion species, as before
    np.testing.assert_allclose(old.rho_surf.values, old.ni_m3_surf.values*2.5*gv.amu + old.ne_m3_surf.values*gv.me)
    assert 'nimp_m3_surf' not in old and float(old.ne_on_ni_axis) != 1.0
