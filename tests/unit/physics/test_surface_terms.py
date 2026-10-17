import numpy as np

from tearing_physics_suite.drivers.profile_read import read_kin_file
from tearing_physics_suite.physics.surface_terms import mre_terms_on_modes, spitzer_zeff_factor, wc_prefac_m


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


def test_spitzer_zeff_factor():
    """Z N(Z)/N(1) against Spitzer & Harm: eta_par(Z)/eta_par(1) = Z alpha(Z)/alpha(1), alpha = 0.5129, 0.4408, 0.3752."""
    z = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(spitzer_zeff_factor(z), z*np.array([0.5129, 0.4408, 0.3752])/0.5129, rtol=0.02)
    assert spitzer_zeff_factor(1.0) == 1.0


def test_spitzer_eta_carries_zeff(code_fixture, kin_file):
    p = read_kin_file(kin_file)

    def run(zeff):
        ds = code_fixture('rdcon')
        ds.attrs['Zeff'] = [zeff, zeff]                                  # this fixture keeps the table as attributes
        return mre_terms_on_modes(ds, ni_spline=p['ni_spline'], ne_spline=p['ne_spline'],
                                  ti_spline=p['ti_keV_spline'], te_spline=p['te_keV_spline'], average_ion_mass=2.5)
    a, b = run(1.0), run(2.0)
    for v in ('eta_spitz_surf', 'eta_spitz'):
        np.testing.assert_allclose(b[v].values, spitzer_zeff_factor(2.0)*a[v].values)
    np.testing.assert_allclose(b.taue_surf.values, a.taue_surf.values)   # Zeff enters chi_para_smfp, not tau_e
