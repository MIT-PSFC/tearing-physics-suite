import numpy as np

from tearing_physics_suite.physics.surface_terms import wc_prefac_m


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
