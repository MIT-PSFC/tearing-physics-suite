"""Python replica of how Fortran GPEC builds the F and p knot derivatives of `sq` from a g-file,
for each profile_source (read_eq_efit -> direct_run). Used to compare F', F'' with TokaMaker truth.

GPEC's "extrap" fit is a clamped cubic spline whose end slopes are the derivative of the
4-point Lagrange cubic through the end nodes (spline.f spline_fac/spline_fit_ahg).
"""
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline


def lagrange_end_slope(x, y):
    """Derivative at x[0] of the cubic through (x[0:4], y[0:4])."""
    x0 = x[0]
    s = 0.0
    for j in range(4):
        others = [k for k in range(4) if k != j]
        den = np.prod([x[j] - x[k] for k in others])
        # d/dx prod_{k != j}(x - x_k) at x0
        num = sum(np.prod([x0 - x[k] for k in others if k != m]) for m in others)
        s += y[j] * num / den
    return s


def gpec_extrap_spline(x, y):
    s0 = lagrange_end_slope(x, y)
    s1 = lagrange_end_slope(x[::-1], y[::-1])
    return CubicSpline(x, y, bc_type=((1, s0), (1, s1)))


def ldp_grid(psilow, psihigh, mpsi):
    s = np.arange(mpsi + 1) / mpsi
    return psilow + (psihigh - psilow) * np.sin(s * np.pi / 2) ** 2


def sq_profiles(xs_in, F, p, ffp, pp, method, xs_sq):
    """Splines for |F| and mu0 p as GPEC holds them in `sq` (knots xs_sq).

    xs_in: uniform psi_N of the file; F, p (mu0 p): values; ffp, pp: d(F^2/2)/dpsi_N, d(mu0 p)/dpsi_N.
    method: 'values', 'integrate', or 'hermite' (tabulated values with the file's slopes; tested, not in GPEC).
    Returns (F spline, p spline) callables on psi_N with .derivative().
    """
    if method == 'values':
        f_in, p_in = gpec_extrap_spline(xs_in, F), gpec_extrap_spline(xs_in, p)
        return gpec_extrap_spline(xs_sq, f_in(xs_sq)), gpec_extrap_spline(xs_sq, p_in(xs_sq))
    if method == 'integrate':
        g = gpec_extrap_spline(xs_in, ffp)
        h = gpec_extrap_spline(xs_in, pp)
        gi, hi = g.antiderivative(), h.antiderivative()
        F = np.sqrt(F[-1] ** 2 - 2 * (gi(xs_in[-1]) - gi(xs_in)))
        p = p[-1] - (hi(xs_in[-1]) - hi(xs_in))
    f_in = CubicHermiteSpline(xs_in, F, ffp / F)
    p_in = CubicHermiteSpline(xs_in, p, pp)
    return (CubicHermiteSpline(xs_sq, f_in(xs_sq), f_in(xs_sq, 1)),
            CubicHermiteSpline(xs_sq, p_in(xs_sq), p_in(xs_sq, 1)))


def gfile_profiles(g, mu0=4e-7 * np.pi):
    """psi_N, |F|, mu0 p and their psi_N derivatives from an OFT read_eqdsk dict (as read_eq_efit does)."""
    x = np.linspace(0, 1, g['nr'])
    psio = g['psibry'] - g['psimag']
    F = np.abs(g['fpol'])
    return x, F, np.maximum(g['pres'] * mu0, 0), g['ffprim'] * psio, g['pprime'] * mu0 * psio
