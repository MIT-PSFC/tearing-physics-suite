import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.physics.mre_model import (
    Delta_GGJ, Delta_nc, DeltaPrime_bar, dwdtau, extract_mre_factors, generate_wd_function, get_local_max)

W = np.logspace(-8, 0, num=1000)
K1, C0 = 1.7, 0.6
SURF = dict(Dr=-0.05, Di=-0.2, Dnc=0.02, H=0.01)


def test_extract_mre_factors_always_decaying():
    assert extract_mre_factors(-1 - W, W)[:2] == (1.0, 0.0)


def test_extract_mre_factors_always_growing():
    w_marg, w_sat, w_max, dw_max = extract_mre_factors(1 + W, W)
    assert (w_marg, w_sat) == (0.0, 1.0)
    assert np.isnan(w_max) and np.isnan(dw_max)  # peak at w=1 is not onset-relevant


def test_extract_mre_factors_roots_and_peak():
    # dw/dtau = -(w-a)(w-b) is negative, crosses 0 at a and b, peaks at (a+b)/2
    a, b = 0.1, 0.5
    w_marg, w_sat, w_max, dw_max = extract_mre_factors(-(W - a) * (W - b), W)
    np.testing.assert_allclose([w_marg, w_sat, w_max], [a, b, (a + b) / 2], rtol=1e-4)
    np.testing.assert_allclose(dw_max, ((b - a) / 2) ** 2, rtol=1e-4)


def test_extract_mre_factors_nonfinite():
    v = -(W - 0.1) * (W - 0.5)
    v[3] = np.nan
    assert all(np.isnan(extract_mre_factors(v, W)))


def test_get_local_max():
    x = np.linspace(0, 1, 101)
    assert get_local_max(x, np.sin(np.pi * x))[:2] == (0.5, 1.0)
    assert np.isnan(get_local_max(x, x)[0])


def test_dwdtau_is_sum_of_terms():
    wd = lambda w: 0.01 + 0 * w  # noqa: E731
    w = np.array([1e-3, 1e-2, 1e-1])
    dp = 3.0
    want = (DeltaPrime_bar(w, dp, SURF['Di']) + Delta_GGJ(w, 0.01, SURF['Dr'], SURF['Di'], SURF['H'], K1, C0)
            + Delta_nc(w, 0.01, SURF['Dnc'], K1, C0))
    np.testing.assert_allclose(dwdtau(w, wd, dp, *SURF.values(), K1, C0), want)


def test_mre_terms_by_hand():
    w, wd, Di = 0.02, 0.01, SURF['Di']
    a_s, a_l = 0.5 + np.sqrt(-Di), 0.5 - np.sqrt(-Di)
    np.testing.assert_allclose(Delta_nc(w, wd, 0.02, K1, C0), K1 * 0.02 * w / (w**2 + wd**2 * K1 / (C0 * 0.81)))
    np.testing.assert_allclose(Delta_GGJ(w, wd, -0.05, Di, 0.01, K1, C0),
                               K1 * (-0.05 / (a_s - 0.01)) / (w + 2 * K1 * wd / (C0 * (1 + a_s))))
    np.testing.assert_allclose(DeltaPrime_bar(w, 3.0, Di), 3.0 * (w / 2) ** (-2 * a_l) * np.sqrt(-4 * Di))


def _surf(**kw):
    v = dict(chi_perp_surf=1.0, chi_para_smfp_surf=1e8, chi_para_lmfp_no_w_surf=1e6, Wc_prefac_m_surf=1e-6,
             X0_surf=1e-3)
    v.update(kw)
    return xr.Dataset({k: ((), x) for k, x in v.items()})


def test_wd_function_min_chi_para():
    s = _surf()
    wd = generate_wd_function(s)
    w = 1e-3  # chi_lmfp = 1e9 > chi_smfp = 1e8, so smfp is used
    np.testing.assert_allclose(wd(w), (1.0 / 1e8 * 1e-6) ** 0.25)
    np.testing.assert_allclose(generate_wd_function(s, force_lmfp=True)(w), (1.0 / 1e9 * 1e-6) ** 0.25)


def test_wd_function_negative_prefac_is_nan():
    assert np.isnan(generate_wd_function(_surf(Wc_prefac_m_surf=-1.0))(1e-3))


@pytest.mark.parametrize('iterator', [False, True])
def test_wd_function_finite(iterator):
    assert np.isfinite(generate_wd_function(_surf(), iterator=iterator)(1e-2))
