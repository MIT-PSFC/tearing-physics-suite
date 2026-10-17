"""chi_model: ion chi_par, the single-fluid ion weight, chi_perp sources, and the averaged chi_par in W_c."""
import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import PchipInterpolator

import tearing_physics_suite.physics.constants as gv
from tearing_physics_suite.drivers.mre_recompute import _inputs
from tearing_physics_suite.physics.cross_field_transport import (
    chi_ion_weight_on_modes,
    chi_para_ion_on_modes,
    chi_perp_on_modes,
)
from tearing_physics_suite.physics.mre_model import generate_wd_function

PSI = np.array([0.3, 0.6])


def _surfs(**kw):
    v = dict(ne_m3_surf=[5e19, 4e19], ni_m3_surf=[4e19, 3e19], te_keV_surf=[2.0, 1.0], ti_keV_surf=[2.0, 1.5],
             te1_keV_surf=[-2.0, -3.0], ti1_keV_surf=[-1.0, -3.0], Zeff_surf=[1.5, 1.5], v_te_surf=[2.6e7, 1.9e7],
             chi_para_lmfp_no_w_surf=[1e5, 2e5], avg_r_surf=[0.2, 0.4])
    v.update(kw)
    return xr.Dataset({k: ('r', np.asarray(x, float)) for k, x in v.items()}, coords={'r': [0, 1]}).assign(
        psi_n_rational=('r', PSI))


def test_ion_chi_para_by_hand():
    s = chi_para_ion_on_modes(_surfs())
    ti, n_e, n_i, z = 2.0, 5e19, 4e19, 1.5
    lnl = 17.3 - 0.5*np.log(n_i/1e20) + 1.5*np.log(ti)
    m = 2.0*gv.amu
    taui = 6.60e17*np.sqrt(m/gv.mp)*ti**1.5/(z*n_e*lnl)
    np.testing.assert_allclose(s.taui_surf[0], taui)
    np.testing.assert_allclose(s.chi_para_smfp_i_surf[0], 3.9*gv.e*1e3*ti*taui/m)
    vti = np.sqrt(2*gv.e*1e3*ti/m)
    np.testing.assert_allclose(s.chi_para_lmfp_no_w_i_surf[0], 1e5*vti/2.6e7)
    assert 1e8 < s.chi_para_smfp_i_surf[0] < 1e10                   # ~1e9 m^2/s at 2 keV, 5e19 m^-3 (E6)


def test_ion_chi_para_takes_species_main_mass():
    dt = chi_para_ion_on_modes(_surfs().assign(main_ion_mass=2.5))          # species model (mre_terms_on_modes)
    np.testing.assert_allclose(dt.v_ti_main_surf, np.sqrt(2*gv.e*1e3*dt.ti_keV_surf/(2.5*gv.amu)))
    xr.testing.assert_identical(chi_para_ion_on_modes(_surfs(), main_ion_mass_amu=2.5), dt.drop_vars('main_ion_mass'))


def test_ion_weight():
    s = _surfs()
    f = chi_ion_weight_on_modes(s, 'total').chi_ion_weight_surf.values
    np.testing.assert_allclose(f, [4e19*1/(5e19*2 + 4e19*1), 3e19*3/(4e19*3 + 3e19*3)])
    np.testing.assert_array_equal(chi_ion_weight_on_modes(s, 'tau_E').chi_ion_weight_surf, f)
    assert np.all(chi_ion_weight_on_modes(s, 'electron').chi_ion_weight_surf == 0)
    with pytest.raises(ValueError):
        chi_ion_weight_on_modes(s, 'bogus')


def test_chi_perp_models():
    x = np.linspace(0, 1, 11)
    ce, ci = PchipInterpolator(x, 1 + x), PchipInterpolator(x, 3 + x)
    s = chi_ion_weight_on_modes(_surfs(), 'total')
    f = s.chi_ion_weight_surf.values
    tot = chi_perp_on_modes(s, chi_perp_spline=ce, chi_perp_i_spline=ci, chi_model='total').chi_perp_surf.values
    np.testing.assert_allclose(tot, (1 - f)*(1 + PSI) + f*(3 + PSI))
    el = chi_perp_on_modes(s, chi_perp_spline=ce, chi_model='electron').chi_perp_surf.values
    np.testing.assert_allclose(el, 1 + PSI)
    tau = chi_perp_on_modes(s, energy_confinement_time=0.1, chi_perp_spline=ce, chi_model='tau_E').chi_perp_surf.values
    np.testing.assert_allclose(tau, np.array([0.2, 0.4])**2/0.6)        # spline ignored under 'tau_E'
    np.testing.assert_allclose(chi_perp_on_modes(s, energy_confinement_time=0.1, chi_perp_spline=ce).chi_perp_surf.values,
                               tau)                                     # default 'tau_E'
    with pytest.raises(ValueError):
        chi_perp_on_modes(s, chi_perp_spline=ce, chi_model=None)
    with pytest.raises(ValueError):
        chi_perp_on_modes(s, energy_confinement_time=0.1, chi_model='electron')
    with pytest.raises(ValueError):
        chi_perp_on_modes(s, chi_perp_spline=ce, chi_model='total')


def test_total_falls_back_to_electron_where_nonpositive():
    x = np.linspace(0, 1, 11)
    ce, ci = PchipInterpolator(x, 1 + 0*x), PchipInterpolator(x, -100 + 200*x)     # chi_i < 0 at psi 0.3 only
    s = chi_ion_weight_on_modes(_surfs(), 'total')
    f = s.chi_ion_weight_surf.values
    out = chi_perp_on_modes(s, chi_perp_spline=ce, chi_perp_i_spline=ci, chi_model='total')
    np.testing.assert_allclose(out.chi_perp_surf.values, [1.0, (1 - f[1]) + f[1]*20])
    np.testing.assert_array_equal(out.chi_ion_weight_surf.values, [0.0, f[1]])  # electron chi_par there too


def _surf(**kw):
    v = dict(chi_perp_surf=1.0, chi_para_smfp_surf=1e8, chi_para_lmfp_no_w_surf=1e6, Wc_prefac_m_surf=1e-6,
             X0_surf=1e-3, chi_para_smfp_i_surf=1e6, chi_para_lmfp_no_w_i_surf=1e4)
    v.update(kw)
    return xr.Dataset({k: ((), x) for k, x in v.items()})


@pytest.mark.parametrize('iterator', [False, True])
def test_wd_function_ion_weight(iterator):
    w = 1e-3                                       # smfp regime for both species (lmfp/w = 1e9, 1e7)
    base = generate_wd_function(_surf(), iterator=iterator)(w)
    assert generate_wd_function(_surf(chi_ion_weight_surf=0.0), iterator=iterator)(w) == base
    if not iterator:
        np.testing.assert_allclose(base, (1.0/1e8*1e-6)**0.25)
        np.testing.assert_allclose(generate_wd_function(_surf(chi_ion_weight_surf=0.25))(w),
                                   (1.0/(0.75*1e8 + 0.25*1e6)*1e-6)**0.25)
    same = _surf(chi_ion_weight_surf=0.4, chi_para_smfp_i_surf=1e8, chi_para_lmfp_no_w_i_surf=1e6)
    np.testing.assert_allclose(generate_wd_function(same, iterator=iterator)(w), base)   # equal species: no change


def test_recompute_inputs_optional():
    ds = xr.Dataset({'chi_ion_weight_surf': 0.0})
    assert 'chi_ion_weight_surf' in _inputs(ds) and 'chi_para_smfp_i_surf' not in _inputs(ds)
