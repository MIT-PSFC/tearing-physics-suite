"""Bouquet-reader helpers: g-file q, the Φ_N -> ψ_N map, and multi_run's equilibrium-named markers."""
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from tearing_physics_suite.drivers.multi_run import _profile_kwargs, _validate_case
from tearing_physics_suite.drivers.profile_read import _bouquet_draw_profiles, _phi_n_to_psi_n, gfile_q, read_bouquet_archive


def test_gfile_q(eq_file):
    psi, q = gfile_q(open(eq_file, 'rb').read())
    assert psi[0] == 0.0 and psi[-1] == 1.0 and len(psi) == len(q)
    assert np.all(q > 0.5) and q[-1] > 3 * q[0]


def test_phi_n_to_psi_n(eq_file):
    raw = open(eq_file, 'rb').read()
    psi, q = gfile_q(raw)
    phi = cumulative_trapezoid(q, psi, initial=0.0) / np.trapezoid(q, psi)   # bouquet.coords.phi_n_from_q
    x = np.linspace(0, 1, 41)
    out = _phi_n_to_psi_n({'psi_N': x, 'psi_N_kinetic': x, 'n_e': x}, raw)
    np.testing.assert_allclose(out['psi_N'], np.interp(x, phi, psi), atol=1e-12)
    assert out['psi_N'][0] == 0 and out['psi_N'][-1] == 1 and np.all(np.diff(out['psi_N']) > 0)
    assert out['psi_N'][20] > 0.5                        # q rises outward: Φ_N < ψ_N inside
    assert out['n_e'] is x


def test_marker_names_equilibrium(tmp_path):
    (tmp_path / 'done_0.marker').write_text('/a/eq1')
    (tmp_path / 'combined_xr_0.nc').write_text('x')
    assert _validate_case(0, str(tmp_path), eq_filename='/a/eq1')
    assert not _validate_case(0, str(tmp_path), eq_filename='/a/eq2')
    assert not (tmp_path / 'done_0.marker').exists() and not (tmp_path / 'combined_xr_0.nc').exists()
    (tmp_path / 'done_1.marker').write_text('')           # markers from before this change: accepted
    assert _validate_case(1, str(tmp_path), eq_filename='/a/eq3')


def test_archive_needs_eqdsk_out_dir():
    with pytest.raises(ValueError, match='eqdsk_out_dir'):
        read_bouquet_archive('any.h5')


def test_draw_transport_fields():
    x = np.linspace(0, 1, 11)
    prof = dict(psi_N=x, psi_N_kinetic=x, n_e=1e19 + 0*x, n_i=0.9e19 + 0*x, T_e=1e3 + 0*x, T_i=1e3 + 0*x,
                chi_perp_e=1 + x, chi_perp_i=2 + x)
    out = _bouquet_draw_profiles(prof, dict(tau_E_th=0.2, tau_E=0.25), None, '1500', 0, {}, False, None)
    assert out['tau_E_th'] == 0.2 and out['tau_E'] == 0.25
    np.testing.assert_allclose(out['chi_perp_e_spline'](0.5), 1.5)
    np.testing.assert_allclose(out['chi_perp_i_spline'](0.5), 2.5)
    old = _bouquet_draw_profiles({k: v for k, v in prof.items() if 'chi' not in k}, {}, None, '1500', 0, {}, False, None)
    assert not {'tau_E_th', 'tau_E', 'chi_perp_e_spline', 'chi_perp_i_spline'} & set(old)


def test_profile_kwargs_labels():
    prof = dict(tau_E_th=0.2, chi_perp_e_spline='S')
    out = _profile_kwargs(prof, dict(tau_e_label='tau_E_th', chi_perp_label='chi_perp_e_spline'))
    assert out == dict(energy_confinement_time=0.2, chi_perp_spline='S')
    assert _profile_kwargs(prof, {}) == {}
    assert _profile_kwargs(prof, dict(chi_perp_label='chi_perp_e_spline', chi_perp_spline='K')) == dict(chi_perp_spline='K')


def test_draw_species():
    x = np.linspace(0, 1, 11)
    prof = dict(psi_N=x, psi_N_kinetic=x, n_e=1e19 + 0*x, n_i=0.8e19 + 0*x, T_e=1e3 + 0*x, T_i=1e3 + 0*x)
    out = _bouquet_draw_profiles(prof, dict(Z_imp=6.0), None, '1500', 0, {}, False, None)
    assert out['species'] == dict(main_mass=2.0, impurity_Z=6.0, impurity_mass=12.0) and 'average_ion_mass' not in out
    assert _bouquet_draw_profiles(prof, {}, None, '1500', 0, {}, False, 16.0)['species']['impurity_mass'] == 16.0
