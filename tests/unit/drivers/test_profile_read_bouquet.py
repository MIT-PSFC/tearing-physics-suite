"""Bouquet-reader helpers: g-file and i-file q, the Φ_N -> ψ_N map, and multi_run's equilibrium-named markers."""
import struct

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from tearing_physics_suite.drivers.multi_run import _profile_kwargs, _validate_case
from tearing_physics_suite.drivers.profile_read import _bouquet_draw_profiles, _phi_n_to_psi_n, gfile_q, ifile_q, read_bouquet_archive


def test_gfile_q(eq_file):
    psi, q = gfile_q(open(eq_file, 'rb').read())
    assert psi[0] == 0.0 and psi[-1] == 1.0 and len(psi) == len(q)
    assert np.all(q > 0.5) and q[-1] > 3 * q[0]


def test_phi_n_to_psi_n(eq_file):
    raw = open(eq_file, 'rb').read()
    psi, q = gfile_q(raw)
    phi = cumulative_trapezoid(q, psi, initial=0.0) / np.trapezoid(q, psi)   # bouquet.coords.phi_n_from_q
    x = np.linspace(0, 1, 41)
    out = _phi_n_to_psi_n({'psi_N': x, 'psi_N_kinetic': x, 'n_e': x}, psi, q)
    np.testing.assert_allclose(out['psi_N'], np.interp(x, phi, psi), atol=1e-12)
    assert out['psi_N'][0] == 0 and out['psi_N'][-1] == 1 and np.all(np.diff(out['psi_N']) > 0)
    assert out['psi_N'][20] > 0.5                        # q rises outward: Φ_N < ψ_N inside
    assert out['n_e'] is x


def test_ifile_q():
    """ψ_N from the psi-bounds record, an i-file grid stopping at the LCFS pad held to ψ_N = 1."""
    pytest.importorskip('bouquet')
    psi = np.linspace(0.3, -0.02, 5)                    # axis to the padded edge, decreasing psi
    q = np.array([-1.1, -1.5, -2.2, -3.4, -6.0])
    recs = [struct.pack('<ii', 5, 4)] + [a.astype('<f8').tobytes() for a in (psi, 0*psi, 0*psi, q,
                                                                              np.zeros(20), np.zeros(20),
                                                                              np.array([0.3, -0.03]))]
    raw = b''.join(struct.pack('<i', len(r)) + r + struct.pack('<i', len(r)) for r in recs)
    x, qa = ifile_q(raw)
    np.testing.assert_allclose(x, (psi - 0.3) / -0.33)
    np.testing.assert_array_equal(qa, -q)
    out = _phi_n_to_psi_n({'psi_N': np.array([0.0, 1.0])}, x, qa)
    assert x[-1] < 1 and out['psi_N'][-1] == 1.0


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
    np.testing.assert_allclose(out['chi_perp_spline'](0.5), 1.5)
    np.testing.assert_allclose(out['chi_perp_i_spline'](0.5), 2.5)
    zeff = _bouquet_draw_profiles(dict(prof, Zeff=1.5 + 0*x), {}, None, '1500', 0, {}, False, None)
    np.testing.assert_allclose(zeff['chi_perp_i_spline'](0.5), 2.5*(1e19/1.5)/0.9e19)   # bouquet's ion heat flux
    old = _bouquet_draw_profiles({k: v for k, v in prof.items() if 'chi' not in k}, {}, None, '1500', 0, {}, False, None)
    assert not {'tau_E_th', 'tau_E', 'chi_perp_spline', 'chi_perp_i_spline'} & set(old)


def test_profile_kwargs_labels():
    prof = dict(tau_E_th=0.2, chi_perp_spline='S', chi_perp_i_spline='I')
    out = _profile_kwargs(prof, dict(tau_e_label='tau_E_th'))
    assert out == dict(energy_confinement_time=0.2, chi_perp_spline='S', chi_perp_i_spline='I')
    assert _profile_kwargs(prof, dict(chi_perp_spline='K'))['chi_perp_spline'] == 'K'


def test_draw_species():
    x = np.linspace(0, 1, 11)
    prof = dict(psi_N=x, psi_N_kinetic=x, n_e=1e19 + 0*x, n_i=0.8e19 + 0*x, T_e=1e3 + 0*x, T_i=1e3 + 0*x)
    out = _bouquet_draw_profiles(prof, dict(Z_imp=6.0), None, '1500', 0, {}, False, None)
    assert out['species'] == dict(main_mass=2.0, impurity_Z=6.0, impurity_mass=12.0) and 'average_ion_mass' not in out
    assert _bouquet_draw_profiles(prof, {}, None, '1500', 0, {}, False, 16.0)['species']['impurity_mass'] == 16.0
    assert _bouquet_draw_profiles(prof, {}, None, '1500', 0, {}, False, None, 2.5)['species']['main_mass'] == 2.5
    beam = _bouquet_draw_profiles(dict(prof, z_fast=1e18 + 0*x, z2_fast=1e18 + 0*x), {}, None, '1500', 0, {}, False, None)
    np.testing.assert_allclose([beam['species'][k](0.5) for k in ('z_fast', 'z2_fast')], 1e18)
