"""Bouquet-reader helpers: g-file q, the Φ_N -> ψ_N map, and multi_run's equilibrium-named markers."""
import numpy as np
from scipy.integrate import cumulative_trapezoid

from tearing_physics_suite.drivers.multi_run import _validate_case
from tearing_physics_suite.drivers.profile_read import _phi_n_to_psi_n, gfile_q


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
