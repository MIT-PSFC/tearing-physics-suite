"""Timescales at the rational surfaces (physics/timescales.py) and their pass store (drivers/timescales_pass.py)."""
import json

import numpy as np
import xarray as xr

from tearing_physics_suite.drivers import parallel
from tearing_physics_suite.drivers import timescales_pass as tp
from tearing_physics_suite.physics.timescales import INPUTS, surface_timescales


def _store(n_runs=1, **over):
    """A store with every input on (run_idx, nn=[2], r=1): one 3/2 surface."""
    val = dict(taua_surf=1e-6, taur_surf=10.0, taue_surf=2e-5, taui_surf=1e-3, ftr_surf=0.6, fc_surf=0.4, Zeff_surf=2.0,
               m_rational=3, avg_R_surf=1.7, eps_surf=0.25, v_te_surf=2.6e7, v_ti_main_surf=4e5, te_keV_surf=2.0,
               ne_m3_surf=5e19, ni_m3_surf=4e19, nimp_m3_surf=1e18, lnLamb_ei_surf=17.0, chi_perp_surf=1.0,
               avg_Bsq_surf=4.0, avg_Bsq_on_nabla_psisq_surf=1.0, eta_star_surf=1e-3, w_marg_surf=0.01,
               wd_at_marg_surf=0.02, main_ion_mass=2.0, impurity_Z=6.0, impurity_mass=12.0) | over
    surf = {v: (('run_idx', 'nn', 'r'), np.full((n_runs, 1, 1), val[v])) for v in INPUTS if v.endswith('_surf')
            or v == 'm_rational'}
    runs = {v: ('run_idx', np.full(n_runs, val[v])) for v in INPUTS if v not in surf}
    return xr.Dataset(surf | runs, coords=dict(nn=[2]))


def test_timescales_by_hand():
    """Seeding times from tau_A, tau_R; Callen's nu_e = Z_eff/tau_e, the banana electron damping time and its
    low-collisionality limit, the ion damping time (B19 written out, at f_t/f_c = 1.46 sqrt(eps), where B12's nu* is
    B7's); the seed pressure time in both conventions,
    (3/2) w^2/(chi G)/(1 + (w/w_d)^4) and (3/32) w^2/(chi G)/(1 + (w/w_d)^4/64), and their perpendicular limits;
    beam-ion times NaN without a beam energy, finite with one."""
    one = lambda da: float(da.squeeze())  # noqa: E731
    ts = surface_timescales(_store())
    assert np.isclose(one(ts.tau_SP_surf), 1e-6 ** 0.5 * 10 ** 0.5, rtol=1e-6)
    assert np.isclose(one(ts.tau_AS13_surf), 1e-4 * 10 ** (1 / 3), rtol=1e-6)
    assert np.isclose(one(ts.inv_nu_e_surf), 1e-5, rtol=1e-6)
    assert np.isclose(one(ts.inv_mu_e_banana_surf), 1e-5 / (1.5 * (1 + 0.533 / 2)), rtol=1e-4)
    rare = surface_timescales(_store(taue_surf=1e3))
    assert np.isclose(one(rare.inv_mu_e_surf), one(rare.inv_mu_e_banana_surf), rtol=1e-3)
    zs, eps, ft = 36 * 1e18 / 4e19, 0.25, 0.73 / 1.73                           # f_t/f_c = 1.46 sqrt(eps) = 0.73
    b19 = surface_timescales(_store(ftr_surf=ft, fc_surf=1 - ft))
    nu_i = (4e19 + 2 ** 0.5 * 36e18) / (2 * 5e19 * 1e-3)
    nus = nu_i * 1.7 * 1.5 / (eps ** 1.5 * 4e5)
    k = (zs + 0.533) / (zs + 0.707)
    mu_i = (0.73 * k * nu_i / (1 + nus ** 0.5 + 1.65 * k * nus)
            / (1 + 1.18 * (2.4 * zs ** 2 + 5.32 * zs + 2.225) / ((zs + 0.707) * (4.25 * zs + 3.02)) * eps ** 1.5 * nus))
    assert np.isclose(one(b19.inv_mu_i_surf), 1 / mu_i, rtol=2e-3)              # B19 rounds Table I's coefficients
    assert np.isclose(one(b19.nustar_i_surf), nus, rtol=1e-6)
    assert np.isclose(one(ts.tau_p_seed_SH_surf), 1.5 * 1e-4 / 4 / (1 + 1 / 16), rtol=1e-6)
    assert np.isclose(one(ts.tau_p_seed_Fitz_surf), 1.5 / 16 * 1e-4 / 4 / (1 + 1 / 16 / 64), rtol=1e-6)
    assert np.isclose(one(ts.tau_eta_wmarg_surf), 0.1, rtol=1e-6)
    narrow = surface_timescales(_store(w_marg_surf=1e-4))
    assert np.isclose(one(narrow.tau_p_seed_SH_surf), 1.5 * 1e-8 / 4, rtol=1e-6)
    assert np.isclose(one(narrow.tau_p_seed_Fitz_surf), 1.5 / 16 * 1e-8 / 4, rtol=1e-6)
    assert np.isnan(one(ts.tau_th_fast_surf)) and np.isfinite(one(ts.tau_s_fast_surf))
    beam = surface_timescales(_store(), xr.DataArray([80.0], dims='run_idx'))
    assert 0 < one(beam.tau_th_fast_surf) < one(beam.tau_s_fast_surf) and np.isfinite(one(beam.tau_perp_fast_surf))


def test_mu_plateau_limit_independent_of_trapped_fraction():
    """Callen's B12 nu* carries mu's f_t/f_c, so in the plateau regime (1 << nu* << eps^-3/2) mu_e tends to
    K^P_00 v_T^2 <(b.grad B)^2>/(omega_t <B^2>) = sqrt(pi) v_T eps^2/(2 R q) whatever f_t/f_c is (banana: in
    proportion to it)."""
    mu = [1 / float(surface_timescales(_store(eps_surf=1e-4, taue_surf=2 / 150, ftr_surf=f, fc_surf=1 - f))
                    .inv_mu_e_surf.squeeze()) for f in (0.5, 2 / 3)]
    np.testing.assert_allclose(mu, np.pi ** 0.5 * 2.6e7 * 1e-8 / (2 * 1.7 * 1.5), rtol=0.02)


def test_mu_uses_rdcon_bdotgradb_where_present():
    """RDCON's <(b.grad B)^2> replaces Callen's eps^2/(2 R^2 q^2) inside mu: at Callen's value, or NaN, mu is
    unchanged; doubled, the plateau mu_e doubles."""
    plateau = dict(eps_surf=1e-4, taue_surf=2 / 150)
    callen = 4.0 * 1e-8 / (2 * (1.7 * 1.5) ** 2)                                # <B^2> eps^2/(2 R^2 q^2)
    mu = lambda bgb: 1 / float(surface_timescales(_store(**plateau).assign(  # noqa: E731
        avg_bdotgradB_sq_surf=(('run_idx', 'nn', 'r'), [[[bgb]]]))).inv_mu_e_surf.squeeze())
    ref = 1 / float(surface_timescales(_store(**plateau)).inv_mu_e_surf.squeeze())
    np.testing.assert_allclose([mu(callen), mu(np.nan)], ref, rtol=1e-12)
    np.testing.assert_allclose(mu(2 * callen), 2 * ref, rtol=0.03)


def test_pass_store_joined_by_open_shot(tmp_path):
    """The pass writes <shot>_timescales.zarr with each run's beam energy at its t_ms (NaN off the table) and eps
    from the study's i-file eps at the rational surfaces; open_shot joins it by case_key; a present store is skipped
    unless forced."""
    from tearing_physics_suite.analysis.shot_data import open_shot
    keys = np.array(['u_0', 'u_1'], dtype=object)
    main = _store(2).drop_vars('eps_surf').assign(
        case_key=('run_idx', keys), t_ms=('run_idx', [1500, 1700]), draw=('run_idx', [0, 1]),
        shot_id=('run_idx', [100] * 2),
        psi_n_rational=(('run_idx', 'nn', 'r'), np.full((2, 1, 1), 0.5)),
        eps=(('run_idx', 'psi_ifile'), np.tile(np.linspace(0, 0.4, 9), (2, 1))),
        eps_psi_n=(('run_idx', 'psi_ifile'), np.tile(np.linspace(0, 1, 9), (2, 1))))
    main.to_zarr(parallel.shot_store(str(tmp_path), 100), consolidated=True)
    table = tmp_path / 'e_beam.jsonl'
    table.write_text(json.dumps(dict(shot=100, t_s=[1.4, 1.6], E_b_keV=[70.0, 90.0])) + '\n{"cut')
    assert tp.write_shot(str(tmp_path), 100, tp.read_e_beam(table)) == 2
    assert tp.write_shot(str(tmp_path), 100) is None
    ds = open_shot(str(tmp_path), 100)
    np.testing.assert_allclose(ds.e_beam_keV.values, [80.0, np.nan])
    np.testing.assert_allclose(ds.nustar_e_surf, surface_timescales(_store(2, eps_surf=0.2)).nustar_e_surf, rtol=1e-5)
    assert np.isfinite(ds.tau_th_fast_surf.values[0]).all() and np.isnan(ds.tau_th_fast_surf.values[1]).all()
    assert tp.write_shot(str(tmp_path), 100, force=True) == 2
    assert np.isnan(open_shot(str(tmp_path), 100).e_beam_keV.values).all()
