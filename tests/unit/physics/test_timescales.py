"""Timescales at the rational surfaces (physics/timescales.py) and their pass store (drivers/timescales_pass.py)."""
import json

import numpy as np
import xarray as xr

from tearing_physics_suite.drivers import parallel
from tearing_physics_suite.drivers import timescales_pass as tp
from tearing_physics_suite.physics.timescales import INPUTS, surface_timescales


def _store(n_runs=1, **over):
    """A store with every input on (run_idx, nn=[2], r=1): one 3/2 surface."""
    val = dict(taua_surf=1e-6, taur_surf=10.0, taue_surf=2e-5, taui_surf=1e-3, mu_e_on_nu_e_surf=2.0, m_rational=3,
               avg_R_surf=1.7, avg_r_surf=0.4, v_te_surf=2.6e7, v_ti_surf=4e5, te_keV_surf=2.0, ne_m3_surf=5e19,
               ni_m3_surf=4e19, nimp_m3_surf=1e18, lnLamb_ei_surf=17.0, chi_perp_surf=1.0, avg_Bsq_surf=4.0,
               avg_Bsq_on_nabla_psisq_surf=1.0, eta_star_surf=1e-3, w_marg_surf=0.01, wd_at_marg_surf=0.02,
               main_ion_mass=2.0, impurity_Z=6.0, impurity_mass=12.0) | over
    surf = {v: (('run_idx', 'nn', 'r'), np.full((n_runs, 1, 1), val[v])) for v in INPUTS if v.endswith('_surf')
            or v == 'm_rational'}
    runs = {v: ('run_idx', np.full(n_runs, val[v])) for v in INPUTS if v not in surf}
    return xr.Dataset(surf | runs, coords=dict(nn=[2]))


def test_timescales_by_hand():
    """Seeding times from tau_A, tau_R; resistive diffusion across w_marg; beam-ion times NaN without a beam
    energy, finite with one."""
    one = lambda da: float(da.squeeze())  # noqa: E731
    ts = surface_timescales(_store())
    assert np.isclose(one(ts.tau_SP_surf), 1e-6 ** 0.5 * 10 ** 0.5, rtol=1e-6)
    assert np.isclose(one(ts.tau_AS13_surf), 1e-4 * 10 ** (1 / 3), rtol=1e-6)
    assert np.isclose(one(ts.tau_eta_wmarg_surf), 0.1, rtol=1e-6)
    assert np.isnan(one(ts.tau_th_fast_surf)) and np.isfinite(one(ts.tau_s_fast_surf))
    beam = surface_timescales(_store(), xr.DataArray([80.0], dims='run_idx'))
    assert 0 < one(beam.tau_th_fast_surf) < one(beam.tau_s_fast_surf) and np.isfinite(one(beam.tau_perp_fast_surf))


def test_pass_store_joined_by_open_shot(tmp_path):
    """The pass writes <shot>_timescales.zarr with each run's beam energy at its t_ms (NaN off the table); open_shot
    joins it by case_key; a present store is skipped unless forced."""
    from tearing_physics_suite.analysis.shot_data import open_shot
    keys = np.array(['u_0', 'u_1'], dtype=object)
    main = _store(2).assign(case_key=('run_idx', keys), t_ms=('run_idx', [1500, 1700]), shot_id=('run_idx', [100] * 2))
    main.to_zarr(parallel.shot_store(str(tmp_path), 100), consolidated=True)
    table = tmp_path / 'e_beam.jsonl'
    table.write_text(json.dumps(dict(shot=100, t_s=[1.4, 1.6], E_b_keV=[70.0, 90.0])) + '\n{"cut')
    assert tp.write_shot(str(tmp_path), 100, tp.read_e_beam(table)) == 2
    assert tp.write_shot(str(tmp_path), 100) is None
    ds = open_shot(str(tmp_path), 100)
    np.testing.assert_allclose(ds.e_beam_keV.values, [80.0, np.nan])
    assert np.isfinite(ds.tau_th_fast_surf.values[0]).all() and np.isnan(ds.tau_th_fast_surf.values[1]).all()
    assert tp.write_shot(str(tmp_path), 100, force=True) == 2
    assert np.isnan(open_shot(str(tmp_path), 100).e_beam_keV.values).all()
