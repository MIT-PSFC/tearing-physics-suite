"""Argument passing in drivers/pipeline.py."""
import pytest

import tearing_physics_suite.drivers.pipeline as pl
from tearing_physics_suite.wrappers.run_codes import CodeResults


class _Stop(Exception):
    pass


def test_analyse_with_mre_passes_te_and_ti_to_their_parameters(monkeypatch):
    seen = {}

    def fake_terms(rdcon_xr, **kwargs):
        seen.update(kwargs)
        raise _Stop
    monkeypatch.setattr(pl, 'run_resistive_calculation',
                        lambda *a, **k: CodeResults('rdcon_xr', None, None, True, False, False, {'nn': 1}, None, {}, {}))
    monkeypatch.setattr(pl, 'mre_terms_on_modes', fake_terms)
    with pytest.raises(_Stop):
        pl.analyse_with_mre('eq', 1, 'ni', 'ne', 'te', 'ti', energy_confinement_time=0.1)
    assert (seen['ni_spline'], seen['ne_spline'], seen['te_spline'], seen['ti_spline']) == ('ni', 'ne', 'te', 'ti')


def test_rdcon_only_variables_have_no_code_dim(monkeypatch, code_fixture, kin_file):
    """F P5: RDCON's surface terms and RDCON-only outputs are stored once, without 'code'; nothing else changes."""
    import numpy as np
    import tearing_physics_suite.physics.combine as cb
    from tearing_physics_suite.drivers.profile_read import read_kin_file
    p = read_kin_file(kin_file)

    def run(stride=True):
        res = CodeResults(code_fixture('rdcon'), code_fixture('stride') if stride else None, None, True, stride, False,
                          {'nn': 1}, None, {}, {})
        monkeypatch.setattr(pl, 'run_resistive_calculation', lambda *a, **k: res)
        return pl.analyse_with_mre('eq', 1, p['ni_spline'], p['ne_spline'], p['te_keV_spline'], p['ti_keV_spline'],
                                   energy_confinement_time=0.1, run_pest3=False)[0]
    new, no_stride = run(), run(stride=False)
    monkeypatch.setattr(cb, 'RDCON_ONLY', ())
    monkeypatch.setattr(cb, 'split_rdcon_only', lambda datasets, rdcon_only=(): (datasets, None))
    old = run()
    moved = [v for v in old.data_vars if 'code' in old[v].dims and 'code' not in new[v].dims]
    assert {'X0_surf', 'Dnc_surf', 'chi_perp_surf', 'avg_B', 'Dnc'} <= set(moved)
    assert 'code' in new.Delta_prime_surf.dims and 'code' in new.w_marg_surf.dims
    for v in old.data_vars:
        a = old[v].sel(code='rdcon', drop=True) if v in moved else old[v]
        np.testing.assert_array_equal(new[v].transpose(*a.dims).values, a.values, err_msg=v)
    assert all('code' not in no_stride[v].dims for v in moved)  # fixed by name: the same when STRIDE fails
