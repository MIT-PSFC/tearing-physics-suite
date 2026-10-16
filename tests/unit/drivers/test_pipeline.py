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
