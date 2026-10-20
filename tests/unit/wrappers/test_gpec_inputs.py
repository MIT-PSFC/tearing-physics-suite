"""GPEC input writers: zeff_dict and byte-identical namelists against stored fixtures.

Fixtures were written by the pre-refactor writer. Regenerate only on purpose:
TPS_REGEN_NAMELISTS=1 pytest tests/unit/wrappers/test_gpec_inputs.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tearing_physics_suite.wrappers.gpec_inputs import ZEFF_MAX_PTS, write_rdcon_stride_inputs, zeff_dict

FIX = Path(__file__).resolve().parents[2] / 'data' / 'fixtures' / 'namelists'
EQ = 'g147131.02300_DIIID_KEFIT'  # written verbatim into equil.in; no file is read
FILES = ('equil.in', 'vac.in', 'rdcon.in', 'stride.in', 'input_dict.json')
CONFIGS = {
    'default': {},
    'reduced_grid': dict(nx=64, mpsi=128, mtheta=129, etol=1e-7),
    'conducting_wall': dict(vac_flag='f', ode_flag='f'),
    'wall_off_conformal': dict(vac_flag='t', a_wall=0, ishape=6),
    'zeff_profile': dict(Zeff={'x': [0.0, 0.5, 1.0], 'y': [1.5, 2.0, 2.5]}),
    'ldp_i': dict(eq_type="'ldp_i'"),
    'gal_xmin': dict(gal_xmin_flag='t'),
    'classic_splines': dict(use_classic_splines='t'),
    'sing_start': dict(sing_start=2, sing_start_str=3),
    'odd_nx_and_overrides': dict(nx=63, set_delta_mlow_to_delta_mhigh=True, set_singfac_min_to_dx=True,
                                 set_int_tolerances_equal=True),
}


def test_zeff_scalar():
    assert zeff_dict(1.5) == {'x': [0.0, 1.0], 'y': [1.5, 1.5]}


def test_zeff_profile_and_limits():
    assert zeff_dict({'x': np.array([0, 1]), 'y': [2, 3]}) == {'x': [0.0, 1.0], 'y': [2.0, 3.0]}
    with pytest.raises(ValueError):
        zeff_dict({'x': [0, 1], 'y': [1]})
    n = ZEFF_MAX_PTS + 1
    with pytest.raises(ValueError):
        zeff_dict({'x': np.linspace(0, 1, n), 'y': np.ones(n)})
    with pytest.raises(ValueError):
        zeff_dict('abc')


def _write(cfg, d):
    """Write the namelists and the returned input dict (as input_dict.json) into d."""
    d.mkdir(parents=True, exist_ok=True)
    out = write_rdcon_stride_inputs(str(d), EQ, verbose=False, **cfg)
    (d / 'input_dict.json').write_text(json.dumps(out, sort_keys=True, indent=1, default=str))


@pytest.mark.parametrize('name', CONFIGS)
def test_namelists_match_fixtures(name, tmp_path, monkeypatch):
    monkeypatch.delenv('TPS_THREADS', raising=False)
    if os.environ.get('TPS_REGEN_NAMELISTS'):
        _write(CONFIGS[name], FIX / name)
    _write(CONFIGS[name], tmp_path)
    for f in FILES:
        ref = FIX / name / f
        assert ref.exists(), f'missing fixture {ref}'
        assert (tmp_path / f).read_text() == ref.read_text(), f'{name}/{f} differs'


def test_key_groups_match_writers():
    import inspect

    from tearing_physics_suite.wrappers import gpec_inputs as gi
    shared, rdcon, stride = set(gi.SHARED_KEYS), set(gi.RDCON_KEYS), set(gi.STRIDE_KEYS)
    assert not (shared & rdcon or shared & stride or rdcon & stride)

    def kw(f):
        return {p.name for p in inspect.signature(f).parameters.values() if p.kind == p.KEYWORD_ONLY}
    assert kw(gi.write_rdcon_in) == shared | rdcon
    warn_only = {'sing1_flag', 'sing_order_ceiling', 'gal_xmin_flag'}  # RDCON inputs STRIDE only warns about
    assert kw(gi.write_stride_in) == shared | stride | warn_only
    params = set(inspect.signature(gi.write_rdcon_stride_inputs).parameters)
    assert shared | rdcon | stride <= params


def test_stride_only_shared_keys(tmp_path):
    """'<key>_stride' gives STRIDE its own value of a shared key; rdcon.in keeps the shared one."""
    out = write_rdcon_stride_inputs(str(tmp_path), EQ, verbose=False, tol_nr=1e-10, tol_nr_stride=1e-12,
                                    singfac_min_stride=2e-5)
    rdcon, stride = ((tmp_path / f).read_text() for f in ('rdcon.in', 'stride.in'))
    assert 'tol_nr=1e-10\n' in rdcon and 'tol_nr=1e-12\n' in stride
    assert 'singfac_min=0.0001\n' in rdcon and 'singfac_min=2e-05\n' in stride
    assert out['tol_nr'] == 1e-10 and out['tol_nr_stride'] == 1e-12


def test_stride_threads_from_tps_threads(tmp_path, monkeypatch):
    """STRIDE's nThreads is $TPS_THREADS (1 when unset) unless given."""
    monkeypatch.setenv('TPS_THREADS', '12')
    out = write_rdcon_stride_inputs(str(tmp_path), EQ, verbose=False)
    assert out['nThreads'] == 12 and 'nThreads=12\n' in (tmp_path / 'stride.in').read_text()
    assert write_rdcon_stride_inputs(str(tmp_path), EQ, verbose=False, nThreads=3)['nThreads'] == 3
    monkeypatch.delenv('TPS_THREADS')
    assert write_rdcon_stride_inputs(str(tmp_path), EQ, verbose=False)['nThreads'] == 1


def test_set_tolerances(tmp_path):
    """set_tolerances sets tol_r, tol_nr (rdcon.in and stride.in) and gal_tol (rdcon.in)."""
    out = write_rdcon_stride_inputs(str(tmp_path), EQ, verbose=False, tol_r=1e-12, tol_nr=1e-12, gal_tol=1e-10,
                                    set_tolerances=1e-11)
    rdcon, stride = ((tmp_path / f).read_text() for f in ('rdcon.in', 'stride.in'))
    assert out['tol_r'] == out['tol_nr'] == out['gal_tol'] == 1e-11
    assert 'tol_r=1e-11\n' in rdcon and 'tol_nr=1e-11\n' in rdcon and 'gal_tol=1e-11\n' in rdcon
    assert 'tol_r=1e-11\n' in stride and 'tol_nr=1e-11\n' in stride
