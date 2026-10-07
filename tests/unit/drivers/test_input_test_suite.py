"""The ScanSpec registry makes the same scan_1D_input calls as the old per-scan functions.

scan_calls_pre_refactor.json was recorded from the old input_test_suite.py with scan_1D_input mocked.
Known differences (fixes, docs/refactor_changes.md): output_prefix is now forwarded, and the full
matching_point/psilow scans no longer raise NameError.
"""
import json
from pathlib import Path

import pytest

import tearing_physics_suite.drivers.input_test_suite as its
from tearing_physics_suite.utils import create_dense_log_paramvals

REC = json.loads((Path(__file__).resolve().parents[2] / 'data/fixtures/scan_calls_pre_refactor.json').read_text())
OLD_ORDER = ['fourier_mode_scan', 'RDCON_solver_scan', 'sing_order_scan', 'matching_point_scan', 'psilow_truncation_scan',
             'equilibrium_integrator_tolerance_scan', 'mtheta_scan', 'mtheta_scan_no_wall', 'mpsi_scan',
             'mpsi_scan_no_wall', 'edge_truncation_within_surface_scan', 'edge_truncation_within_surface_scan_no_wall',
             'rdcon_finite_element_scan', 'pest3_finite_element_scan', 'integrator_tolerance_scan',
             'mag_coord_type_scan', 'grid_type_scan', 'vacuum_mtheta_scan', 'RDCON_cutoff_scan_n1',
             'RDCON_cutoff_scan_n4', 'STRIDE_axis_mid_pt_skew_scan', 'large_sol_extent_pest_scan', 'RDCON_nq_scan',
             'sing1_flag_scan', 'regrid_flag_scan', 'edge_truncation_q_scan', 'edge_truncation_q_scan_no_wall',
             'wall_radius_scan']


@pytest.fixture
def calls(monkeypatch):
    rec = []
    monkeypatch.setattr(its, 'scan_1D_input', lambda *a, **k: rec.append([list(a), k]) or ({'xarrays': []}, 'm'))
    return rec


def _norm(call):
    return json.loads(json.dumps(call, sort_keys=True, default=str))


def test_registry_order_and_lists():
    assert list(its.scan_functions) == OLD_ORDER
    for lst in (its.key_numerical_tests, its.key_numerical_tests_n4, its.physics_tests, its.extra_numerical_tests):
        assert set(lst) <= set(its.SCANS)
    assert its.key_numerical_tests == its.key_numerical_tests_n4[:-1]


@pytest.mark.parametrize('key', sorted(REC))
def test_same_call_as_old_function(key, calls):
    name, tag = key.split('|')
    kw = REC[key]['kwargs']
    want = REC[key]['call']
    if isinstance(want, str):  # old full matching_point/psilow scans raised NameError (tps.)
        assert want.startswith('NameError')
        its.run_scan(name, 'EQ', results_dir='RES', verbose=False, **kw)
        start, end = {'matching_point_scan': (1e-1, 1e-6), 'psilow_truncation_scan': (1e-1, 1e-5)}[name]
        assert calls[0][0][1] == list(create_dense_log_paramvals(start=start, end=end, points_per_decade=2))
        return
    its.run_scan(name, 'EQ', results_dir='RES', verbose=False, **kw)
    got = _norm(calls[0])
    if tag == 'prefix':  # old functions dropped the caller's output_prefix
        assert got[1].pop('output_prefix') == 'pre_' + want[1].pop('output_prefix', '')
    assert got == _norm(want)


def test_repeated_fixed_kwarg_raises(calls):
    with pytest.raises(TypeError):
        its.run_scan('mtheta_scan', 'EQ', results_dir='RES', verbose=False, vac_flag='t')


def test_banner_printed(calls, capsys):
    its.run_scan('mpsi_scan_no_wall', 'EQ', results_dir='RES')
    assert 'mpsi scan results (no wall):\nm' in capsys.readouterr().out
