"""Spot-checks (drivers/spot_check.py, multi_run_(spot_check=...)); no codes run (scans mocked or from fixtures)."""
import json
import os
import pickle as pkl

import numpy as np
import pytest
import xarray as xr

import tearing_physics_suite.drivers.multi_run as mr
import tearing_physics_suite.drivers.spot_check as sc
from tearing_physics_suite.drivers.input_scans import extract_scanned_xrs
from tearing_physics_suite.drivers.input_test_suite import SCANS, key_numerical_tests, spot_check_tests
from tearing_physics_suite.wrappers.run_codes import CodeResults

FIX = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fixtures', 'g147131.02300_DIIID_KEFIT_')


@pytest.mark.parametrize('run_pest3, vac_flag, n, edge', [
    (True, 't', 8, 'edge_truncation_q_scan_no_wall'), (True, 'f', 8, 'edge_truncation_q_scan'),
    (False, 't', 7, 'edge_truncation_q_scan_no_wall'), (False, 'f', 7, 'edge_truncation_q_scan')])
def test_spot_check_tests(run_pest3, vac_flag, n, edge):
    scans = spot_check_tests(run_pest3, vac_flag)
    assert len(scans) == n and scans[-1] == edge
    assert ('pest3_finite_element_scan' in scans) == run_pest3
    assert set(scans[:-1]) <= set(key_numerical_tests)


def test_db_scans_reads_run_settings():
    assert sc.db_scans({}) == spot_check_tests(True, 't')  # run_resistive_calculation defaults
    assert sc.db_scans({'run_pest3': False, 'vac_flag': "'f'"}) == spot_check_tests(False, 'f')


def test_plan_reproducible_and_reused(tmp_path):
    spot = sc.SpotCheck(fraction=0.01, seed=3)
    a = sc.plan_spot_checks(250, spot, {'nvec': [1, 2]}, tmp_path / 'a')
    b = sc.plan_spot_checks(250, spot, {'nvec': [1, 2]}, tmp_path / 'b')
    assert a == b and len(a['run_idx']) == 3 and a['nvec'] == [1, 2]  # ceil(2.5)
    other = sc.plan_spot_checks(250, sc.SpotCheck(seed=4), {}, tmp_path / 'a')  # warm start: same plan
    assert other == a
    fresh = sc.plan_spot_checks(250, sc.SpotCheck(seed=4), {}, tmp_path / 'a', warm_start=False)
    assert fresh['seed'] == 4 and json.load(open(tmp_path / 'a' / 'spot_checks' / 'plan.json')) == fresh


def test_resolve_spec_database_settings():
    db = dict(vac_flag='f', a_wall=0.3, ishape=8, run_pest3=False, run_jgpec=True, mtheta=257, ode_flag='t',
              psihigh=0.99, Zeff=1.5, k1=2.0, nvec=[1, 2], working_dir='W', output_location='O', tau_e_label='t',
              nx_pest=100, jgpec_solvers=('galerkin',))
    spec, kw = sc.resolve_spec(SCANS['vacuum_mtheta_scan'], db, nn=2)  # spec fixes NO_WALL, a_wall=0.1
    assert spec.nn == 2 and 'vac_flag' not in spec.fixed and 'a_wall' not in spec.fixed
    assert kw['vac_flag'] == 'f' and kw['a_wall'] == 0.3 and kw['ishape'] == 8
    assert spec.fixed['run_pest3'] is False
    for k in ('Zeff', 'k1', 'nvec', 'working_dir', 'output_location', 'tau_e_label'):
        assert k not in kw
    assert kw['run_jgpec'] and kw['jgpec_solvers'] == ('galerkin',) and kw['nx_pest'] == 100

    spec, kw = sc.resolve_spec(SCANS['pest3_finite_element_scan'], db, nn=1)
    assert spec.fixed['run_pest3'] is False and spec.fixed['run_stride'] is True  # never turns on a code
    assert spec.fixed['run_rdcon'] is False and spec.fixed['ode_flag'] == 'f' and 'ode_flag' not in kw

    spec, kw = sc.resolve_spec(SCANS['edge_truncation_q_scan'], db, nn=1)  # keeps its edge settings
    assert spec.fixed['psihigh'] == 0.9999 and spec.fixed['sas_flag'] == 'f' and 'psihigh' not in kw
    assert kw['vac_flag'] == 'f'


def test_run_spot_task_calls_scan_and_keeps_data(tmp_path, monkeypatch):
    calls = []

    def fake(spec, eq, **k):
        calls.append((spec, eq, k))
        return {'xarrays': ['x'], 'input_name': spec.input_name}, 'msg'
    monkeypatch.setattr(sc, 'run_spec', fake)
    args = ('spot', 4, 'EQ', 'mpsi_scan', 2, {'vac_flag': 'f', 'Zeff': 1.5}, sc.SpotCheck(rel_threshold=0.1), True)
    assert sc.run_spot_task(args, 'W', tmp_path) == (('spot', 4, 'mpsi_scan', 2), True, None)
    spec, eq, k = calls[0]
    assert spec.nn == 2 and k['quick_test'] and k['working_dir'] == 'W' and k['rel_threshold'] == 0.1
    assert k['vac_flag'] == 'f' and 'Zeff' not in k
    assert k['results_dir'] == str(tmp_path / 'spot_checks' / 'run_4' / 'n2')
    with open(tmp_path / 'spot_checks' / 'run_4' / 'mpsi_scan_n2.pkl', 'rb') as f:
        assert pkl.load(f)['xarrays'] == ['x']
    sc.run_spot_task(args, 'W', tmp_path)  # warm start: done marker skips the scan
    assert len(calls) == 1


class _FakePool:
    """Runs tasks in order in-process; spot tasks fail, main tasks succeed."""
    log = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, func, args, chunksize=1):
        _FakePool.log = list(args)
        for a in args:
            yield (('spot',) + a[1:2] + a[3:5], False, 'boom') if a[0] == 'spot' else (a[0], True, None)

    def terminate(self):
        pass

    def join(self):
        pass


class _FakeCtx:
    Pool = _FakePool

    def Queue(self):
        class Q:
            def put(self, x):
                pass
        return Q()


def test_multi_run_spot_tasks_after_main_and_failures_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(mr.multiprocessing, 'get_context', lambda *_: _FakeCtx())
    monkeypatch.setattr(mr, '_get_num_cpus', lambda: 2)
    _, _, errors = mr.multi_run_([f'eq{i}' for i in range(5)], [{}] * 5, str(tmp_path), fail_fast=True,
                                 spot_check=sc.SpotCheck(fraction=0.4), nvec=[1, 2], run_pest3=False)
    tasks = _FakePool.log
    assert [t[0] for t in tasks[:5]] == list(range(5)) and all(t[0] == 'spot' for t in tasks[5:])
    plan = json.load(open(tmp_path / 'spot_checks' / 'plan.json'))
    assert len(plan['run_idx']) == 2
    assert len(tasks) - 5 == 2 * 7 * 2  # cases x scans (no PEST3) x n
    assert errors == {}  # spot failures never count as run failures or trigger fail_fast
    with open(tmp_path / 'spot_checks' / 'errors.pkl', 'rb') as f:
        assert len(pkl.load(f)) == 28


def _scan_result(idx, scan, nn, stride_scale):
    """A real extract_scanned_xrs result from the rdcon/stride fixtures; stride Delta' scaled at the 2nd value."""
    res = []
    for v, scale in ((129, 1.0), (257, stride_scale)):
        r = xr.open_dataset(FIX + 'rdcon_n1.nc').load()
        s = xr.open_dataset(FIX + 'stride_n1.nc').load()
        s['Delta_prime'] = s['Delta_prime'] * scale
        res.append(CodeResults(r, s, None, True, True, False, {'mtheta': v, 'nn': nn}, None, {}, {}))
    out, _ = extract_scanned_xrs(res, 'mtheta', 0.05, 0.05, '')
    return dict(out, run_idx=idx, scan=scan, nn=nn)


@pytest.fixture(scope='module')
def spot_results():
    return [_scan_result(3, 'mtheta_scan', 1, 1.0), _scan_result(3, 'mpsi_scan', 1, 1.2),
            _scan_result(7, 'pest3_finite_element_scan', 1, 1.3)]


def test_compile_spot_checks_and_flag(tmp_path, spot_results):
    for r in spot_results:
        d = tmp_path / 'spot_checks' / f'run_{r["run_idx"]}'
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f'{r["scan"]}_n{r["nn"]}.pkl', 'wb') as f:
            pkl.dump(r, f)
    json.dump({'rel_threshold': 0.05}, open(tmp_path / 'spot_checks' / 'plan.json', 'w'))
    ds, report = sc.compile_spot_checks(str(tmp_path), verbose=False)
    assert ds.sizes['spot'] == 3 and list(ds.code.values) == ['rdcon', 'stride']
    np.testing.assert_array_equal(ds.rel_exceeded_psi95.values, [[0, 1], [0, 0], [0, 1]])  # sorted by (run, scan): mpsi, mtheta, pest3
    assert ds.Delta_prime_surf.dims == ('spot', 'scan_value', 'Delta_prime_type', 'code', 'surf')
    assert list(ds.scan_value_str.isel(spot=0).values) == ['129', '257'] and ds.r_value.values[0, 0] == 2.0
    back = xr.open_zarr(tmp_path / 'spot_checks' / 'spot_checks.zarr').load()
    np.testing.assert_array_equal(back.rel_exceeded_psi95.values, ds.rel_exceeded_psi95.values)
    assert 'run 3  mpsi_scan  n=1  codes [\'stride\']' in report
    assert 'run 7  pest3_finite_element_scan' in report and '[known:' in report
    np.testing.assert_array_equal(sc.spot_check_flag(ds, 3, [1, 2], ['rdcon', 'stride', 'pest3']),
                                  [[0, 1, -1], [-1, -1, -1]])
    np.testing.assert_array_equal(sc.spot_check_flag(ds, 5, [1], ['rdcon']), [[-1]])
    assert sc.spot_check_flag(None, 0, [1], ['rdcon']).tolist() == [[-1]]


def test_stride_rdcon_disagreement(spot_results):
    assert sc.stride_rdcon_disagreements(spot_results, 0.05) == []  # first scan value is unscaled
    r = dict(spot_results[0])
    dp = r['DP_surf_xarray'].copy()
    dp['Delta_prime_surf'] = dp.Delta_prime_surf.where(dp.code != 'stride', 2 * dp.Delta_prime_surf)
    r['DP_surf_xarray'] = dp
    (got,) = sc.stride_rdcon_disagreements([r], 0.05)
    assert got[:2] == (3, 1) and np.isclose(got[3], 2 * got[2], rtol=0.01)


def test_multi_compile_zarr_adds_spot_check_flag(tmp_path, spot_results):
    for i in range(4):
        ds = xr.Dataset({'Delta_prime_surf': (('nn', 'code'), np.full((1, 2), float(i)))},
                        coords={'nn': [1], 'code': ['rdcon', 'stride']})
        with open(tmp_path / f'result_{i}.pkl', 'wb') as f:
            pkl.dump({'combined_xr': ds, 'input_dict_out': {'nx': 64}}, f)
    d = tmp_path / 'spot_checks' / 'run_3'
    d.mkdir(parents=True)
    with open(d / 'mpsi_scan_n1.pkl', 'wb') as f:
        pkl.dump(spot_results[1], f)
    json.dump({'rel_threshold': 0.05}, open(tmp_path / 'spot_checks' / 'plan.json', 'w'))
    out, _ = mr.multi_compile_zarr([f'eq{i}' for i in range(4)], str(tmp_path), report_errs=False)
    flag = out.spot_check_flag.load()
    assert flag.dims == ('run_idx', 'nn', 'code')
    np.testing.assert_array_equal(flag.values[:, 0], [[-1, -1], [-1, -1], [-1, -1], [0, 1]])
    assert (tmp_path / 'spot_checks' / 'spot_checks.zarr').exists()
