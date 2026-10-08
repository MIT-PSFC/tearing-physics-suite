"""Study mode (drivers/study.py) without Fortran: feed, run_item with a fake pipeline, compile, the breaker."""
import json
import os

import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.drivers import fsqueue as fq
from tearing_physics_suite.drivers import multi_run, profile_read
from tearing_physics_suite.drivers import study as st


def _record(study, shot, t_ms, variant='base', status='done', selected=(0, 1, 2, 3, 4)):
    d = os.path.join(study, str(shot), 'bouquet')
    os.makedirs(d, exist_ok=True)
    uid = f'{shot}_{t_ms}_{variant}'
    fq.write_json(os.path.join(d, f'{uid}.json'), dict(unit_id=uid, shot=shot, t_ms=t_ms, variant=variant,
                                                      status=status, archive=f'{uid}.h5',
                                                      selected_counts=list(selected),
                                                      all_counts=list(range(6))))
    return uid


@pytest.fixture
def study(tmp_path, monkeypatch):
    s = str(tmp_path / 'study')
    os.makedirs(s)
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps(dict(draws_per_item=2, tps_kwargs=dict(nvec=[1]))))
    monkeypatch.setattr(st, 'versions', lambda: dict(tps='v1', gpec=None, jgpec=None))
    st.main(['init', '--study', s, '--config', str(cfg)])
    monkeypatch.setenv('TPS_WORK_ROOT', str(tmp_path / 'scratch'))
    os.makedirs(tmp_path / 'scratch')
    return s


def test_feed(study):
    uid = _record(study, 100, 1500)
    _record(study, 100, 1600, status='failed')
    _record(study, 101, 1500, variant='other')
    assert st.feed(study) == 3                          # 5 selected draws, 2 per item
    items = fq.Queue(st.qroot(study)).units()
    assert sorted(items) == [f'{uid}.c0', f'{uid}.c1', f'{uid}.c2']
    assert items[f'{uid}.c2']['draws'] == [4]
    assert st.feed(study) == 0
    _record(study, 102, 900, selected=(1,))
    assert st.feed(study) == 1


def _run_dataset(c, n_r=3):
    return xr.Dataset({'Delta_prime': (('nn', 'code', 'r'), np.full((1, 2, n_r), float(c))), 'qlim': ((), 3.0)},
                      coords={'nn': [1], 'code': ['rdcon', 'stride'], 'r': np.arange(n_r)})


def _fake_pipeline(monkeypatch, behaviour):
    """read_bouquet_archive / run_case stand-ins; behaviour[count] is None (ok), an exception, or 'none'."""
    def read(path, draw_counts=None, eqdsk_out_dir=None, **kw):
        os.makedirs(eqdsk_out_dir, exist_ok=True)
        return [f'{eqdsk_out_dir}/eq{c}' for c in draw_counts], [dict(bq_count=c) for c in draw_counts]

    def run_case(eq, prof, kwargs, wd, nc_path, pkl_path=None, label=None, labels=None):
        b = behaviour.get(prof['bq_count'])
        if isinstance(b, list):
            b = b.pop(0) if b else None
        if isinstance(b, BaseException):
            raise b
        if b == 'none':
            return None, {}
        ds = multi_run._augment_combined_xr(_run_dataset(prof['bq_count']), label, prof, labels)
        ds.to_netcdf(nc_path, engine='scipy')
        return ds, dict(nn=1, vac_flag='t')
    monkeypatch.setattr(profile_read, 'read_bouquet_archive', read)
    monkeypatch.setattr(multi_run, 'run_case', run_case)


def _status(study, uid, c):
    return json.load(open(os.path.join(study, '100', 'tps', uid, 'draws', f'{c}.json')))


def test_run_item_statuses_and_retries(study, monkeypatch):
    uid = _record(study, 100, 1500, selected=(0, 1, 2, 3))
    st.feed(study)
    _fake_pipeline(monkeypatch, {1: ValueError('rdcon: no root\ndetail'), 2: [TimeoutError('hung'), None],
                                 3: 'none'})
    out = st.run_item(study, f'{uid}.c0')
    assert [d['status'] for d in out['draws']] == ['ok', 'error']
    assert _status(study, uid, 1)['error']['signature'] == 'ValueError: rdcon: no root'
    with pytest.raises(RuntimeError, match='timeout'):         # a first timeout fails the item, to retry
        st.run_item(study, f'{uid}.c1')
    assert _status(study, uid, 2)['status'] == 'timeout' and _status(study, uid, 3)['status'] == 'error'
    out = st.run_item(study, f'{uid}.c1')                      # only the timed-out draw reruns
    assert [(d['draw'], d['status']) for d in out['draws']] == [(2, 'ok')]
    assert _status(study, uid, 2)['attempts'] == 2
    assert st.run_item(study, f'{uid}.c0')['draws'] == []      # all final


def test_crashed_draw_is_final_after_two_attempts(study, monkeypatch):
    uid = _record(study, 100, 1500, selected=(0,))
    st.feed(study)
    d = os.path.join(study, '100', 'tps', uid, 'draws')
    os.makedirs(d)
    fq.write_json(os.path.join(d, '0.json'), dict(status='running', attempts=2))   # killed twice mid-draw
    _fake_pipeline(monkeypatch, {})
    assert st.run_item(study, f'{uid}.c0')['draws'] == []
    assert _status(study, uid, 0)['status'] == 'crashed'


def test_infra_error_propagates(study, monkeypatch):
    uid = _record(study, 100, 1500, selected=(0,))
    st.feed(study)
    _fake_pipeline(monkeypatch, {0: ImportError('no netCDF4')})
    with pytest.raises(ImportError):
        st.run_item(study, f'{uid}.c0')
    assert not os.path.exists(os.path.join(study, '100', 'tps', uid, 'draws', '0.json'))


def test_compile_incremental(study, monkeypatch):
    uid = _record(study, 100, 1500, selected=(0, 1, 2, 3))
    st.feed(study)
    _fake_pipeline(monkeypatch, {3: ValueError('x')})
    st.run_item(study, f'{uid}.c0')
    state = st.compile_shot(study, 100)
    assert state['compiled'] == [f'{uid}_0', f'{uid}_1']
    st.run_item(study, f'{uid}.c1')
    st.compile_shot(study, 100, finalize=True)
    ds = xr.open_zarr(st.shot_store(study, 100)).load()
    assert list(ds.case_key.values) == [f'{uid}_0', f'{uid}_1', f'{uid}_2']
    np.testing.assert_array_equal(ds.draw.values, [0, 1, 2])
    np.testing.assert_array_equal(ds.Delta_prime.isel(nn=0, code=0, r=0).values, [0.0, 1.0, 2.0])
    assert set(ds.variant.values) == {'base'} and set(ds.t_ms.values) == {1500}
    assert list(ds.code.values) == ['rdcon', 'stride']
    inputs = xr.open_dataset(st.shot_store(study, 100).removesuffix('.zarr') + '_inputs.nc')
    assert list(inputs.case_key.values) == list(ds.case_key.values) and set(inputs.vac_flag.values) == {'t'}
    assert st.compile_shot(study, 100)['compiled'] == [f'{uid}_0', f'{uid}_1', f'{uid}_2']   # nothing new
    assert st.open_study(study).sizes['run_idx'] == 3


def test_runner_breaker(study, monkeypatch):
    q = fq.Queue(st.qroot(study))
    calls = iter([dict(draws=[dict(status='error', signature='E: rdcon missing', unit_id=u, draw=i)
                              for i in range(3)]) for u in ('a', 'b')])
    monkeypatch.setattr(fq, 'run_child', lambda *a, **k: next(calls))
    a = type('A', (), dict(study=study, item_timeout=None, min_left=0, allow_build_drift=False))
    run = st.Runner(a)(q, 'w')
    item = dict(shot=100, unit_id='a')
    assert run('a.c0', item) == dict(n_draws=3, error=3)
    with pytest.raises(fq.Broken, match='rdcon missing'):
        run('b.c0', dict(item, unit_id='b'))


def test_build_drift(study, monkeypatch):
    cfg = st.load_config(study)
    st.check_build(cfg)
    monkeypatch.setattr(st, 'versions', lambda: dict(tps='v2', gpec=None, jgpec=None))
    with pytest.raises(fq.Broken, match='build differs'):
        st.check_build(cfg)
    st.check_build(cfg, allow_drift=True)
