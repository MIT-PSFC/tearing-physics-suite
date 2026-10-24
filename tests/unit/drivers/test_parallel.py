"""Study mode (drivers/parallel.py) without Fortran: feed, run_item with a fake pipeline, compile, the breaker."""
import json
import os
import shutil
import zipfile

import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.analysis.shot_data import open_shot
from tearing_physics_suite.drivers import fsqueue as fq
from tearing_physics_suite.drivers import multi_run, profile_read
from tearing_physics_suite.drivers import parallel as st
from tearing_physics_suite.utils import CODE_KILLED, CODE_SKIPPED


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
    cfg.write_text(json.dumps(dict(draws_per_item=2, tps_kwargs=dict(nvec=[1], run_pest3=False))))
    monkeypatch.setattr(st, 'versions', lambda: dict(tps='v1', gpec=None, jgpec=None))
    st.main(['init', '--study', s, '--config', str(cfg)])
    monkeypatch.setenv('TPS_WORK_ROOT', str(tmp_path / 'scratch'))
    os.makedirs(tmp_path / 'scratch')
    return s


def test_feed(study):
    uid = _record(study, 100, 1500)
    _record(study, 100, 1600, status='failed')
    _record(study, 101, 1500, variant='other')
    assert st.feed(study) == 4                          # 5 selected draws, 2 per item, and the baseline
    items = fq.Queue(st.qroot(study)).units()
    assert sorted(items) == [f'{uid}.base', f'{uid}.c0', f'{uid}.c1', f'{uid}.c2']
    assert items[f'{uid}.c2']['draws'] == [4]
    assert items[f'{uid}.base']['draws'] == [-1]
    assert st.feed(study) == 0
    _record(study, 102, 900, selected=(1,))
    assert st.feed(study) == 2


def test_feed_adds_draws_appended_to_a_fed_unit(study):
    """A bouquet fill-up appends draws to a unit's archive after it was fed: only those get items."""
    uid = _record(study, 100, 1500)
    assert st.feed(study) == 4
    _record(study, 100, 1500, selected=(0, 1, 2, 3, 4, 6, 7, 9))
    assert st.feed(study) == 2
    items = fq.Queue(st.qroot(study)).units()
    assert items[f'{uid}.d6']['draws'] == [6, 7] and items[f'{uid}.d9']['draws'] == [9]
    assert st.feed(study) == 0


def test_feed_adds_baseline_to_a_fed_unit(study):
    uid = _record(study, 100, 1500, selected=(0,))
    cfg = st.load_config(study)
    assert st.feed(study, dict(cfg, baseline=False)) == 1
    assert st.feed(study) == 1                          # only <unit>.base is new
    assert sorted(fq.Queue(st.qroot(study)).units()) == [f'{uid}.base', f'{uid}.c0']


def _run_dataset(c, n_r=3):
    dp = (('nn', 'code', 'r'), np.full((1, 2, n_r), float(c)))
    return xr.Dataset({'Delta_prime': dp, 'Delta_prime_surf': dp, 'qlim': ((), 3.0),
                       'dwdt_surf': (('nn', 'code', 'r'), np.zeros((1, 2, n_r)))},
                      coords={'nn': [1], 'code': ['rdcon', 'stride'], 'r': np.arange(n_r)})


def _fake_pipeline(monkeypatch, behaviour):
    """read_bouquet_archive / run_case stand-ins; behaviour[count] is None (ok), an exception, 'none', 'killed' (STRIDE
    killed: NaN, in CODE_KILLED) or 'skipped' (STRIDE not started: also in CODE_SKIPPED)."""
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
        ds = _run_dataset(prof['bq_count'])
        if prof['bq_count'] == 1:                         # a draw where STRIDE did not run
            ds = ds.sel(code=['rdcon'])
        if b in ('killed', 'skipped'):
            for lost in (CODE_KILLED, CODE_SKIPPED) if b == 'skipped' else (CODE_KILLED,):
                lost['stride'] = [1]
            ds['Delta_prime'].loc[dict(code='stride')] = np.nan
        ds = multi_run._augment_combined_xr(ds, label, prof, labels)
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


def test_species_masses_reach_reader(study, monkeypatch):
    cfg = st.load_config(study)
    cfg.update(main_ion_mass_amu=2.5, impurity_mass_amu=16.0)
    fq.write_json(st.config_path(study), cfg)
    uid = _record(study, 100, 1500, selected=(0,))
    st.feed(study)
    _fake_pipeline(monkeypatch, {})
    seen, read = [], profile_read.read_bouquet_archive
    monkeypatch.setattr(profile_read, 'read_bouquet_archive', lambda *a, **k: seen.append(k) or read(*a, **k))
    st.run_item(study, f'{uid}.c0')
    assert (seen[0]['main_ion_mass_amu'], seen[0]['impurity_mass_amu']) == (2.5, 16.0)


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
    assert 'dwdt_surf' not in ds and ds.Delta_prime.dtype == np.float32    # reduced as multi_compile_zarr does (F)
    dp = ds.Delta_prime.isel(nn=0, r=0)                  # draw 1 had no STRIDE: NaN there, RDCON in place
    assert dp.sel(code='rdcon').values[1] == 1.0 and np.isnan(dp.sel(code='stride').values[1])
    inputs = xr.open_dataset(st.shot_store(study, 100).removesuffix('.zarr') + '_inputs.nc')
    assert list(inputs.case_key.values) == list(ds.case_key.values) and set(inputs.vac_flag.values) == {'t'}
    assert st.compile_shot(study, 100)['compiled'] == [f'{uid}_0', f'{uid}_1', f'{uid}_2']   # nothing new
    assert st.open_study(study).sizes['run_idx'] == 3


def test_compile_repairs_an_interrupted_append(study, monkeypatch):
    import zarr
    uid = _record(study, 100, 1500, selected=(0, 1, 2, 3))
    st.feed(study)
    _fake_pipeline(monkeypatch, {3: ValueError('x')})
    st.run_item(study, f'{uid}.c0')
    st.compile_shot(study, 100)
    z = st.shot_store(study, 100)
    arr = zarr.open_group(z, mode='r+')['Delta_prime']        # a killed append: one array a run longer
    arr.resize((3,) + arr.shape[1:])
    st.run_item(study, f'{uid}.c1')
    assert st.compile_shot(study, 100)['compiled'] == [f'{uid}_0', f'{uid}_1', f'{uid}_2']
    ds = xr.open_zarr(z).load()
    np.testing.assert_array_equal(ds.Delta_prime.isel(nn=0, code=0, r=0).values, [0.0, 1.0, 2.0])


def test_compile_finalize_waits_for_a_dead_lease(study, monkeypatch):
    uid = _record(study, 100, 1500, selected=(0,))
    st.feed(study)
    _fake_pipeline(monkeypatch, {})
    st.run_item(study, f'{uid}.c0')
    qcfg = os.path.join(st.qroot(study), 'config.json')
    fq.write_json(qcfg, dict(STALE_S=60, HEARTBEAT_S=0.1))
    assert fq.Queue(st.qroot(study)).acquire('leases', 'compile_100', 'killed job')   # never beats again
    assert st.compile_shot(study, 100) is None
    fq.write_json(qcfg, dict(STALE_S=0.3, HEARTBEAT_S=0.1))
    assert st.compile_shot(study, 100, finalize=True)['compiled'] == [f'{uid}_0']


def test_open_study_reads_published_zip(tmp_path):
    """A shot published as one <shot>.zarr.zip (directory store removed) reads like the directory store."""
    ds = xr.Dataset({'Delta_prime': ('run_idx', [1.0, 2.0]), 'shot_id': ('run_idx', [100, 100]),
                     't_ms': ('run_idx', [1500, 1600])})
    z = st.shot_store(str(tmp_path), 100)
    ds.to_zarr(z, consolidated=True)
    with zipfile.ZipFile(z + '.zip', 'w') as f:
        for root, _, files in os.walk(z):
            for name in files:
                f.write(os.path.join(root, name), os.path.relpath(os.path.join(root, name), z))
    shutil.rmtree(z)
    xr.testing.assert_equal(st.open_study(str(tmp_path)).load(), ds)
    np.testing.assert_array_equal(open_shot(str(tmp_path), 100)['Delta_prime'].values, [1.0, 2.0])
    np.testing.assert_array_equal(open_shot(z + '.zip', 100)['t_ms'].values, [1500, 1600])


def test_runner_breaker(study, monkeypatch):
    q = fq.Queue(st.qroot(study))
    calls = iter([dict(draws=[dict(status='error', signature='E: rdcon missing', unit_id=u, draw=i)
                              for i in range(3)]) for u in ('a', 'b')])
    monkeypatch.setattr(fq, 'run_child', lambda *a, **k: next(calls))
    a = type('A', (), dict(study=study, item_timeout=None, min_left=0, allow_build_drift=False, no_compile=True))
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


def test_tau_e_from_i2(study, monkeypatch, tmp_path):
    import xarray as xr
    tau = np.full((2, 4), np.nan)
    tau[0] = [0.07, 0.08, 0.09, np.nan]
    xr.Dataset({'tau_e': (('shot', 'time_idx'), tau), 'time': ('time_idx', [1.4, 1.5, 1.6, 1.7])},
               coords={'shot': [100, 101]}).to_netcdf(tmp_path / 'i2.nc')
    look = st.tau_e_lookup(dict(path=str(tmp_path / 'i*.nc'), variable='tau_e'))
    assert (look(100, 1.52), look(100, 1.64), look(100, 1.7), look(101, 1.5), look(102, 1.5)) == \
        (0.08, 0.09, None, None, None)   # 1.7: its sample is NaN, the next is 100 ms away
    cfg = st.load_config(study)
    cfg['tau_e'] = dict(path=str(tmp_path / 'i2.nc'), variable='tau_e')
    fq.write_json(st.config_path(study), cfg)
    seen = []
    uid = _record(study, 100, 1500, selected=(0,))
    _record(study, 100, 3000, selected=(0,))
    st.feed(study)
    _fake_pipeline(monkeypatch, {})
    run_case = multi_run.run_case
    monkeypatch.setattr(multi_run, 'run_case', lambda *a, **k: seen.append(a[2]['energy_confinement_time']) or run_case(*a, **k))
    st.run_item(study, f'{uid}.c0')
    assert seen == [0.08]
    assert st.run_item(study, '100_3000_base.c0')['draws'][0]['signature'].startswith('LookupError: no tau_e')


def test_spot_check_draws(study, monkeypatch):
    from tearing_physics_suite.drivers import spot_check as sc
    cfg = st.load_config(study)
    cfg['spot_check'] = dict(fraction=0.5, scans=['mpsi_scan'])
    fq.write_json(st.config_path(study), cfg)
    uid = _record(study, 100, 1500, selected=(0, 1, 2, 3))
    st.feed(study)
    _fake_pipeline(monkeypatch, {})
    tasks = []

    def run_spot_task(task, wd, sdir):
        tasks.append((os.path.basename(task[2]), task[3], task[4]))
        if task[2].endswith('eq3'):
            raise RuntimeError('scan failed')
    monkeypatch.setattr(sc, 'run_spot_task', run_spot_task)
    monkeypatch.setattr(sc, 'spot_dataset', lambda results: xr.Dataset(
        {'rel_exceeded_psi95': (('spot', 'code'), [[0, 1]]), 'run_idx': ('spot', [0]), 'nn': ('spot', [1])},
        coords={'code': ['rdcon', 'stride']}))
    st.run_item(study, f'{uid}.c0')
    st.run_item(study, f'{uid}.c1')
    assert tasks == [('eq0', 'mpsi_scan', 1), ('eq3', 'mpsi_scan', 1)]        # the seeded coin picks draws 0 and 3
    assert [_status(study, uid, c)['status'] for c in range(4)] == ['ok'] * 4   # a failed spot task never fails a draw
    assert _status(study, uid, 0)['spot_errors'] == {} and 'spot_errors' not in _status(study, uid, 1)
    assert _status(study, uid, 3)['spot_errors'] == {'mpsi_scan_n1': 'RuntimeError: scan failed'}
    st.compile_shot(study, 100)
    flag = xr.open_zarr(st.shot_store(study, 100)).spot_check_flag.load()
    np.testing.assert_array_equal(flag.isel(nn=0).values, [[0, 1], [-1, -1], [-1, -1], [0, 1]])
    with pytest.raises(ValueError, match='unknown scans'):
        st.spot_config(dict(spot_check=dict(scans=['no_scan'])))
    with pytest.raises(ValueError, match='unknown scans'):
        st.spot_config(dict(spot_check=dict(scan_vals={'no_scan': [1]})))


def test_draw_timing(study, monkeypatch):
    """A draw records its code wall times and the rest (other_wall_s); the store gets them per draw."""
    from tearing_physics_suite.utils import CODE_WALL_S
    uid = _record(study, 100, 1500, selected=(0, 1))
    st.feed(study)
    _fake_pipeline(monkeypatch, {})
    fake = multi_run.run_case

    def timed(*a, **k):
        CODE_WALL_S.update({'rdcon_n1': 3.0, 'stride_n1': 2.0, 'jGPEC_start': 1.0})
        return fake(*a, **k)
    monkeypatch.setattr(multi_run, 'run_case', timed)
    st.run_item(study, f'{uid}.c0')
    rec = _status(study, uid, 0)
    assert rec['code_wall_s'] == {'rdcon_n1': 3.0, 'stride_n1': 2.0, 'jGPEC_start': 1.0}
    assert rec['other_wall_s'] == pytest.approx(rec['wall_s'] - 6.0, abs=0.11)
    st.compile_shot(study, 100)
    ds = xr.open_zarr(st.shot_store(study, 100)).load()
    np.testing.assert_array_equal(ds.code_wall_s.sel(code=['rdcon', 'stride']).isel(nn=0).values, [[3.0, 2.0]] * 2)
    assert (ds.jgpec_start_wall_s == 1.0).all() and (ds.spot_wall_s == 0).all() and ds.draw_wall_s.notnull().all()
    CODE_WALL_S.clear()


def _set_code_check(study, **check):
    cfg = st.load_config(study)
    fq.write_json(st.config_path(study), dict(cfg, code_check=dict(dict(by='draw', fraction=1.0, codes=['rdcon']),
                                                                   **check)))


def test_code_check_config():
    cfg = dict(tps_kwargs=dict(nvec=[1, 2], jgpec_nvec=[1]))
    assert st.code_check_config(dict(cfg, code_check=dict(codes=['rdcon']))) == dict(
        by='draw', fraction=0.01, seed=0, codes=['rdcon'], nvec=None, flag=None, flagged_n_only=False, mem_classes=None)
    flag = dict(im_re_max=0.01, psi_n=[0.01, 0.9])
    assert st.code_check_config(dict(cfg, code_check=dict(codes=['rdcon'], flag=flag)))['flag'] == flag
    big = dict(flag, dp_abs_max=1e3)
    assert st.code_check_config(dict(cfg, code_check=dict(codes=['rdcon'], flag=big)))['flag'] == big
    full = dict(flag, dp_range=[-300, 150], im_re_m_offset=3)
    assert st.code_check_config(dict(cfg, code_check=dict(codes=['rdcon'], flag=full, flagged_n_only=True)))[
        'flagged_n_only'] is True
    classes = dict(codes=['rdcon'], mem_classes=[9, 16])
    assert st.code_check_config(dict(cfg, code_check=classes))['mem_classes'] == [9, 16]
    for bad in (dict(by='slice', codes=['rdcon']), dict(codes=['stride']), dict(codes=[]),
                dict(codes=['rdcon'], nvec=[3]), dict(codes=['rdcon'], budget='extra'),
                dict(codes=['rdcon'], flag=dict(flag, no_dp=True)), dict(codes=['rdcon'], mem_classes=[16, 9]),
                dict(codes=['rdcon'], mem_classes=[])):
        with pytest.raises(ValueError):
            st.code_check_config(dict(cfg, code_check=bad))
    with pytest.raises(ValueError):                                             # the flag reads STRIDE's Delta'
        st.code_check_config(dict(tps_kwargs=dict(run_stride=False), code_check=dict(codes=['rdcon'], flag=flag)))
    assert st.code_check_kwargs(cfg, dict(codes=['rdcon', 'jGPEC_riccati'])) == dict(
        nvec=[1, 2], jgpec_nvec=[1], run_rdcon=True, rdcon_delta_prime=True, run_stride=False, run_pest3=False,
        run_jgpec=True, jgpec_solvers=['riccati'])
    only_galerkin_n2 = st.code_check_kwargs(cfg, dict(codes=['jGPEC_galerkin'], nvec=[2]))   # RDCON: surfaces only
    assert {k: only_galerkin_n2[k] for k in ('nvec', 'jgpec_nvec', 'rdcon_delta_prime', 'jgpec_solvers')} == dict(
        nvec=[2], jgpec_nvec=[2], rdcon_delta_prime=False, jgpec_solvers=['galerkin'])
    both = dict(codes=['rdcon', 'jGPEC_galerkin', 'jGPEC_riccati'], nvec=None)
    cfg4 = dict(tps_kwargs=dict(nvec=[1, 2, 3, 4], jgpec_nvec=[1, 2]))
    n4 = st.code_check_kwargs(cfg4, both, only_n=[4])                       # flagged at n=4: RDCON there, no Julia
    assert {k: n4[k] for k in ('nvec', 'jgpec_nvec', 'run_jgpec', 'jgpec_solvers')} == dict(
        nvec=[4], jgpec_nvec=[], run_jgpec=False, jgpec_solvers=[])
    n24 = st.code_check_kwargs(cfg4, both, only_n=[2, 4])                   # Julia at its n among the flagged
    assert {k: n24[k] for k in ('nvec', 'jgpec_nvec', 'run_jgpec')} == dict(nvec=[2, 4], jgpec_nvec=[2], run_jgpec=True)
    assert st.code_check_kwargs(cfg4, both)['nvec'] == [1, 2, 3, 4]         # not flagged-only: the study's n


def test_check_n():
    """flagged_n_only: a draw checked for its flags alone needs its flagged n; the random sample and the extras every
    n."""
    assert st._check_n(dict(reason='flagged', flagged_n=[3, 4])) == [3, 4]
    assert st._check_n(dict(reason='killed+flagged', flagged_n=[4])) == [4]                 # an item fed before fills
    for it in (dict(reason='flagged+random', flagged_n=[4]), dict(reason='random'), dict(reason='extra'), dict()):
        assert st._check_n(it) is None


def test_code_check_fills_ok_draws(study, monkeypatch):
    """--code-check: picked 'ok' draws get <count>.codes.nc/.json (their own files untouched) and a separate store."""
    uid = _record(study, 100, 1500, selected=(0, 1, 2, 3))
    st.feed(study)
    _fake_pipeline(monkeypatch, {3: ValueError('x')})
    st.run_item(study, f'{uid}.c0')
    st.run_item(study, f'{uid}.c1')                                        # draws 0-2 ok, 3 error
    _set_code_check(study, by='shot', fraction=0.0)
    assert st.feed_codes(study) == 0
    _set_code_check(study, by='draw', seed=0)
    assert st.feed_codes(study) == 3 and st.feed_codes(study) == 0
    assert sorted(fq.Queue(st.qroot(study, code_check=True)).units()) == [f'{uid}.d0', f'{uid}.d1', f'{uid}.d2']
    ddir = os.path.join(study, '100', 'tps', uid, 'draws')
    before = open(os.path.join(ddir, '0.nc'), 'rb').read()
    out = st.run_item(study, f'{uid}.d0', code_check=True)
    assert [d['status'] for d in out['draws']] == ['ok'] and _status(study, uid, 0)['status'] == 'ok'
    assert json.load(open(os.path.join(ddir, '0.codes.json')))['status'] == 'ok'
    assert open(os.path.join(ddir, '0.nc'), 'rb').read() == before
    assert st.run_item(study, f'{uid}.d0', code_check=True)['draws'] == []   # final
    assert st.compile_shot(study, 100, code_check=True)['compiled'] == [f'{uid}_0']
    codes = xr.open_zarr(st.shot_store(study, 100, code_check=True))
    assert list(codes.case_key.values) == [f'{uid}_0'] and list(codes.code_check_reason.values) == ['random']
    assert st.compile_shot(study, 100)['compiled'] == [f'{uid}_0', f'{uid}_1', f'{uid}_2']   # main store: own files only


def test_code_check_store_takes_checks_with_and_without_julia(study):
    """A check flagged only beyond jgpec_nvec runs no Julia, so its file lacks Julia's cross-code comparisons; the
    store redoes them over its codes, so it takes both kinds of check."""
    from tearing_physics_suite.physics.global_quantities import delta_prime_variability
    uid = '100_1500_base'
    _set_code_check(study, codes=['rdcon', 'jGPEC_galerkin', 'jGPEC_riccati'])
    for c, codes in ((0, ['rdcon', 'stride']), (1, ['rdcon', 'stride', 'jGPEC_galerkin', 'jGPEC_riccati'])):
        base = _ok_draw(study, uid, c).removesuffix('.nc')
        ds = xr.Dataset({'Delta_prime_surf': (('nn', 'code', 'r'), np.ones((1, len(codes), 2)))},
                        coords={'nn': [1], 'code': codes})
        delta_prime_variability(ds).to_netcdf(base + '.codes.nc', engine='scipy')
        fq.write_json(base + '.codes.json', dict(case_key=f'{uid}_{c}', status='ok', reason='flagged'))
    assert st.compile_shot(study, 100, code_check=True)['compiled'] == [f'{uid}_0', f'{uid}_1']


def test_stride_flagged_n():
    """STRIDE's single-helicity Delta' is suspect where |Im| > im_re_max |Re|, or it is missing, in the psi_N window."""
    flag = dict(im_re_max=0.01, psi_n=[0.01, 0.9])
    dims = ('nn', 'Delta_prime_type', 'code', 'r')
    re, im = np.ones((2, 2, 2, 2)), np.zeros((2, 2, 2, 2))
    im[0, 0, 1, 0] = 0.02                                       # n=1, single helicity, STRIDE, at psi_N 0.5
    im[1, 0, 1, 1] = 0.02                                       # n=2 at psi_N 0.95: outside the window
    im[1, 1, 1, 0] = 0.02                                       # n=2, nn coupled: not checked
    ds = xr.Dataset({'Delta_prime_surf': (dims, re), 'Im_Delta_prime_surf': (dims, im),
                     'psi_n_rational': (('nn', 'code', 'r'), np.tile([0.5, 0.95], (2, 2, 1)))},
                    coords={'nn': [1, 2], 'Delta_prime_type': ['single helicity', 'nn coupled'],
                            'code': ['rdcon', 'stride']})
    assert st.stride_flagged_n(ds, flag) == [1]
    ds.Delta_prime_surf.values[1, 0, 1, 0] = np.nan              # no Delta' at n=2, psi_N 0.5
    assert st.stride_flagged_n(ds, flag) == [1, 2]
    assert st.stride_flagged_n(ds.sel(code=['rdcon']), flag) == [1, 2]       # STRIDE dropped: no Delta'


def test_stride_flagged_n_dp_abs_max():
    """With dp_abs_max, |Re Delta'| above it is suspect too (Im/Re small there); psi_n [0, hi] reaches the axis."""
    dims = ('nn', 'Delta_prime_type', 'code', 'r')
    re, im = np.ones((2, 1, 2, 2)), np.zeros((2, 1, 2, 2))
    re[0, 0, 1, 0] = 5e7                                        # n=1 at psi_N 0.005: huge, Im/Re 0
    re[1, 0, 1, 1] = -2e3                                       # n=2 at psi_N 0.5: over 1e3
    ds = xr.Dataset({'Delta_prime_surf': (dims, re), 'Im_Delta_prime_surf': (dims, im),
                     'psi_n_rational': (('nn', 'code', 'r'), np.tile([0.005, 0.5], (2, 2, 1)))},
                    coords={'nn': [1, 2], 'Delta_prime_type': ['single helicity'], 'code': ['rdcon', 'stride']})
    assert st.stride_flagged_n(ds, dict(im_re_max=0.01, psi_n=[0.01, 0.9])) == []           # without it: missed
    assert st.stride_flagged_n(ds, dict(im_re_max=0.01, psi_n=[0.01, 0.9], dp_abs_max=1e3)) == [2]
    assert st.stride_flagged_n(ds, dict(im_re_max=0.01, psi_n=[0.0, 0.9], dp_abs_max=1e3)) == [1, 2]
    assert st.stride_flagged_n(ds, dict(im_re_max=0.01, psi_n=[0.0, 0.9], dp_abs_max=1e8)) == []


def test_stride_flagged_n_dp_range_and_m_offset():
    """dp_range flags Re outside [lo, hi] (asymmetric); im_re_m_offset keeps the Im/Re test to m <= n + offset."""
    dims = ('nn', 'Delta_prime_type', 'code', 'r')
    re, im = -np.ones((2, 1, 2, 3)) * 10, np.zeros((2, 1, 2, 3))
    re[0, 0, 1, 1] = 200                                        # n=1: above hi 150
    im[1, 0, 1, 2] = 0.5                                        # n=2, outer surface (m=7 > 2+3): Im/Re 5 %
    ds = xr.Dataset({'Delta_prime_surf': (dims, re), 'Im_Delta_prime_surf': (dims, im),
                     'psi_n_rational': (('nn', 'code', 'r'), np.tile([0.2, 0.5, 0.7], (2, 2, 1))),
                     'm_rational': (('nn', 'r'), [[2, 3, 4], [3, 5, 7]])},
                    coords={'nn': [1, 2], 'Delta_prime_type': ['single helicity'], 'code': ['rdcon', 'stride']})
    flag = dict(im_re_max=0.01, psi_n=[0.01, 0.9])
    assert st.stride_flagged_n(ds, flag) == [2]
    assert st.stride_flagged_n(ds, dict(flag, dp_range=[-300, 150])) == [1, 2]
    assert st.stride_flagged_n(ds, dict(flag, dp_range=[-300, 150], im_re_m_offset=3)) == [1]
    ds.m_rational.values[1, 2] = 5                                         # now m = n + 3: re-solved
    assert st.stride_flagged_n(ds, dict(flag, dp_range=[-300, 150], im_re_m_offset=3)) == [1, 2]
    ds.Delta_prime_surf.values[0, 0, 1, 1] = -301                            # below lo -300
    assert st.stride_flagged_n(ds, dict(flag, dp_range=[-300, 150], im_re_m_offset=3)) == [1, 2]


def _ok_draw(study, uid, c, **rec):
    d = os.path.join(study, '100', 'tps', uid, 'draws')
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f'{c}.nc'), 'w').close()
    fq.write_json(os.path.join(d, f'{c}.json'), dict(case_key=f'{uid}_{c}', shot=100, t_ms=1500, variant='base',
                                                     unit_id=uid, draw=c, archive='a.h5', status='ok', **rec))
    return os.path.join(d, f'{c}.nc')


def test_code_check_feed_order(study):
    """Flagged draws first; the coin's picks once no flagged one waits for a worker; then, from the work feeder,
    its worker count of further draws in the coin's order, whenever nothing more urgent waits. Draws whose check is
    to come are pending (their files stay); cleaned draws are not fed."""
    uid = '100_1500_base'
    nc = [_ok_draw(study, uid, c, stride_flagged_n=[2] if c == 1 else []) for c in range(7)]
    os.remove(nc[6])                                            # cleaned before it was picked
    _set_code_check(study, fraction=0.0)
    q = fq.Queue(st.qroot(study, code_check=True))

    def reasons():
        return {k: it['reason'] for k, it in q.units(refresh=True).items()}
    assert st.code_check_pending(study, 100) == [nc[1]]       # flagged, not queued yet
    assert st.feed_codes(study, extra=2) == 1 and reasons() == {f'{uid}.d1': 'flagged'}
    assert q.units()[f'{uid}.d1']['flagged_n'] == [2]           # the check can run at the flagged n only
    assert st.feed_codes(study, extra=2) == 0                   # the flagged draw waits
    q.finish(f'{uid}.d1')
    assert st.code_check_pending(study, 100) == []
    assert st.feed_codes(study) == 0 and st.feed_codes(study, extra=2) == 2
    coin = sorted((st._coin(0, f'{uid}_{c}'), c) for c in (0, 2, 3, 4, 5))
    assert {k for k, r in reasons().items() if r == 'extra'} == {f'{uid}.d{c}' for _, c in coin[:2]}
    assert st.code_check_pending(study, 100) == sorted(nc[c] for _, c in coin[:2])
    _set_code_check(study, fraction=1.0)                        # the rest picked: ahead of the waiting extras
    assert st.feed_codes(study, extra=2) == 3 and list(reasons().values()).count('random') == 3
    _ok_draw(study, uid, 7, stride_flagged_n=[1])
    assert st.feed_codes(study, extra=2) == 1 and reasons()[f'{uid}.d7'] == 'flagged+random'
    _ok_draw(study, uid, 8, stride_flagged_n=[1])
    nc9 = _ok_draw(study, uid, 9, code_killed={'stride': [4]}, stride_flagged_n=[4])   # a lost STRIDE n: fill first
    assert st.feed_codes(study) == 1 and reasons()[f'{uid}.f9'] == 'killed' and q.units()[f'{uid}.f9']['fill_n'] == [4]
    assert st.feed_codes(study) == 0 and nc9 in st.code_check_pending(study, 100)


def test_stride_flag_recorded_and_compiled(study, monkeypatch):
    """With code_check.flag, each draw records its STRIDE-flagged n, and the main store gets stride_flag(nn). A flag
    that cannot be evaluated flags every n and does not fail the draw."""
    uid = _record(study, 100, 1500, selected=(0, 2, 3))
    st.feed(study)
    _fake_pipeline(monkeypatch, {})
    _set_code_check(study, flag=dict(im_re_max=0.01, psi_n=[0.01, 0.9]))

    def flagged(ds, flag):
        if int(ds.draw) == 3:
            raise KeyError('Im_Delta_prime_surf')
        return [1] if int(ds.draw) == 0 else []
    monkeypatch.setattr(st, 'stride_flagged_n', flagged)
    st.run_item(study, f'{uid}.c0')
    st.run_item(study, f'{uid}.c1')
    assert [_status(study, uid, c)['stride_flagged_n'] for c in (0, 2, 3)] == [[1], [], [1]]
    assert _status(study, uid, 3)['status'] == 'ok' and 'KeyError' in _status(study, uid, 3)['stride_flag_error']
    st.compile_shot(study, 100)
    np.testing.assert_array_equal(xr.open_zarr(st.shot_store(study, 100)).stride_flag.values, [[1], [0], [1]])


def test_lost_stride_n_filled_first(study, monkeypatch):
    """A draw that lost a STRIDE n (killed, or skipped as too big) gets a fill before its check: STRIDE at that n alone,
    into the draw itself (code_filled, flag redone); the main compile holds the draw until then. Lost again, the fill
    fails and the draw goes in as it is, then gets its check."""
    uid = _record(study, 100, 1500, selected=(0, 2, 3))
    st.feed(study)
    _fake_pipeline(monkeypatch, {0: ['skipped', None], 3: ['killed', 'killed']})
    monkeypatch.setattr(st, 'stride_flagged_n', lambda ds, flag: [
        int(n) for n in ds.nn.values if np.isnan(ds.Delta_prime.sel(code='stride', nn=n)).all()])
    _set_code_check(study, fraction=0.0, flag=dict(im_re_max=0.01, psi_n=[0.01, 0.9]))
    st.run_item(study, f'{uid}.c0')
    st.run_item(study, f'{uid}.c1')
    assert [_status(study, uid, c).get('code_killed') for c in (0, 2, 3)] == [{'stride': [1]}, None, {'stride': [1]}]
    assert _status(study, uid, 0)['code_skipped'] == {'stride': [1]} and 'code_skipped' not in _status(study, uid, 3)
    assert st.compile_shot(study, 100)['compiled'] == [f'{uid}_2']                 # 0 and 3 wait for their fills
    q = fq.Queue(st.qroot(study, code_check=True))
    assert st.feed_codes(study) == 2 and sorted(q.units(refresh=True)) == [f'{uid}.f0', f'{uid}.f3']   # no checks yet
    st.run_item(study, f'{uid}.f0', code_check=True)
    st.run_item(study, f'{uid}.f3', code_check=True)
    assert {k: _status(study, uid, 0)[k] for k in ('code_filled', 'stride_flagged_n')} == dict(
        code_filled=['stride'], stride_flagged_n=[])
    fill3 = json.load(open(os.path.join(study, '100', 'tps', uid, 'draws', '3.fill.json')))
    assert fill3['error']['signature'] == "RuntimeError: lost code(s) not re-run: {'stride': [1]}"
    assert 'code_filled' not in _status(study, uid, 3)
    assert st.compile_shot(study, 100)['compiled'] == [f'{uid}_2', f'{uid}_0', f'{uid}_3']   # 3: its fill is final
    dp = xr.open_zarr(st.shot_store(study, 100)).Delta_prime.sel(code='stride').isel(nn=0, r=0).values
    assert dp[:2].tolist() == [2.0, 0.0] and np.isnan(dp[2])
    counts = st.draw_status(study)[0][100]
    assert {k: counts[k] for k in ('code_killed', 'code_skipped', 'code_filled')} == dict(
        code_killed=2, code_skipped=1, code_filled=1)
    assert {k: v for k, v in st.draw_status(study, code_check=True)[0][100].items() if k.startswith('fill')} == {
        'fill ok': 1, 'fill error': 1}
    for f in ('f0', 'f3'):
        q.finish(f'{uid}.{f}')
    assert st.feed_codes(study) == 1 and q.units(refresh=True)[f'{uid}.d3']['reason'] == 'flagged'  # then its check


def test_fill_keeps_the_draws_other_n(tmp_path, monkeypatch):
    """A fill re-runs STRIDE at its lost n only and fills that n: the draw keeps STRIDE's other n."""
    from tearing_physics_suite.drivers import pipeline
    monkeypatch.setattr(pipeline, 'delta_prime_variability', lambda ds: ds)   # the fakes have no Delta_prime_surf

    def dataset(nn, stride):
        return xr.Dataset({'Delta_prime': (('nn', 'code', 'r'), [[[1.0], [s]] for s in stride]), 'qlim': ((), 3.0)},
                          coords={'nn': nn, 'code': ['rdcon', 'stride'], 'r': [0]})
    dataset([1, 2], [2.0, np.nan]).to_netcdf(tmp_path / '0.nc', engine='scipy')     # STRIDE n=2 lost
    fq.write_json(str(tmp_path / '0.json'), dict(status='ok', code_killed={'stride': [2]}))
    monkeypatch.setattr(multi_run, 'run_case', lambda *a: (dataset([2], [4.0]), {}))   # the fill, at n=2 only
    CODE_KILLED.clear()
    st._fill_case(None, None, {}, str(tmp_path), str(tmp_path / '0.fill.nc'))
    stride = xr.open_dataset(tmp_path / '0.nc', engine='scipy').Delta_prime.sel(code='stride', r=0)
    assert stride.values.tolist() == [2.0, 4.0] and not (tmp_path / '0.fill.nc').exists()


def _sized_draw(study, uid, c, mpert, msing, **rec):
    """An 'ok' draw at n = 4 whose RDCON surface pass found mpert and msing surfaces."""
    d = os.path.join(study, '100', 'tps', uid, 'draws')
    os.makedirs(d, exist_ok=True)
    psi = np.full((1, 2, msing), 0.5)
    xr.Dataset({'mpert': (('nn', 'code'), [[mpert, mpert + 8]]), 'psi_n_rational': (('nn', 'code', 'r'), psi)},
               coords={'nn': [4], 'code': ['rdcon', 'stride']}).to_netcdf(os.path.join(d, f'{c}.nc'), engine='scipy')
    fq.write_json(os.path.join(d, f'{c}.json'), dict(case_key=f'{uid}_{c}', shot=100, t_ms=1500, variant='base',
                                                     unit_id=uid, draw=c, archive='a.h5', status='ok',
                                                     inputs=dict(nx=256, delta_mhigh=8), **rec))


def test_memory_classes(study, monkeypatch):
    """Each item carries its predicted peak (a fill: STRIDE at its lost n; a check: RDCON's Delta' at its n) and is
    fed by priority within its memory class; a session takes only the items that fit it, and refuses a run that
    does not."""
    uid = '100_1500_base'
    _set_code_check(study, fraction=0.0, flagged_n_only=True, mem_classes=[9, 16, 35, 64])
    _sized_draw(study, uid, 0, 65, 38, code_killed={'stride': [4]}, stride_flagged_n=[4])   # fill: STRIDE 1.7 GiB
    _sized_draw(study, uid, 1, 49, 28, stride_flagged_n=[4])                                 # RDCON 8.7 GiB
    _sized_draw(study, uid, 2, 65, 38, stride_flagged_n=[4])                                 # RDCON 19.1 GiB
    _sized_draw(study, uid, 3, 93, 64, stride_flagged_n=[4])                                 # RDCON 64.4: no class
    q = fq.Queue(st.qroot(study, code_check=True))
    assert st.feed_codes(study) == 4
    need = {k: it['need_gib'] for k, it in q.units(refresh=True).items()}
    assert need == {f'{uid}.f0': 1.7, f'{uid}.d1': 8.7, f'{uid}.d2': 19.1, f'{uid}.d3': 64.4}
    assert [st._mem_class(need[f'{uid}.{k}'], [9, 16, 35, 64]) for k in ('f0', 'd1', 'd2', 'd3')] == [9, 16, 35, 64]
    _set_code_check(study, fraction=1.0, flagged_n_only=True, mem_classes=[9, 16, 35, 64])
    _sized_draw(study, uid, 4, 20, 5)                     # picked, class 9, where the fill still waits
    _sized_draw(study, uid, 5, 65, 38)                    # picked, class 35: behind the waiting flagged d2
    _sized_draw(study, uid, 6, 30, 10)                    # picked, class 9 too
    assert st.feed_codes(study) == 0
    q.finish(f'{uid}.f0')
    assert st.feed_codes(study) == 2 and {f'{uid}.d4', f'{uid}.d6'} <= set(q.units(refresh=True))
    assert [st._fits_session(study, 16, q.units()[f'{uid}.{k}']) for k in ('d1', 'd2', 'd3')] == [True, False, False]
    assert st._fits_session(study, 64, q.units()[f'{uid}.d3'])               # fits no class: taken, then refused
    monkeypatch.setenv('TPS_MEM_GIB', '64')
    with pytest.raises(RuntimeError, match='predicted peak 64 GiB does not fit the session'):
        st._check_memory(64.4)
    st._check_memory(19.1)


def test_item_need_follows_the_studys_inputs(study):
    """An item runs with the study's tps_kwargs as they are now, so its predicted peak uses them over the draw's
    recorded inputs (here nx)."""
    uid = '100_1500_base'
    _set_code_check(study, mem_classes=[9, 16, 35, 64])
    _sized_draw(study, uid, 0, 49, 28, stride_flagged_n=[4])
    it, cfg = dict(shot=100, unit_id=uid, draws=[0]), st.load_config(study)
    assert st._item_need_gib(study, cfg, it) == pytest.approx(8.7, abs=0.01)
    cfg['tps_kwargs']['nx'] = 128
    assert st._item_need_gib(study, cfg, it) == pytest.approx(4.9, abs=0.01)
