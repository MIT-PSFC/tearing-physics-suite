"""Ideal-stability pass (drivers/ideal_pass.py): stop rules, STRIDE where RDCON is pathological, store."""
import json
import os

import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.drivers import fsqueue as fq
from tearing_physics_suite.drivers import ideal_pass as ip
from tearing_physics_suite.drivers import parallel as st
from tearing_physics_suite.drivers import profile_read
from tearing_physics_suite.wrappers import gpec


def _fake_gpec(monkeypatch, rdcon, stride, calls=None):
    """GPEC_resistive_calculation stand-in: rdcon[n] = (nzero, [(psi, q)]) (crit.out), stride[n] = nzero (None: no
    attribute); calls collects (code, n, kwargs)."""
    def run(eq, n, run_rdcon=False, run_stride=False, working_dir=None, **kw):
        os.makedirs(working_dir, exist_ok=True)
        code = 'rdcon' if run_rdcon else 'stride'
        if calls is not None:
            calls.append((code, n, kw))
        if code == 'rdcon':
            z, cross = rdcon[n]
            with open(os.path.join(working_dir, 'crit.out'), 'w') as fh:
                fh.writelines(f' Zero crossing at psi = {p:.3E}, q = {q:.3E}\n    12  1.0E-01\n' for p, q in cross)
        else:
            z = stride[n]
        ds = xr.Dataset(attrs={} if z is None else dict(nzero=z))
        return (ds, None, True, False, {}) if code == 'rdcon' else (None, ds, False, True, {})
    monkeypatch.setattr(gpec, 'GPEC_resistive_calculation', run)


def test_stop_rules_and_stride(tmp_path, monkeypatch):
    """n=1 stable; n=2 RDCON crosses on q = 3/2 (pathological): STRIDE's 0 decides, go on; n=3 on q = 1.333 as
    printed, 3 q - 4 = -1e-3, within 3 x its print resolution (pathological): STRIDE's 1 decides, stop; n=4 not
    run. Then a RDCON-unstable run sampled for STRIDE."""
    calls = []
    _fake_gpec(monkeypatch, {1: (0, []), 2: (1, [(0.126, 1.5)]), 3: (1, [(0.2, 1.3334)])}, {2: 0, 3: 1}, calls)
    icfg = dict(ip.DEFAULTS, nvec=[1, 2, 3, 4], sample=1.0)
    ds, _ = ip._ideal_case('eq', {}, dict(vac_flag='t'), str(tmp_path), str(tmp_path / 'd.ideal.nc'), label='k',
                           icfg=icfg)
    assert [(c, n) for c, n, _ in calls] == [('rdcon', 1), ('rdcon', 2), ('stride', 2), ('rdcon', 3), ('stride', 3)]
    assert calls[0][2]['ode_flag'] == 't' and calls[0][2]['gal_flag'] == calls[0][2]['vac_flag'] == 'f'
    assert calls[2][2]['vac_flag'] == 't'                                  # STRIDE at the study's settings
    np.testing.assert_array_equal(ds.nzero.values, [[0, np.nan], [1, 0], [1, 1], [np.nan, np.nan]])
    assert ds.ideal_unstable.values.tolist() == [0, 0, 1, -1]
    assert ds.ideal_rational_crossing.values.tolist() == [0, 1, 1, -1]
    assert ds.ideal_stride_reason.values.tolist() == ['', 'pathological', 'pathological', '']
    assert ds.ideal_crossing_q.sel(nn=2).values[0] == 1.5
    _fake_gpec(monkeypatch, {1: (1, [(0.2, 1.1)])}, {1: 0})
    ds, _ = ip._ideal_case('eq', {}, {}, str(tmp_path), str(tmp_path / 'f.ideal.nc'), label='k', icfg=icfg)
    assert ds.ideal_stride_reason.values[0] == 'sample' and ds.ideal_unstable.values[0] == 1   # RDCON decides
    _fake_gpec(monkeypatch, {1: (5, [])}, {1: None})                       # a STRIDE build without nzero
    with pytest.raises(RuntimeError, match='no nzero'):
        ip._ideal_case('eq', {}, {}, str(tmp_path), str(tmp_path / 'e.ideal.nc'), label='k', icfg=icfg)


@pytest.fixture
def study(tmp_path, monkeypatch):
    s = str(tmp_path / 'study')
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps(dict(draws_per_item=2, tps_kwargs=dict(nvec=[1, 2], qlow=1.01, k0=0.8))))
    monkeypatch.setattr(st, 'versions', lambda: dict(tps='v1', gpec=None, jgpec=None))
    st.main(['init', '--study', s, '--config', str(cfg)])
    monkeypatch.setenv('TPS_WORK_ROOT', str(tmp_path))
    return s


def test_feed_run_compile(study, monkeypatch):
    """Items from the main pass's 'ok' draws; per-draw results compiled into <shot>_ideal.zarr; i-files kept until
    every draw's pass is final; only GPEC inputs reach GPEC."""
    uid = '100_1500_base'
    d = os.path.join(study, '100', 'tps', uid, 'draws')
    os.makedirs(d)
    for c, status in ((0, 'ok'), (1, 'ok'), (2, 'error'), (3, 'ok')):
        fq.write_json(os.path.join(d, f'{c}.json'), dict(case_key=f'{uid}_{c}', shot=100, t_ms=1500, variant='base',
                                                         unit_id=uid, draw=c, archive='a.h5', status=status))
    ip.main(['init', '--study', study])
    assert ip.ideal_pending(study, 100) == [f'{uid}_0', f'{uid}_1', f'{uid}_3']
    assert ip.feed(study) == 2 and ip.feed(study) == 0
    assert sorted(fq.Queue(ip.qroot(study)).units()) == [f'{uid}.i0', f'{uid}.i3']

    def read(path, draw_counts=None, eqdsk_out_dir=None, **kw):
        return [f'eq{c}' for c in draw_counts], [dict(bq_count=c) for c in draw_counts]
    monkeypatch.setattr(profile_read, 'read_bouquet_archive', read)
    calls = []
    _fake_gpec(monkeypatch, {1: (0, []), 2: (3, [(0.3, 1.25), (0.4, 1.3), (0.5, 1.4)])}, {2: 1}, calls)
    for item in (f'{uid}.i0', f'{uid}.i3'):
        assert [x['status'] for x in ip.run_item(study, item)['draws']] in (['ok', 'ok'], ['ok'])
    assert 'k0' not in calls[0][2] and calls[0][2]['qlow'] == 1.01
    assert ip.ideal_pending(study, 100) == []
    assert ip.compile_shot(study, 100) == 3
    ds = xr.open_zarr(ip.shot_store(study, 100))
    assert ds.case_key.values.tolist() == [f'{uid}_0', f'{uid}_1', f'{uid}_3']
    np.testing.assert_array_equal(ds.nzero.sel(code='stride').values, [[np.nan, 1]] * 3)
    assert ds.ideal_unstable.values.tolist() == [[0, 1]] * 3                # n=2: 3 crossings, STRIDE decides
    assert ds.ideal_stride_reason.values.tolist() == [['', 'pathological']] * 3
