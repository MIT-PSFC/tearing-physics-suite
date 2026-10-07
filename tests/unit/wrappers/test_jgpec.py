"""jGPEC wrapper without Julia: inputs, h5 reader, solver runner (fake server) and the server protocol."""
import os
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tearing_physics_suite.physics.combine import align_surfaces, compile_xarrays
from tearing_physics_suite.utils import eq_stem
from tearing_physics_suite.wrappers import jgpec
from tearing_physics_suite.wrappers.d3d_wall import d3d_wall_points, write_jgpec_wall_file
from tearing_physics_suite.wrappers.gpec_inputs import write_rdcon_stride_inputs

DATA = Path(__file__).resolve().parents[2] / 'data'
EQ = DATA / 'g147131.02300_DIIID_KEFIT'
H5 = DATA / 'jgpec'
FORTRAN_WALL = '/fusion/projects/tmdb/src/tps_D_work/jobs/part2/runs/d3d/d3dwall_gpec/gpecvac_wall.dat'


def _h5(solver, wall):
    return str(H5 / f'g147131_n1_{solver}_{wall}.h5')


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    return lambda **kw: write_rdcon_stride_inputs(str(tmp_path_factory.mktemp('in')) + '/', EQ.name, verbose=False, **kw)


# ---- inputs ----
def test_toml_sections_walls(inputs):
    sec = lambda solver='galerkin', **kw: jgpec.gpec_toml_sections(EQ.name, 1, solver, inputs(**kw))  # noqa: E731
    assert sec(vac_flag='t', a_wall=21)['Wall'] == {'shape': 'nowall'}
    assert sec(vac_flag='t', a_wall=0.3)['Wall'] == {'shape': 'conformal', 'a': 0.3}
    assert sec(vac_flag='t', a_wall=1, ishape=8)['Wall']['shape'] == 'd3d'
    assert sec(vac_flag='t', a_wall=21, ishape=8)['Wall'] == {'shape': 'nowall'}
    assert sec(vac_flag='f')['ForceFreeStates']['vac_flag'] is False
    with pytest.raises(ValueError, match='Riccati'):
        sec('riccati', vac_flag='f')
    with pytest.raises(NotImplementedError):
        sec(vac_flag='t', a_wall=1, ishape=4)


def test_toml_sections_solver_keys_and_overrides(inputs):
    ins = inputs()
    gal = jgpec.gpec_toml_sections(EQ.name, 2, 'galerkin', ins, mtheta_jgpec=129, nstep_jgpec=7, a_jgpec=0.5)
    ric = jgpec.gpec_toml_sections(EQ.name, 2, 'riccati', ins)
    assert gal['ForceFreeStates']['integrator'] == 'galerkin' and gal['ForceFreeStates']['gal_rpec_flag'] is False
    assert not any(k.startswith('gal_') for k in ric['ForceFreeStates'])
    assert gal['ForceFreeStates']['nn_low'] == gal['ForceFreeStates']['nn_high'] == 2
    assert ric['ForceFreeStates']['sing_order'] == int(ins['sing_order'])
    assert gal['Equilibrium']['mtheta'] == 129 and gal['ForceFreeStates']['nstep'] == 7 and gal['Wall']['a'] == 0.5


def test_write_gpec_toml_round_trip(tmp_path, inputs):
    sections = jgpec.gpec_toml_sections(EQ.name, 1, 'galerkin', inputs())
    jgpec.write_gpec_toml(tmp_path, sections)
    with open(tmp_path / 'gpec.toml', 'rb') as f:
        assert tomllib.load(f) == sections


def test_check_jgpec_request(capsys):
    assert jgpec.check_jgpec_request('galerkin', 't', True) == ('galerkin',)
    assert jgpec.check_jgpec_request(['galerkin', 'riccati'], "'f'", True) == ('galerkin', 'riccati')
    assert 'Riccati is skipped' in capsys.readouterr().out
    for args in ((('riccati',), 'f', True), (('stride',), 't', True), ((), 't', True), (('galerkin',), 't', False)):
        with pytest.raises(ValueError):
            jgpec.check_jgpec_request(*args)


# ---- reader ----
@pytest.mark.parametrize('solver,wall', [('riccati', 'nowall'), ('galerkin', 'nowall'), ('galerkin', 'wall')])
def test_read_jgpec_h5(solver, wall):
    ds = jgpec.read_jgpec_h5(_h5(solver, wall))
    assert ds.Delta_prime.dims == ('i', 'r_prime', 'r')
    np.testing.assert_array_equal(ds.r, [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(ds.q_rational, [2, 3, 4, 5, 6])
    assert ('A_prime' in ds) == (solver == 'galerkin')
    assert ds.attrs['n'] == 1 and ds.attrs['qlim'] == 6.2


def test_reader_orientation_matches_rdcon(code_fixture):
    """Galerkin with the wall on the plasma vs the stored RDCON run (same case): same off-diagonal placement."""
    rd = code_fixture('rdcon').Delta_prime.sel(i=0).values
    jg = jgpec.read_jgpec_h5(_h5('galerkin', 'wall')).Delta_prime.sel(i=0).values
    assert np.isclose(jg[0, 0], rd[0, 0], rtol=0.05)
    for a, b in ((0, 1), (1, 0), (0, 2), (1, 2)):
        assert abs(jg[a, b] - rd[a, b]) < abs(jg[b, a] - rd[a, b])


def test_galerkin_subset(tmp_path):
    """A Delta' matrix over fewer surfaces than listed: Galerkin's in-band, in-domain subset."""
    import h5py
    p = str(tmp_path / 'gpec.h5')
    shutil.copy(_h5('galerkin', 'nowall'), p)
    with h5py.File(p, 'r+') as f:
        dp = f['SingularSurfaces/Delta_prime_matrix'][()]
        del f['SingularSurfaces/Delta_prime_matrix']
        f['SingularSurfaces/Delta_prime_matrix'] = dp[:4, :4]
        psilim = f['Info/psilim'][()]
        psi = f['SingularSurfaces/rational_psi'][()]
        psi[-1] = psilim + 1e-4
        f['SingularSurfaces/rational_psi'][...] = psi
        for k in ('pest3_A', 'pest3_B', 'pest3_Gamma'):
            del f['SingularSurfaces/' + k]
    ds = jgpec.read_jgpec_h5(p)
    np.testing.assert_array_equal(ds.r, [2, 3, 4, 5])


# ---- alignment and combining ----
def test_align_surfaces_subset_and_extra(code_fixture, capsys):
    """jGPEC on surfaces q = 9 (not in RDCON), 2, 4, 5: RDCON's q = 3 and 6 become NaN, q = 9 is dropped."""
    rd = code_fixture('rdcon')
    sub = jgpec.read_jgpec_h5(_h5('riccati', 'nowall')).isel(r=[0, 2, 3], r_prime=[0, 2, 3])
    dp = np.full((2, 4, 4), 7.0)
    dp[:, 1:, 1:] = sub.Delta_prime.values
    r = [9.0, 2.0, 4.0, 5.0]
    ds = xr.Dataset({'Delta_prime': (('i', 'r_prime', 'r'), dp), 'q_rational': ('r', r),
                     'psi_n_rational': ('r', np.r_[0.3, sub.psi_n_rational.values + 1e-4])},
                    coords={'i': [0, 1], 'r': r, 'r_prime': r})
    out = align_surfaces(ds, rd)
    assert 'not in the reference dropped' in capsys.readouterr().out
    np.testing.assert_array_equal(out.r, rd.r)
    got = out.Delta_prime.sel(i=0).values
    assert np.isnan(got[1]).all() and np.isnan(got[:, 4]).all()
    np.testing.assert_array_equal(got[np.ix_([0, 2, 3], [0, 2, 3])], sub.Delta_prime.sel(i=0).values)
    np.testing.assert_array_equal(out.q_rational[[0, 2, 3]], [2, 4, 5])


def test_compile_xarrays_with_jgpec(code_fixture):
    rd, st = code_fixture('rdcon'), code_fixture('stride')
    gal = jgpec.read_jgpec_h5(_h5('galerkin', 'wall'))
    nan = jgpec._nan_like(rd)
    comb, _, _ = compile_xarrays(rd, st, None, True, True, False, {'nn': 1}, None,
                                 {'jGPEC_riccati': nan, 'jGPEC_galerkin': gal}, {'jGPEC_riccati': False, 'jGPEC_galerkin': True})
    assert list(comb.code.values) == ['rdcon', 'stride', 'jGPEC_galerkin', 'jGPEC_riccati']
    dps = comb.Delta_prime_surf.isel(nn=0, Delta_prime_type=0)
    assert np.isclose(dps.sel(code='jGPEC_galerkin').isel(r=0), dps.sel(code='rdcon').isel(r=0), rtol=0.05)
    assert dps.sel(code='jGPEC_riccati').isnull().all()


# ---- runner with a fake server ----
class _FakeServer:
    def __init__(self, fail=()):
        self.fail, self.runs = fail, []

    def run(self, run_dir, timeout=0):
        self.runs.append(run_dir)
        toml = tomllib.load(open(os.path.join(run_dir, 'gpec.toml'), 'rb'))
        solver = toml['ForceFreeStates']['integrator']
        if solver in self.fail:
            return False, 'boom'
        wall = 'nowall' if toml['ForceFreeStates']['vac_flag'] else 'wall'
        shutil.copy(_h5(solver, wall), os.path.join(run_dir, 'gpec.h5'))
        return True, ''


def test_run_both_solvers(tmp_path, monkeypatch, inputs, code_fixture):
    srv = _FakeServer()
    monkeypatch.setattr(jgpec, 'get_server', lambda threads=1: srv)
    xrs, ran = jgpec.run_jgpec_solvers(str(EQ), 1, str(tmp_path), inputs(vac_flag='t'), code_fixture('rdcon'),
                                       solvers=('galerkin', 'riccati'), output_location=str(tmp_path / 'out'))
    assert ran == {'jGPEC_galerkin': True, 'jGPEC_riccati': True}
    assert [os.path.basename(d) for d in srv.runs] == ['jgpec_galerkin', 'jgpec_riccati']
    for code in ran:
        saved = xr.load_dataset(tmp_path / 'out' / f'{eq_stem(str(EQ))}_{code}_n1.nc')
        np.testing.assert_array_equal(saved.Delta_prime, xrs[code].Delta_prime)


def test_run_riccati_skipped_with_wall_and_failure(tmp_path, monkeypatch, inputs, code_fixture):
    srv = _FakeServer(fail=('galerkin',))
    monkeypatch.setattr(jgpec, 'get_server', lambda threads=1: srv)
    xrs, ran = jgpec.run_jgpec_solvers(str(EQ), 1, str(tmp_path), inputs(vac_flag='f'), code_fixture('rdcon'),
                                       solvers=('galerkin', 'riccati'))
    assert ran == {'jGPEC_galerkin': False, 'jGPEC_riccati': False}
    assert xrs['jGPEC_galerkin'] is None and xrs['jGPEC_riccati'].Delta_prime.isnull().all()
    assert len(srv.runs) == 1


def test_run_writes_d3d_wall(tmp_path, monkeypatch, inputs, code_fixture):
    monkeypatch.setattr(jgpec, 'get_server', lambda threads=1: _FakeServer())
    jgpec.run_jgpec_solvers(str(EQ), 1, str(tmp_path), inputs(vac_flag='t', ishape=8, a_wall=1), code_fixture('rdcon'))
    toml = tomllib.load(open(tmp_path / 'jgpec_galerkin' / 'gpec.toml', 'rb'))
    assert toml['Wall']['shape'] == str(tmp_path / 'jgpec_galerkin' / 'd3d_wall.dat')
    assert int(open(toml['Wall']['shape']).readline()) == toml['ForceFreeStates']['mthvac']


# ---- DIII-D wall ----
def test_d3d_wall_points(tmp_path):
    r, z = d3d_wall_points(480)
    assert len(r) == 480 and np.hypot(r[-1] - r[0], z[-1] - z[0]) > 1e-3  # no repeated endpoint
    assert z[0] == 0 and z[1] < 0  # outboard midplane, then downward (clockwise)
    area = 0.5*np.sum(r*np.roll(z, -1) - np.roll(r, -1)*z)
    assert area < 0
    path = write_jgpec_wall_file(str(tmp_path / 'w.dat'), 480)
    rows = np.loadtxt(path, skiprows=3)
    assert rows.shape == (480, 3) and np.allclose(rows[:, 1], r)


@pytest.mark.skipif(not os.path.exists(FORTRAN_WALL), reason='no Fortran VACUUM wall dump')
def test_d3d_wall_matches_fortran():
    ref = np.loadtxt(FORTRAN_WALL, skiprows=3)
    r, z = d3d_wall_points(200000)
    d = [np.min(np.hypot(p[1] - r, p[2] - z)) for p in ref[::20]]
    assert max(d) < 1e-4


# ---- server protocol (a Python stand-in for Julia) ----
FAKE_JULIA = '''#!{py}
import sys, time
print("TPS_READY", flush=True)
for line in sys.stdin:
    d = line.strip()
    if d == "TPS_QUIT":
        break
    if d.endswith("hang"):
        time.sleep(60)
    if d.endswith("die"):
        sys.exit(1)
    print("TPS_DONE ok" if d.endswith("good") else "TPS_DONE err bad dir", flush=True)
'''


@pytest.fixture
def fake_julia(tmp_path, monkeypatch):
    exe = tmp_path / 'julia'
    exe.write_text(FAKE_JULIA.format(py=sys.executable))
    exe.chmod(0o755)
    monkeypatch.setenv('JGPEC_JULIA', str(exe))
    return tmp_path


def test_server_runs_and_restarts(fake_julia):
    srv = jgpec.JuliaServer(log_path=str(fake_julia / 'log'))
    try:
        assert srv.run('/x/good', timeout=20) == (True, '')
        assert srv.run('/x/bad', timeout=20) == (False, 'bad dir')
        with pytest.raises(RuntimeError, match='exited'):
            srv.run('/x/die', timeout=20)
        assert srv.run('/x/good', timeout=20)[0]  # restarted
    finally:
        srv.close()
    assert not srv.alive()


def test_server_timeout(fake_julia):
    srv = jgpec.JuliaServer(log_path=str(fake_julia / 'log'))
    with pytest.raises(TimeoutError):
        srv.run('/x/hang', timeout=1)
    assert not srv.alive()
    srv.close()
