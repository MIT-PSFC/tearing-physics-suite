# Unit tests for the PEST3/GPEC build logic with git, make and cmake mocked (no compiling, no network).
import os
import subprocess

import pytest

import tearing_physics_suite.wrappers.build.build_GPEC_PEST3 as bg

PIN = 'a' * 40
URL = 'https://example.org/CODE'


@pytest.fixture
def fake_run(monkeypatch):
    """Record subprocess.run calls; answer git queries from state, fail commands starting with state['fail']."""
    state = {'calls': [], 'registered': True, 'head': PIN, 'fail': None}

    def run(cmd, cwd=None, **kwargs):
        cmd = list(cmd)
        state['calls'].append((cmd, cwd))
        out, code = '', 0
        if cmd[:3] == ['git', 'ls-files', '-s']:
            out = f'160000 {PIN} 0\t{cmd[-1]}' if state['registered'] else ''
        elif cmd[:3] == ['git', 'rev-parse', 'HEAD']:
            out = state['head']
        if state['fail'] and cmd[:len(state['fail'])] == state['fail']:
            code = 1
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr='')

    monkeypatch.setattr(bg.subprocess, 'run', run)
    return state


def _cmds(state):
    return [c for c, _ in state['calls']]


def test_fetch_uses_submodule_when_registered(tmp_path, fake_run):
    src = tmp_path / 'submodules' / 'GPEC'
    src.mkdir(parents=True)  # uninitialised submodule: empty dir
    assert bg.fetch_source(src, URL, branch='OFT_interface', home_dir=tmp_path)
    assert (['git', 'submodule', 'update', '--init', '--', 'submodules/GPEC'], tmp_path) in fake_run['calls']
    assert not any(c[:2] == ['git', 'clone'] for c in _cmds(fake_run))


def test_fetch_clones_when_not_registered(tmp_path, fake_run):
    fake_run['registered'] = False
    src = tmp_path / 'submodules' / 'GPEC'
    assert bg.fetch_source(src, URL, branch='OFT_interface', home_dir=tmp_path)
    assert ['git', 'clone', '--branch', 'OFT_interface', URL, str(src)] in _cmds(fake_run)
    assert not any(c[:2] == ['git', 'submodule'] for c in _cmds(fake_run))


def test_fetch_outside_tps_clones(tmp_path, fake_run):
    src = tmp_path / 'elsewhere' / 'PEST3'
    assert bg.fetch_source(src, URL, home_dir=tmp_path / 'tps')
    assert ['git', 'clone', URL, str(src)] in _cmds(fake_run)


@pytest.mark.parametrize('head, warns', [(PIN, False), ('b' * 40, True)])
def test_existing_clone_kept_and_drift_warned(tmp_path, fake_run, capsys, head, warns):
    fake_run['head'] = head
    src = tmp_path / 'submodules' / 'PEST3'
    (src / '.git').mkdir(parents=True)
    assert bg.fetch_source(src, URL, home_dir=tmp_path)
    assert ('WARNING' in capsys.readouterr().out) == warns
    assert not any(c[:2] in (['git', 'clone'], ['git', 'submodule']) for c in _cmds(fake_run))


def test_fetch_failure(tmp_path, fake_run):
    fake_run['fail'] = ['git', 'submodule']
    assert not bg.fetch_source(tmp_path / 'submodules' / 'GPEC', URL, home_dir=tmp_path)


def test_git_describe_needs_own_checkout(tmp_path, fake_run):
    assert bg.git_describe(tmp_path) == 'unknown'  # would describe the enclosing repo
    assert fake_run['calls'] == []


def _mock_compilers(monkeypatch):
    info = dict(fc='gfortran', cc='gcc', f77='gfortran', fflags_base='-O2', compiler_type='gfortran')
    monkeypatch.setattr(bg, 'detect_compilers', lambda mpi=True: info)
    monkeypatch.setattr(bg, 'get_cmake_c_flags', lambda cc: '')


@pytest.mark.parametrize('rebuild, remake, cleans, deps_removed', [
    (False, False, False, False),
    (False, True, True, False),
    (True, False, True, True),
])
def test_gpec_clean_logic(tmp_path, monkeypatch, fake_run, rebuild, remake, cleans, deps_removed):
    """remake: make clean then make; rebuild: also deps/lib; neither: no clean. Source always kept."""
    gpec = tmp_path / 'submodules' / 'GPEC'
    for d in ('.git', 'install', 'deps/lib'):
        (gpec / d).mkdir(parents=True)
    (gpec / 'deps' / 'lib' / 'libvac.a').write_text('x')
    (gpec / 'install' / 'makefile').write_text('x')
    monkeypatch.setattr(bg, 'tps_home', lambda: str(tmp_path))
    monkeypatch.setattr(bg, 'is_gpec_built', lambda d=None: False)
    _mock_compilers(monkeypatch)

    assert bg.build_GPEC({}, rebuild=rebuild, remake=remake, run_tests=False)
    make_cmds = [c for c in _cmds(fake_run) if c[0] == 'make']
    assert make_cmds[-1] == ['make']
    assert (['make', 'clean'] in make_cmds) == cleans
    if cleans:
        assert make_cmds.index(['make', 'clean']) < make_cmds.index(['make'])
    assert (not (gpec / 'deps' / 'lib' / 'libvac.a').exists()) == deps_removed
    assert (gpec / 'install' / 'makefile').exists()
    assert ['make', 'v'] not in make_cmds  # only with debug=True


def test_pest3_rebuild_keeps_source(tmp_path, monkeypatch, fake_run):
    """rebuild removes cmake_build/ in the tree, not the source; stop at cmake."""
    pest3 = tmp_path / 'submodules' / 'PEST3'
    (pest3 / '.git').mkdir(parents=True)
    (pest3 / 'scimake').mkdir()
    (pest3 / 'scimake' / 'SciInit.cmake').write_text('x')
    (pest3 / 'CMakeLists.txt').write_text('x')
    old = pest3 / 'cmake_build' / 'pest3' / 'pest3x'
    old.parent.mkdir(parents=True)
    old.write_text('x')
    monkeypatch.setattr(bg, 'tps_home', lambda: str(tmp_path))
    _mock_compilers(monkeypatch)
    fake_run['fail'] = ['cmake']

    assert not bg.build_PEST3({}, rebuild=True, run_tests=False, gpec_vacuum=False, work_dir=tmp_path / 'work')
    assert (pest3 / 'CMakeLists.txt').exists()
    assert not old.exists()
    assert not any(c[:2] in (['git', 'clone'], ['git', 'submodule']) for c in _cmds(fake_run))


def test_env_file_records_versions(tmp_path, monkeypatch):
    import tearing_physics_suite.wrappers.build.compiler_utils as cu
    monkeypatch.setenv('TPSHOME', os.environ.get('TPSHOME', ''))  # the script sets TPSHOME on import
    import tearing_physics_suite.wrappers.build.build_tearing_physics_suite as bts
    monkeypatch.setattr(cu, 'detect_compilers', lambda mpi=True: dict(fc='gfortran', cc='gcc', fflags_base='', compiler_type='gfortran'))
    monkeypatch.setattr(bts, 'git_describe', lambda p: f'desc-{p.name}')
    text = bts.write_env_file({}, tmp_path).read_text()
    assert '#   GPEC : desc-GPEC' in text and '#   PEST3: desc-PEST3' in text
