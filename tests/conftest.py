"""Shared pytest fixtures for TPS.

Markers (pyproject.toml): fortran (runs RDCON/STRIDE/PEST3), julia (runs jGPEC),
parallel (multiprocessing pool), slow. The default run excludes them all.
"""
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault('TPSHOME', str(REPO))

DATA = REPO / 'tests' / 'data'
FIXTURES = DATA / 'fixtures'
EQ = DATA / 'g147131.02300_DIIID_KEFIT'
# Reduced grids used by every Fortran test (fast, same paths as production).
REDUCED = dict(etol=1e-7, nx=64, mpsi=128, mtheta=129)
MRE = dict(Zeff=1.5, average_ion_mass=2.5, energy_confinement_time=0.12)


@pytest.fixture(autouse=True)
def _restore_cwd():
    """The wrappers chdir into working dirs; put the cwd back after every test."""
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


@pytest.fixture(scope='session')
def eq_file():
    return str(EQ)


@pytest.fixture(scope='session')
def kin_file():
    return str(EQ) + '.kin'


@pytest.fixture(scope='session')
def code_fixture():
    """Return a function giving the stored n=1 output Dataset for 'rdcon', 'stride' or 'pest3'."""
    import xarray as xr

    def _load(code):
        return xr.load_dataset(FIXTURES / f'g147131.02300_DIIID_KEFIT_{code}_n1.nc')
    return _load


@pytest.fixture(scope='session')
def golden_dir():
    """Golden reference outputs (A0). Set TPS_GOLDEN_DIR, or symlink tests/data/golden."""
    d = Path(os.environ.get('TPS_GOLDEN_DIR', DATA / 'golden'))
    if not d.is_dir():
        pytest.skip(f'no golden reference at {d}')
    return d
