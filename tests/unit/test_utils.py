import os

import numpy as np
import pytest

from tearing_physics_suite.utils import _get_num_cpus, create_dense_log_paramvals, eq_stem, run_code, trim_nans


def test_trim_nans():
    M = np.full((3, 3), np.nan)
    M[:2, :2] = [[1, 2], [3, 4]]
    E = np.ones((3, 3))
    dp, n_nan, err = trim_nans(M, E)
    np.testing.assert_array_equal(dp, [[1, 2], [3, 4]])
    assert n_nan == 1 and err.shape == (2, 2)


@pytest.mark.parametrize('name,stem', [
    ('/a/b/g147131.02300_DIIID_KEFIT', 'g147131.02300_DIIID_KEFIT'),
    ('/a/eq.geqdsk', 'eq'), ('eq.eqdsk', 'eq'), ('eq.gfile', 'eq'), ('eq.ifile', 'eq'), ('eq.v2.txt', 'eq.v2.txt')])
def test_eq_stem(name, stem):
    assert eq_stem(name) == stem


def test_get_num_cpus(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '3')
    assert _get_num_cpus() == 3


def test_create_dense_log_paramvals_runs():
    vals = create_dense_log_paramvals(1e-1, 1e-3, points_per_decade=5)
    assert len(vals) >= 10 and np.isclose(max(vals), 1e-1) and np.isclose(min(vals), 1e-3)


@pytest.mark.parametrize('val,n', [('16(x2)', 16), ('8,4', 8), ('12', 12)])
def test_get_num_cpus_slurm_forms(monkeypatch, val, n):
    for var in ('TPS_NUM_WORKERS', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', val)
    assert _get_num_cpus() == n
    monkeypatch.setenv('TPS_NUM_WORKERS', '5')
    assert _get_num_cpus() == 5


def test_run_code_timeout_kills_group(monkeypatch, tmp_path):
    import time
    monkeypatch.setenv('TPS_CODE_TIMEOUT', '1')
    assert run_code('exit 3') == 3
    t0 = time.time()
    with pytest.raises(TimeoutError):
        run_code(f'sleep 30 & echo $! > {tmp_path}/pid; wait')
    assert time.time() - t0 < 20
    pid = int((tmp_path / 'pid').read_text())
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
