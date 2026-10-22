import os
import time

import numpy as np
import pytest

from tearing_physics_suite.utils import (
    CODE_WALL_S,
    _get_num_cpus,
    code_timer,
    create_dense_log_paramvals,
    eq_stem,
    fits_memory,
    gpec_home,
    node_slot,
    run_code,
    session_mem_gib,
    trim_nans,
)


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


def test_run_code_stack_and_env(tmp_path):
    """stack_bytes raises the child's soft stack limit (to at most its hard limit); env is its environment."""
    import resource
    assert run_code(f'echo $(ulimit -s) $X > {tmp_path}/o', env={**os.environ, 'X': 'y'}, stack_bytes=1 << 30) == 0
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    want = 1 << 30 if hard == resource.RLIM_INFINITY else min(1 << 30, hard)
    assert (tmp_path / 'o').read_text().split() == [str(want // 1024), 'y']


def test_node_slot_limits_concurrency(monkeypatch, tmp_path):
    import threading
    monkeypatch.setenv('TPS_WORK_ROOT', str(tmp_path))
    active, peak, lock = [0], [0], threading.Lock()

    def job():
        with node_slot('x', 2, poll=0.01):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            with lock:
                active[0] -= 1
    ts = [threading.Thread(target=job) for _ in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 2
    with node_slot('x', None):
        pass


def test_code_timer_adds_up_and_counts_failures():
    """code_timer adds each block's wall time to CODE_WALL_S[key], also when the block raises."""
    CODE_WALL_S.clear()
    for _ in range(2):
        with code_timer('rdcon_n1'):
            time.sleep(0.05)
    with pytest.raises(TimeoutError), code_timer('stride_n1'):
        time.sleep(0.05)
        raise TimeoutError
    assert 0.1 <= CODE_WALL_S['rdcon_n1'] < 1 and 0.05 <= CODE_WALL_S['stride_n1'] < 1
    CODE_WALL_S.clear()


def test_fits_memory(monkeypatch):
    """A run fits a session when its predicted peak is within 90 % of the session's memory ($TPS_MEM_GIB)."""
    monkeypatch.delenv('TPS_MEM_GIB', raising=False)
    assert session_mem_gib() is None and fits_memory(1e3)          # no limit
    monkeypatch.setenv('TPS_MEM_GIB', '5')
    assert session_mem_gib() == 5 and fits_memory(4.5) and not fits_memory(4.6)
    assert fits_memory(30, 35) and not fits_memory(32, 35)


def test_gpec_home(monkeypatch):
    monkeypatch.setenv('TPSHOME', '/tps')
    monkeypatch.delenv('TPS_GPEC_DIR', raising=False)
    assert gpec_home() == '/tps/submodules/GPEC'
    monkeypatch.setenv('TPS_GPEC_DIR', '/runtime/gpec')
    assert gpec_home() == '/runtime/gpec'
