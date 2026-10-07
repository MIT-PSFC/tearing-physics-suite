import numpy as np
import pytest

from tearing_physics_suite.utils import _get_num_cpus, create_dense_log_paramvals, eq_stem, trim_nans


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
