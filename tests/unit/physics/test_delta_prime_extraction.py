import numpy as np
import pytest

from tearing_physics_suite.physics.delta_prime_extraction import (
    delta_prime_2nn_couple,
    delta_prime_full_couple,
    delta_prime_nn_couple,
    delta_prime_no_couple,
    extract_delta_primes,
    get_delta_prime_divisors,
    matrix_cofactor,
)

M2 = np.array([[2.0, 0.5], [0.3, -4.0]])
M3 = np.array([[2.0, 0.5, 0.2], [0.3, -4.0, 0.7], [0.1, 0.6, -9.0]])


def test_no_couple_is_diagonal():
    np.testing.assert_array_equal(delta_prime_no_couple(M3), np.diag(M3))


def test_cofactor_2x2():
    a, b, c, d = M2.ravel()
    np.testing.assert_allclose(matrix_cofactor(M2), [[d, -c], [-b, a]], rtol=1e-6)


def test_full_couple_2x2_by_hand():
    a, b, c, d = M2.ravel()
    np.testing.assert_allclose(delta_prime_full_couple(M2), [a - b * c / d, d - b * c / a], rtol=1e-6)


@pytest.mark.parametrize('M', [M2, M3])
def test_full_couple_is_inverse_diagonal(M):
    # sum_j M_ij C_ij = det(M), so the fully coupled value is det/C_ii = 1/(M^-1)_ii
    np.testing.assert_allclose(delta_prime_full_couple(M), 1 / np.diag(np.linalg.inv(M)), rtol=1e-6)


def test_nn_couple_3x3_by_hand():
    C = np.linalg.inv(M3).T * np.linalg.det(M3)
    want = [M3[i, i] + sum(M3[i, j] * C[i, j] for j in (i - 1, i + 1) if 0 <= j < 3) / C[i, i] for i in range(3)]
    np.testing.assert_allclose(delta_prime_nn_couple(M3), want, rtol=1e-6)


def test_2nn_equals_full_for_3x3():
    np.testing.assert_allclose(delta_prime_2nn_couple(M3), delta_prime_full_couple(M3), rtol=1e-6)


def test_divisors_are_cofactor_diagonal():
    C = np.linalg.inv(M3).T * np.linalg.det(M3)
    np.testing.assert_allclose(get_delta_prime_divisors(M3), np.diag(C), rtol=1e-6)


@pytest.mark.parametrize('code', ['rdcon', 'stride', 'pest3'])
def test_extract_delta_primes_on_stored_output(code_fixture, code):
    ds = code_fixture(code)
    out = extract_delta_primes(ds)
    dp = out.Delta_prime_surf
    assert dp.dims == ('Delta_prime_type', 'r')
    assert list(dp.Delta_prime_type.values) == ['single helicity', 'nn coupled', '2nn coupled', 'full coupled']
    M = ds.Delta_prime.sel(i=0).values
    ok = ~np.isnan(np.diag(M))  # PEST3 pads unconverged surfaces with NaN
    Mk = M[np.ix_(ok, ok)]
    np.testing.assert_allclose(dp.sel(Delta_prime_type='single helicity').values[ok], np.diag(Mk))
    np.testing.assert_allclose(dp.sel(Delta_prime_type='full coupled').values[ok],
                               1 / np.diag(np.linalg.inv(Mk)), rtol=1e-5)
    assert np.isnan(dp.values[:, ~ok]).all()
    if code == 'pest3':
        assert 'Delta_prime_perr_surf' in out
