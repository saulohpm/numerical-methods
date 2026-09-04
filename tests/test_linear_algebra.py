import numpy as np
from numerical_methods.linear_algebra import (jacobian, elimination, decomposition, determinant, linear_system, eigenvalues)
from numerical_methods.linear_algebra.eigenvalues import power_method, inverse_power_method

# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------
def F(x):
    return np.array([x[0] ** 2 + x[1], x[0] * x[1]])

x = np.array([3.0, 2.0])
h = 1e-6
EXPECTED_jacobian = np.array([[6.0, 1.0], [2.0, 3.0]])
J = jacobian.calculate(F, x, h)
assert np.all(np.abs(J - EXPECTED_jacobian) < 1e-4)

# ---------------------------------------------------------------------------
# Gauss Elimination
# ---------------------------------------------------------------------------
matrix = np.array([[2.0, 1.0, 5.0], [4.0, 4.0, 6.0], [2.0, 3.0, 8.0]])
EXPECTED_gausselimination = np.array([[1.0, 0.5, 2.5], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
result, _ = elimination.gauss(matrix.copy())
assert np.all(np.abs(result - EXPECTED_gausselimination) < 1e-6)

# ---------------------------------------------------------------------------
# Pivoting
# ---------------------------------------------------------------------------
matrix_pivoting = np.array([[1.0, 2.0, 2.0], [3.0, 6.0, 1.0], [2.0, 6.0, -1.0]])
EXPECTED_pivoting = np.array([[3.0, 6.0, 1.0], [2.0, 6.0, -1.0], [1.0, 2.0, 2.0]])
result_pivoting, _ = elimination.pivoting(matrix_pivoting.copy())
assert np.all(np.abs(result_pivoting - EXPECTED_pivoting) < 1e-6)

# ---------------------------------------------------------------------------
# LU Decomposition (Crout)
# ---------------------------------------------------------------------------
A_lu = np.array([[2., 1., 1.], [4., -6., 0.], [-2., 7., 2.]])
EXPECTED_L = np.array([[2., 0., 0.], [4., -8., 0.], [-2., 8., 1.]])
EXPECTED_U = np.array([[1., 0.5, 0.5], [0., 1., 0.25], [0., 0., 1.]])

L, U = decomposition.LU(A_lu.copy())

assert np.allclose(L, EXPECTED_L)
assert np.allclose(U, EXPECTED_U)
assert np.allclose(np.tril(L), L)
assert np.allclose(np.triu(U), U)
assert np.allclose(np.diag(U), np.ones(A_lu.shape[0]))
assert np.allclose(L @ U, A_lu)

# ---------------------------------------------------------------------------
# Cholesky Decomposition
# ---------------------------------------------------------------------------
A_chol = np.array([
    [10., 2., 1., 3., 0.],
    [2., 12., 2., 1., 4.],
    [1., 2., 15., 3., 2.],
    [3., 1., 3., 14., 5.],
    [0., 4., 2., 5., 13.]
])

U_chol = decomposition.cholesky(A_chol.copy())

assert np.allclose(np.triu(U_chol), U_chol)
assert np.allclose(U_chol.T @ U_chol, A_chol, atol=1e-6)

# ---------------------------------------------------------------------------
# QR Decomposition (Alternative Gram-Schmidt)
# ---------------------------------------------------------------------------
A_qr = np.array([
    [12., -51., 4.],
    [6., 167., -68.],
    [-4., 24., -41.]
])

Q, R = decomposition.QR(A_qr.copy())

assert np.allclose(np.triu(R), R)
assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-6)
assert np.allclose(Q @ R, A_qr, atol=1e-6)

# ---------------------------------------------------------------------------
# Linear System Solver (default)
# ---------------------------------------------------------------------------
A_sys = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
b_sys = np.array([8.0, -11.0, -3.0])

EXPECTED_solution = np.array([2.0, 3.0, -1.0])
result = linear_system.solve(A_sys, b_sys)
assert np.all(np.abs(result - EXPECTED_solution) < 1e-6)

# ---------------------------------------------------------------------------
# Linear System Solver + Determinant
# ---------------------------------------------------------------------------
b_chol = np.array([15., 20., 30., 25., 18.])
reference_solution = np.linalg.solve(A_chol, b_chol)
reference_det = np.linalg.det(A_chol)

for method in ("gauss", "lu", "cholesky", "QR"):
    x_method = linear_system.solve(A_chol, b_chol, method=method)
    assert np.linalg.norm(x_method - reference_solution) < 1e-6, method

for method in ("gauss", "lu"):
    det_method = determinant.calculate(A_chol, method=method)
    assert np.abs(det_method - reference_det) < 1e-6, method

# ---------------------------------------------------------------------------
# Eigenvalues
# ---------------------------------------------------------------------------
A = np.array([[5.0, 0.0],
              [0.0, 2.0]])

X0 = np.array([1.0, 1.0])

EXPECTED_MAX = 5.0
EXPECTED_MIN = 2.0


def test_power_method():
    eigenvalue, _ = power_method(A, X0, tol = 1e-10, n = 100)

    assert abs(eigenvalue - EXPECTED_MAX) < 1e-8


def test_inverse_power_method():
    eigenvalue, _ = inverse_power_method(A, X0, tol = 1e-10, n = 100)

    assert abs(eigenvalue - EXPECTED_MIN) < 1e-8


def test_jacobi_method_eigenvalues():
    A_result, _ = eigenvalues.jacobi_method(A, n = 100)

    computed_max = np.max(np.diag(A_result))
    computed_min = np.min(np.diag(A_result))

    assert abs(computed_max - EXPECTED_MAX) < 1e-8
    assert abs(computed_min - EXPECTED_MIN) < 1e-8

# ---------------------------------------------------------------------------
# Eigenvectors
# ---------------------------------------------------------------------------
def test_power_method_eigenvector():
    _, eigenvector = eigenvalues.power_method(A, X0, tol = 1e-10, n = 100)

    expected = np.array([1.0, 0.0])

    assert np.allclose(np.abs(eigenvector), expected, atol = 1e-5)


def test_inverse_power_method_eigenvector():
    _, eigenvector = eigenvalues.inverse_power_method(A, X0, tol = 1e-10, n = 100)

    expected = np.array([0.0, 1.0])

    assert np.allclose(np.abs(eigenvector), expected, atol = 1e-5)


def test_jacobi_method_eigenvalues():
    A_result, _ = eigenvalues.jacobi_method(A, n = 100)

    computed = np.sort(np.diag(A_result))
    expected = np.sort(np.linalg.eigvalsh(A))

    assert np.allclose(computed, expected, atol = 1e-5)

print("All linear_algebra tests PASSED.")