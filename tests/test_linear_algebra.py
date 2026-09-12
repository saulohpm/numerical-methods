from numerical_methods.linear_algebra import (
    jacobian, elimination, decomposition, determinant,
    linear_system, eigenvalues, iterative_methods, nonlinear_system
)
import numpy as np

A_CHOL = np.array([
    [10., 2., 1., 3., 0.],
    [2., 12., 2., 1., 4.],
    [1., 2., 15., 3., 2.],
    [3., 1., 3., 14., 5.],
    [0., 4., 2., 5., 13.]
])
B_CHOL = np.array([15., 20., 30., 25., 18.])


def test_jacobian():
    F = lambda x: np.array([x[0] ** 2 + x[1], x[0] * x[1]])
    x = np.array([3.0, 2.0])
    expected = np.array([[6.0, 1.0], [2.0, 3.0]])

    J = jacobian.calculate(F, x, h=1e-6)

    assert np.allclose(J, expected, atol=1e-4)


def test_gauss_elimination():
    matrix = np.array([[2.0, 1.0, 5.0], [4.0, 4.0, 6.0], [2.0, 3.0, 8.0]])
    expected = np.array([[1.0, 0.5, 2.5], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])

    result, _ = elimination.gauss(matrix.copy())

    assert np.allclose(result, expected, atol=1e-6)


def test_pivoting():
    matrix = np.array([[1.0, 2.0, 2.0], [3.0, 6.0, 1.0], [2.0, 6.0, -1.0]])
    expected = np.array([[3.0, 6.0, 1.0], [2.0, 6.0, -1.0], [1.0, 2.0, 2.0]])

    result, _ = elimination.pivoting(matrix.copy())

    assert np.allclose(result, expected, atol=1e-6)


def test_lu_decomposition():
    A = np.array([[2., 1., 1.], [4., -6., 0.], [-2., 7., 2.]])

    L, U = decomposition.LU(A.copy())

    assert np.allclose(np.tril(L), L)
    assert np.allclose(np.triu(U), U)
    assert np.allclose(np.diag(U), np.ones(A.shape[0]))
    assert np.allclose(L @ U, A)


def test_cholesky_decomposition():
    U = decomposition.cholesky(A_CHOL.copy())

    assert np.allclose(np.triu(U), U)
    assert np.allclose(U.T @ U, A_CHOL, atol=1e-6)


def test_qr_decomposition():
    A = np.array([[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]])

    Q, R = decomposition.QR(A.copy())

    assert np.allclose(np.triu(R), R)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-6)
    assert np.allclose(Q @ R, A, atol=1e-6)


def test_linear_system_solve():
    A = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
    b = np.array([8.0, -11.0, -3.0])
    expected = np.array([2.0, 3.0, -1.0])

    result = linear_system.solve(A, b)

    assert np.allclose(result, expected, atol=1e-6)


def test_linear_system_methods_and_determinant():
    reference_solution = np.linalg.solve(A_CHOL, B_CHOL)
    reference_det = np.linalg.det(A_CHOL)

    for method in ("gauss", "lu", "cholesky", "QR"):
        x = linear_system.solve(A_CHOL, B_CHOL, method=method)
        assert np.linalg.norm(x - reference_solution) < 1e-6, method

    for method in ("gauss", "lu"):
        det = determinant.calculate(A_CHOL, method=method)
        assert np.abs(det - reference_det) < 1e-6, method


def test_gauss_jacobi():
    reference_solution = np.linalg.solve(A_CHOL, B_CHOL)

    x = iterative_methods.GaussJacobi(A_CHOL, B_CHOL, tol=1e-8, max_iter=200)

    assert np.linalg.norm(x - reference_solution) < 1e-6


def test_gauss_seidel():
    reference_solution = np.linalg.solve(A_CHOL, B_CHOL)

    x = iterative_methods.GaussSeidel(A_CHOL, B_CHOL, tol=1e-8, max_iter=200)

    assert np.linalg.norm(x - reference_solution) < 1e-6


def test_power_method():
    A = np.array([[5.0, 0.0], [0.0, 2.0]])
    x0 = np.array([1.0, 1.0])

    eigenvalue, eigenvector = eigenvalues.power_method(A, x0, tol=1e-10, n=100)

    assert abs(eigenvalue - 5.0) < 1e-8
    assert np.allclose(np.abs(eigenvector), [1.0, 0.0], atol=1e-5)


def test_inverse_power_method():
    A = np.array([[5.0, 0.0], [0.0, 2.0]])
    x0 = np.array([1.0, 1.0])

    eigenvalue, eigenvector = eigenvalues.inverse_power_method(A, x0, tol=1e-10, n=100)

    assert abs(eigenvalue - 2.0) < 1e-8
    assert np.allclose(np.abs(eigenvector), [0.0, 1.0], atol=1e-5)


def test_jacobi_method_eigenvalues():
    A = np.array([[5.0, 0.0], [0.0, 2.0]])

    A_result, _ = eigenvalues.jacobi_method(A, n=100)

    computed = np.sort(np.diag(A_result))
    expected = np.sort(np.linalg.eigvalsh(A))

    assert np.allclose(computed, expected, atol=1e-5)


def test_newton_raphson():
    F = lambda x: np.array([
        x[0] ** 2 + x[1] ** 2 - 4,
        x[0] - x[1]
    ])
    J = lambda x: jacobian.calculate(F, x, h=1e-6)
    x0 = np.array([1.0, 1.0])
    expected = np.array([np.sqrt(2), np.sqrt(2)])

    x = nonlinear_system.newton_raphson(F, J, x0, tol=1e-8, max_iter=100)

    assert np.allclose(x, expected, atol=1e-6)


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]

    for test in tests:
        test()

    print(f"All {len(tests)} linear_algebra tests PASSED.")