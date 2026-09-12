import numpy as np

def GaussJacobi(A, b, tol: float, max_iter: int):
    """
    Solve a linear system Ax = b using the iterative Gauss-Jacobi method.

    Parameters
    ----------
    A : array_like
        Coefficient matrix of the system, of shape (n, n).
    b : array_like
        Vector of independent terms, of shape (n,).
    tol : float
        Tolerance for the stopping criterion, based on the Euclidean
        norm of the difference between consecutive iterations.
    max_iter : int
        Maximum number of allowed iterations.

    Returns
    -------
    x : numpy.ndarray
        Approximate solution vector, of shape (n,). If the tolerance is
        not met within `max_iter` iterations, the last approximation
        obtained is returned and a warning message is printed.

    Notes
    -----
    Convergence of the Gauss-Jacobi method is not guaranteed for any
    matrix `A`. A sufficient (but not necessary) condition for
    convergence is that `A` be strictly diagonally dominant.
    """

    n = len(A)
    xp = np.zeros(n)
    x = np.zeros(n)

    for k in range(max_iter):
        norma = 0

        for i in range(n):
            soma = 0

            for j in range(i):
                soma += A[i][j] * x[j]

            for j in range(i + 1, n):
                soma += A[i][j] * x[j]

            xp[i] = (b[i] - soma) / A[i][i]
            norma += (x[i] - xp[i]) ** 2

        if norma ** 0.5 <= tol:
            return xp

        x = xp.copy()

    print("NOTICE: The tolerance was not met.")

    return x


def GaussSeidel(A, b, tol: float, max_iter: int):
    """
    Solve a linear system Ax = b using the iterative Gauss-Seidel method.

    Parameters
    ----------
    A : array_like
        Coefficient matrix of the system, of shape (n, n).
    b : array_like
        Vector of independent terms, of shape (n,).
    tol : float
        Tolerance for the stopping criterion, based on the Euclidean
        norm of the difference between consecutive iterations.
    max_iter : int
        Maximum number of allowed iterations.

    Returns
    -------
    x : numpy.ndarray
        Approximate solution vector, of shape (n,). If the tolerance is
        not met within `max_iter` iterations, the last approximation
        obtained is returned and a warning message is printed.

    Notes
    -----
    Unlike the Gauss-Jacobi method, Gauss-Seidel uses values already
    updated within the same iteration (components j < i), which
    generally results in faster convergence when the method converges.
    A sufficient condition for convergence is that `A` be strictly
    diagonally dominant or symmetric positive definite.
    """
    
    n = len(A)
    x = np.zeros(n)
    xp = np.zeros(n)

    for k in range(max_iter):
        norma = 0

        for i in range(n):
            soma = 0

            for j in range(i):
                soma += A[i][j] * xp[j]

            for j in range(i + 1, n):
                soma += A[i][j] * x[j]

            xp[i] = (b[i] - soma) / A[i][i]
            norma = norma + (x[i] - xp[i]) ** 2

        if norma ** 0.5 <= tol:
            return xp

        x = xp.copy()

    print("NOTICE: The tolerance was not met.")

    return x