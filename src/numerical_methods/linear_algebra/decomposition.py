import numpy as np

def LU(matrix):
    """
    Performs the LU decomposition of a square matrix using Crout's method.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix to decompose.

    Returns
    -------
    L : numpy.ndarray
        Lower triangular matrix.

    U : numpy.ndarray
        Upper triangular matrix with unit diagonal.
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ERROR: The matrix must be a square matrix.")

    n = len(matrix)
    A = matrix.copy()
    U = np.zeros((n, n))
    L = np.zeros((n, n))

    for j in range(n):
        for i in range(n):

            if i > j:
                soma = 0

                for k in range(0, j):
                    soma += L[i][k] * U[k][j]

                L[i][j] = A[i][j] - soma

            elif i < j:
                soma = 0

                for k in range(0, i):
                    soma += L[i][k] * U[k][j]

                U[i][j] = (A[i][j] - soma) / L[i][i]

            else:
                soma = 0
                U[i][i] = 1

                for k in range(i):
                    soma += L[i][k] * U[k][i]

                L[i][i] = A[i][i] - soma

    return L, U


def cholesky(matrix):
    """
    Compute the Cholesky decomposition of a symmetric positive-definite matrix.

    Parameters
    ----------
    matrix : array_like
        Input square symmetric positive-definite matrix.

    Returns
    -------
    U : ndarray
        Upper triangular matrix such that A = U.T @ U.

    Notes
    -----
    Cholesky decomposition requires the input matrix to be symmetric
    and positive-definite.
    """

    if matrix.ndim != 2:
        raise ValueError("ERROR: The matrix must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ERROR: The matrix must be a square matrix.")

    if not np.allclose(matrix, matrix.T):
        raise ValueError("ERROR: The matrix must be equal to its transpose")

    A = matrix.copy()
    n = len(A)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            soma = 0

            if i == j:
                for k in range(i):
                    soma += U[k][i] ** 2

                U[i][i] = np.sqrt(A[i][i] - soma)

            elif j > i:
                for k in range(i):
                    soma += U[k][i] * U[k][j]

                U[i][j] = (A[i][j] - soma) / U[i][i]

    return U

def QR(matrix):
    """
    Compute the QR decomposition of a matrix using the modified
    Gram-Schmidt algorithm.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix to decompose.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple (Q, R), where Q is an orthogonal matrix and
        R is an upper triangular matrix.
    """

    A = matrix.copy()
    n = len(A)

    U = np.zeros((n, n))
    Q = np.zeros((n, n))
    R = np.zeros((n, n))

    for k in range(n):

        U[:, k] = A[:, k]

        for j in range(k):

            R[j, k] = np.dot(U[:, k], Q[:, j])

            U[:, k] = U[:, k] - R[j, k] * Q[:, j]

        R[k, k] = np.linalg.norm(U[:, k])

        Q[:, k] = U[:, k] / R[k, k]

    return Q, R