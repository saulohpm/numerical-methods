def gauss(matrix):
    """
    Upper-triangularizes a square matrix via Gaussian elimination,
    normalizing each pivot row to a unit diagonal.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square (n x n) input matrix. Not modified; operates on
        an internal copy.

    Returns
    -------
    numpy.ndarray
        Upper triangular matrix with unit diagonal.
    list[float]
        The pivot factors F_i used to normalize each row, in
        order. Useful later for computing the determinant via
        det(A) = (-1)**p * prod(F_i), where p is the number of
        row swaps from pivoting(matrix).

    Warnings
    --------
    No internal pivoting. Fails or loses precision if matrix[i][i]
    is zero or near-zero at any step i. Use function pivoting(matrix) to solve this.
    """
    A = matrix.copy()
    n = len(A)

    factors = []

    for i in range(n):
        factors.append(A[i][i])

        Li = A[i] / A[i][i]
        for k in range(i + 1, n):
            Lk = A[k] - A[k][i] * Li
            A[k] = Lk
        A[i] = Li

    return A, factors


def pivoting(matrix):
    """
    Partial pivoting on a square matrix: for each column i, swaps
    rows so the largest-magnitude entry (row i to n-1) sits at
    the diagonal position [i][i].

    Parameters
    ----------
    matrix : numpy.ndarray
        Square (n x n) input matrix. Not modified; operates on
        an internal copy.

    Returns
    -------
    numpy.ndarray
        New matrix with rows permuted so each diagonal entry is
        the largest in its column (from that row downward).
    int
        Number of row swaps (pivotings) performed, p. Useful later
        for determining the sign of the determinant (each swap
        flips the sign by a factor of -1).
    """

    A = matrix.copy()
    n = len(A)

    p = 0

    for i in range(n):
        highest = A[i][i]
        highest_index = i

        for j in range(i, n):
            if abs(A[j][i]) > abs(highest):
                highest = A[j][i]
                highest_index = j

        if A[i][i] != highest:
            actual_line = A[i].copy()
            A[i] = A[highest_index]
            A[highest_index] = actual_line
            
            p += 1

    return A, p