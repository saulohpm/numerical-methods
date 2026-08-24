from .elimination import pivoting, gauss
from .decomposition import LU

def calculate(matrix, method: str = "auto"):
    """
    Computes the determinant of a square matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix.

    method : {"auto", "gauss", "lu"}, default = "auto"
        Algorithm used to compute the determinant.

    Returns
    -------
    float
        Determinant of the input matrix.
    """

    if method.lower() == "gauss":
        A_pivoted, p = pivoting(matrix)
        _, factors = gauss(A_pivoted)

        det = (-1) ** p
        for f in factors:
            det *= f

        return det
    
    else:
        A_pivoted, p = pivoting(matrix)
        L, _ = LU(A_pivoted)
        n = len(L)

        det = (-1) ** p
        for i in range(n):
            det *= L[i][i]

        return det