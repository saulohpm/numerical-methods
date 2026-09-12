import numpy as np

def GaussJacobi(A, b, tol: float, max_iter: int):

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


def GaussSeidel(A, b, tol: float, max_inter: int):

    n = len(A)
    x = np.zeros(n)
    xp = np.zeros(n)

    for k in range(max_inter):
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