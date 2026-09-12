from . import linear_system, determinant
import numpy as np

def newton_raphson(F, J, x0, tol: float, max_iter: int):

    x = x0

    for i in range(max_iter):
        Fx = F(x)

        if np.linalg.norm(Fx) < tol:
            return x

        Jx = J(x)

        if determinant.calculate(Jx) == 0:
            raise ValueError("ERROR: Singular Jacobian, method failed.")

        deltax = linear_system.solve(Jx, -Fx)
        x_next = x + deltax

        if np.linalg.norm(x_next - x) < tol:
            return x_next

        x = x_next

    raise ValueError("ERROR: Maximum number of iterations reached without convergence")