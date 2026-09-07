from typing import Callable, Literal
from .finite_differences import forward, backward, central, central_nth

def derivative_function(f: Callable,
                        method: Literal["forward", "backward", "central", "central_nth"] = "central",
                        deltax: float = 1e-5,
                        degree: int = 1):
    """
    Constructs a derivative function of f, evaluable at a point (float) or on a grid (array), reusing point-by-point finite difference formulas.

    Parameters
    ----------
    f : callable
        Function to be differentiated. Must accept a float or np.ndarray and
        return the same type (i.e., it must be vectorizable — avoid
        `math.sin`, prefer `np.sin`).
    method : {"forward", "backward", "central", "central_nth"}, optional
        Finite difference scheme to use (default "central").
    deltax : float, optional
        Finite difference step size (default 1e-5).
    degree : int, optional
        Order of the derivative to compute (default 1). Only used when
        `method="central_nth"`.

    Returns
    -------
    callable
        Function f'(x), which accepts a scalar or array x.
    """

    methods = {"forward": forward, "backward": backward, "central": central, "central_nth": central_nth}

    if method not in methods:
        raise ValueError(f"The method '{method}' is invalid. Choise from {list(methods)}.")

    base = methods[method]

    if method != "central_nth":
        return lambda x: base(f, x, deltax)
    
    else:
        return lambda x: base(f, x, deltax, degree)