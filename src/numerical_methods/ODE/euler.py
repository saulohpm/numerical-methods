from typing import Callable
from numpy.typing import NDArray
import numpy as np

def explicit(f: Callable, y0: float | NDArray[np.float64], t0: float, tf: float, n: int = 256):
    """
    Solve the initial value problem y' = f(t, y), y(t0) = y0, using the
    explicit (forward) Euler method.

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE. Must have the signature ``f(t, y)``,
        accept two floats and return a float (the derivative y').
    y0 : float or ndarray
        Initial value y(t0). Use a NumPy array for systems of ODEs.
    t0 : float
        Initial time.
    tf : float
        Final time.
    n : int
        Number of steps. The step size is ``h = (tf - t0) / n``.

    Returns
    -------
    t : list of float
        Time grid with n + 1 points, from t0 to tf.
    y : list of float or list of ndarray
        Approximation of y(t) at each point of the time grid.

    Raises
    ------
    ValueError
        If ``n`` is less than 1.

    Notes
    -----
    The update rule is ``y[i + 1] = y[i] + h * f(t[i], y[i])``.
    The local truncation error is O(h**2) and the global error is O(h).
    For systems of ODEs, ``y0`` and the output of ``f`` must be NumPy
    arrays, not Python lists.
    """

    if n < 1:
        raise ValueError("n must be higher than 0!")

    h = (tf - t0) / n

    t = [0.0] * (n + 1)
    y = [0.0] * (n + 1)

    t[0] = t0
    y[0] = y0

    for i in range(n):
        t[i + 1] = t0 + (i + 1) * h
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y