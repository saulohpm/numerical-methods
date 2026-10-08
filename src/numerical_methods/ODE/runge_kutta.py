from typing import Callable
from numpy.typing import NDArray
import numpy as np

def two(f: Callable, y0: float | NDArray[np.float64], t0: float, tf: float, n: int = 256):
    """
    Solve an initial value problem using the explicit midpoint method (RK2).

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE, ``dy/dt = f(t, y)``. It must accept
        ``(t, y)`` and return a value of the same type and shape as ``y``.
    y0 : float or ndarray
        Initial condition ``y(t0)``. Use a NumPy array for systems of ODEs.
    t0 : float
        Initial time.
    tf : float
        Final time.
    n : int, optional
        Number of steps. Must be at least 1. Default is 256.

    Returns
    -------
    t : list of float
        Time grid with ``n + 1`` points, from ``t0`` to ``tf``.
    y : list of float or list of ndarray
        Approximate solution at each point of ``t``.

    Raises
    ------
    ValueError
        If ``n`` is less than 1.

    Notes
    -----
    The method is second-order accurate: local error is O(h^3) and global
    error is O(h^2), with ``h = (tf - t0) / n``. For systems of ODEs,
    ``y0`` and the output of ``f`` must be NumPy arrays, not Python lists.
    """

    if n < 1:
        raise ValueError("n must be higher than 0!")

    h = (tf - t0) / n

    t = [0.0] * (n + 1)
    y = [0.0] * (n + 1)

    t[0] = t0
    y[0] = y0

    for i in range(1, n + 1):
        t[i] = t0 + i * h

        k1 = h * f(t[i - 1], y[i - 1])
        k2 = h * f(t[i - 1] + 0.5 * h, y[i - 1] + 0.5 * k1)

        y[i] = y[i - 1] + k2

    return t, y


def four(f: Callable, y0: float | NDArray[np.float64], t0: float, tf: float, n: int = 256):
    """
    Solve an initial value problem using the classical fourth-order
    Runge-Kutta method (RK4).

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE, ``dy/dt = f(t, y)``. It must accept
        ``(t, y)`` and return a value of the same type and shape as ``y``.
    y0 : float or ndarray
        Initial condition ``y(t0)``. Use a NumPy array for systems of ODEs.
    t0 : float
        Initial time.
    tf : float
        Final time.
    n : int, optional
        Number of steps. Must be at least 1. Default is 256.

    Returns
    -------
    t : list of float
        Time grid with ``n + 1`` points, from ``t0`` to ``tf``.
    y : list of float or list of ndarray
        Approximate solution at each point of ``t``.

    Raises
    ------
    ValueError
        If ``n`` is less than 1.

    Notes
    -----
    The method is fourth-order accurate: local error is O(h^5) and global
    error is O(h^4), with ``h = (tf - t0) / n``. Each step evaluates ``f``
    four times. For systems of ODEs, ``y0`` and the output of ``f`` must be
    NumPy arrays, not Python lists.
    """


    if n < 1:
        raise ValueError("n must be higher than 0!")

    h = (tf - t0) / n

    t = [0.0] * (n + 1)
    y = [0.0] * (n + 1)

    t[0] = t0
    y[0] = y0

    for i in range(1, n + 1):
        t[i] = t0 + i * h

        k1 = h * f(t[i - 1], y[i - 1])
        k2 = h * f(t[i - 1] + 0.5 * h, y[i - 1] + 0.5 * k1)
        k3 = h * f(t[i - 1] + 0.5 * h, y[i - 1] + 0.5 * k2)
        k4 = h * f(t[i - 1] + h, y[i - 1] + k3)

        y[i] = y[i - 1] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y