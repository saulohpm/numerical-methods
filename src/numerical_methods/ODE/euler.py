from typing import Callable

def euler_explicit(f: Callable, t0: float, y0: float, tf: float, n: int):
    """
    Solve the initial value problem y' = f(t, y), y(t0) = y0, using the
    explicit (forward) Euler method.

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE. Must have the signature ``f(t, y)``,
        accept two floats and return a float (the derivative y').
    t0 : float
        Initial time.
    y0 : float
        Initial value y(t0).
    tf : float
        Final time.
    n : int
        Number of steps. The step size is ``h = (tf - t0) / n``.

    Returns
    -------
    t : list of float
        Time grid with n + 1 points, from t0 to tf.
    y : list of float
        Approximation of y(t) at each point of the time grid.

    Notes
    -----
    The update rule is ``y[i + 1] = y[i] + h * f(t[i], y[i])``.
    The local truncation error is O(h**2) and the global error is O(h).
    """

    h = (tf - t0) / n

    t = [0] * (n + 1)
    y = [0] * (n + 1)

    t[0] = t0
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h *f(t[i], y[i])
        t[i + 1] = t[i] + h

    return t, y