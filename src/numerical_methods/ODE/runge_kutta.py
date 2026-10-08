from typing import Callable

def two(f: Callable, t0: float, y0: float, tf: float, n: int = 100):

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