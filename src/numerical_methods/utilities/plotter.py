from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

def plot_function(f: Callable, x, a: float = None, b: float = None, f2: Callable = None):
    """
    Plot a function, optionally shading the area under the curve between
    a and b, or overlaying a second function (e.g. a Fourier approximation).

    Parameters
    ----------
    f : callable
        Function to plot. Must accept an array-like x and return an
        array-like y.
    x : array_like
        Array of x-values over which f is evaluated and plotted.
    a : float, optional
        Lower bound used to shade the area under the curve (default None).
    b : float, optional
        Upper bound used to shade the area under the curve (default None).
    f2 : callable, optional
        Second function to overlay on the same plot, e.g. a Fourier series
        approximation. Must accept an array-like x and return an
        array-like y (default None).

    Returns
    -------
    None
        Displays the plot; does not return a value.
    """

    if (a is None) != (b is None):
        raise ValueError("Provide 'a' and 'b' together, or neither.")

    y = f(x)

    plt.figure(figsize=(14, 6))

    if a is not None and b is not None and f2 is None:
        plt.plot(x, y, color = 'blue', label = f"∫f(x)dx")
        plt.fill_between(x, y, alpha = 0.3)

        plt.title("Definite Integral")
        plt.xlabel("x")
        plt.ylabel(f"∫f(x)dx")
        plt.xlim(a, b)
        y_max = np.max(y)
        margem = y_max * 0.1 if y_max != 0 else 1.0
        plt.ylim(0, y_max + margem)

    elif a is None and b is None and f2 is None:
        plt.plot(x, y, color = 'blue', label = "f(x)")

        plt.title("Function Plot")
        plt.xlabel("x")
        plt.ylabel("y")

    else:
        y2 = f2(x) if callable(f2) else f2

        plt.title("Interactions using two functions")
        plt.plot(x, y, label = "f1")
        plt.plot(x, y2, label = "f2")
        plt.xlim(a, b)

        y_lo = min(np.min(y), np.min(y2))
        y_hi = max(np.max(y), np.max(y2))
        margem = (y_hi - y_lo) * 0.1 if y_hi != y_lo else 1.0
        plt.ylim(y_lo - margem, y_hi + margem)

    plt.legend()
    plt.grid(True, alpha=0.75)
    plt.show()