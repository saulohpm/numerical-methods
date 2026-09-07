import matplotlib.pyplot as plt
import numpy as np

def plot_function(f, x, a = None, b = None, f2 = None):
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
    f2 : array_like, optional
        Precomputed y-values of a second function to overlay on the same
        plot, e.g. a Fourier series approximation (default None).

    Returns
    -------
    None
        Displays the plot; does not return a value.
    """
    
    y = f(x)

    plt.figure(figsize=(14,6))

    if a is not None and b is not None and f2 is None:
        plt.plot(x, y, color='blue', label = "∫f(x)dx")
        plt.fill_between(x, y, alpha=0.3)

        plt.title("Definite Integral")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.xlim(a, b)
        y_max = np.max(y)
        margem = (y_max) * 0.1 if y_max != 0 else 1.0
        plt.ylim(0, y_max + margem)

    elif a is None and b is None and f2 is None:
        plt.plot(x, y, color='blue', label = "f(x)")

        plt.title("Function Plot")
        plt.xlabel("x")
        plt.ylabel("f(x)")

    else:
        y2 = f2(x)
        plt.title(f"Interactions using two functions")
        plt.plot(x, y, label='f(x)')
        plt.plot(x, y2, label='f2(x)')
        plt.xlim(a, b)
        y_min, y_max, f2_min, f2_max = np.min(y), np.max(y), np.min(y2), np.max(y2)
        max = y_max if y_max > f2_max else f2_max
        min = y_min if y_min < f2_min else f2_min
        margem = (max - min) * 0.1 if max != min else 1.0
        plt.ylim(min - margem, max + margem)
        
    plt.legend()
    plt.grid(True, alpha = 0.75)

    plt.show()