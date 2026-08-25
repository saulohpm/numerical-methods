import numpy as np
from scipy import integrate

def approx_function(f, x, L = 1, n = 4):
    """
    Approximate f(x) using its Fourier series expansion on the interval
    [-L, L].

    Parameters
    ----------
    f : callable
        Periodic function to approximate. Must accept and return a float.
    x : ndarray
        Array of points (mesh), at which the Fourier series
        approximation is evaluated.
    L : float, optional
        Half-period of the function, i.e. the function is approximated
        over [-L, L] (default 1).
    n : int, optional
        Number of harmonics (terms) used in the Fourier series (default 4).

    Returns
    -------
    float or ndarray
        Approximate value(s) of f(x) given by the truncated Fourier series
        with n harmonics, matching the shape of x.
    """
    
    a0 = 1 / L * integrate.quad(f, -L, L)[0]

    cn = lambda k, x: np.cos((k * np.pi * x) / L)
    sn = lambda k, x: np.sin((k * np.pi * x) / L)

    an = lambda k: (1 / L) * integrate.quad(lambda x: f(x) * cn(k, x), -L, L)[0]
    bn = lambda k: (1 / L) * integrate.quad(lambda x: f(x) * sn(k, x), -L, L)[0]

    def fn(x):

        un = a0 * 1 / 2

        for k in range(1, n + 1):
            un += an(k) * cn(k, x)
            un += bn(k) * sn(k, x)

        return un
    return fn(x)