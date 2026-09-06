from typing import Callable
import math

def forward(f: Callable, x0: float, deltax: float = 1e-5):
    """
    Approximate the derivative of f at x0 using the forward difference formula.

    Parameters
    ----------
    f : callable
        Function to differentiate. Must accept and return a float.
    x0 : float
        Point at which the derivative is evaluated.
    deltax : float, optional
        Step size used in the finite difference approximation (default 1e-5).

    Returns
    -------
    float
        Approximate value of f'(x0).
    """
    return (f(x0 + deltax) - f(x0)) / deltax

def backward(f: Callable, x0: float, deltax: float = 1e-5):
    """
    Approximate the derivative of f at x0 using the backward difference formula.

    Parameters
    ----------
    f : callable
        Function to differentiate. Must accept and return a float.
    x0 : float
        Point at which the derivative is evaluated.
    deltax : float, optional
        Step size used in the finite difference approximation (default 1e-5).

    Returns
    -------
    float
        Approximate value of f'(x0).
    """
    return (f(x0) - f(x0 - deltax)) / deltax

def central(f: Callable, x0: float, deltax: float = 1e-5):
    """
    Approximate the derivative of f at x0 using the central difference formula.

    Parameters
    ----------
    f : callable
        Function to differentiate. Must accept and return a float.
    x0 : float
        Point at which the derivative is evaluated.
    deltax : float, optional
        Step size used in the finite difference approximation (default 1e-5).

    Returns
    -------
    float
        Approximate value of f'(x0). Generally more accurate than
        forward/backward difference for the same step size.
    """
    return (f(x0 + deltax) - f(x0 - deltax)) / (2 * deltax)


def central_nth(f: Callable, x0: float, deltax: float = 1e-5, degree: int = 1):
    """
    Approximate the n-th derivative of f at x0 using a central
    finite difference formula built directly from f (does not
    chain derivatives of derivatives).

    Parameters
    ----------
    f : callable
        Function to differentiate. Must accept and return a float.
    x0 : float
        Point at which the derivative is evaluated.
    deltax : float, optional
        Step size used in the finite difference approximation (default 1e-5).
    degree : int
        Order of the derivative (degree = 0 returns f(x0), degree = 1 is the
        usual central difference, etc.).

    Returns
    -------
    float
        Approximate value of f^(n)(x0).
    """

    total = 0.0

    for i in range(degree + 1):
        coef = (-1) ** i * math.comb(degree, i)
        total += coef * f(x0 + (degree / 2 - i) * deltax)

    return total / deltax ** degree