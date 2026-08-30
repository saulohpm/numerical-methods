"""
NumericalMethods
================

Educational implementations of classic numerical methods, including
numerical integration, differentiation, root finding, series
approximation, linear algebra, and visualization utilities.

Modules
-------
differentiation
    Forward, backward, central, and nth-order finite differences,
    plus Richardson extrapolation.

integration
    Rectangle, midpoint, trapezoidal, Simpson's, Monte Carlo,
    and Gauss-Legendre integration.

roots
    Bisection, Newton-Raphson, and Ridders' root-finding methods.

series
    Taylor and Fourier series approximation.

linear_algebra
    Gaussian elimination, pivoting, LU and Cholesky decomposition,
    QR decomposition, determinants, linear systems, Jacobians,
    and eigenvalue methods.

visualization
    Simple plotting utilities.

Example
--------
>>> import numerical_methods as nm

>>> f = lambda x: x ** 2
>>> result = nm.trapezoidal_integrate(f, 0, 1, n = 1000)
>>> print(result)
>>> # 0.33333349999999995
"""

# ---------------------------------------------------------------------------
# Differentiation
# ---------------------------------------------------------------------------

from .differentiation.finite_differences import (
    forward as fd_forward_derivative,
    backward as fd_backward_derivative,
    central as fd_central_derivative,
    central_nth as fd_nth_derivative,
)

from .differentiation.richardson import calculate as richardson_derivative

# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

from .integration.rectangle import integrate as rectangle_integrate

from .integration.trapezoidal import (
    integrate as trapezoidal_integrate,
    double_integrate as trapezoidal_double_integrate,
)

from .integration.midpoint import (
    integrate as midpoint_integrate,
    double_integrate as midpoint_double_integrate,
    triple_integrate as midpoint_triple_integrate,
)

from .integration.simpson1 import integrate as simpson1_integrate
from .integration.simpson2 import integrate as simpson2_integrate
from .integration.monte_carlo import integrate as monte_carlo_integrate
from .integration.gauss_legendre import integrate as gauss_legendre_integrate

# ---------------------------------------------------------------------------
# Linear Algebra
# ---------------------------------------------------------------------------

from .linear_algebra.jacobian import calculate as jacobian_calculate

from .linear_algebra.elimination import (
    gauss as gauss_elimination,
    pivoting as pivoting_elimination,
)

from .linear_algebra.decomposition import (
    LU as lu_decomposition,
    cholesky as cholesky_decomposition,
    QR as QR_decomposition,
)

from .linear_algebra.determinant import calculate as determinant_calculate

from .linear_algebra.eigenvalues import (
    power_method as power_method_calculate,
    inverse_power_method as inverse_power_method_calculate,
    jacobi_method as jacobi_method_calculate,
)

from .linear_algebra.linear_system import solve as linearsystem_solve

# ---------------------------------------------------------------------------
# Root Finding
# ---------------------------------------------------------------------------

from .roots.bisection import calculate as bisection_calculate
from .roots.newton_raphson import calculate as newton_raphson_calculate
from .roots.ridders import calculate as ridders_calculate

# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

from .series.taylor import approx_function as taylor_approx
from .series.fourier import approx_function as fourier_approx

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

from .utilities.errors import error_calculate, objective_functions

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

from .visualization.plotter import plot_function

# ---------------------------------------------------------------------------
# Package Metadata
# ---------------------------------------------------------------------------

__version__ = "0.6.2"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Differentiation
    "fd",
    "fd_forward_derivative",
    "fd_backward_derivative",
    "fd_central_derivative",
    "fd_nth_derivative",
    "richardson_derivative",

    # Integration
    "rectangle_integrate",
    "trapezoidal_integrate",
    "trapezoidal_double_integrate",
    "midpoint_integrate",
    "midpoint_double_integrate",
    "midpoint_triple_integrate",
    "simpson1_integrate",
    "simpson2_integrate",
    "monte_carlo_integrate",
    "gauss_legendre_integrate",

    # Linear Algebra
    "jacobian_calculate",
    "gauss_elimination",
    "pivoting_elimination",
    "lu_decomposition",
    "cholesky_decomposition",
    "QR_decomposition",
    "determinant_calculate",
    "power_method_calculate",
    "inverse_power_method_calculate",
    "jacobi_method_calculate",
    "linearsystem_solve",

    # Root Finding
    "bisection_calculate",
    "newton_raphson_calculate",
    "ridders_calculate",

    # Series
    "taylor_approx",
    "fourier_approx",

    # Utilities
    "error_calculate",
    "objective_functions",

    # Visualization
    "plot_function",
]