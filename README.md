# Numerical Methods

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-5.9-orange)
[![Tests](https://img.shields.io/github/actions/workflow/status/saulohpm/NumericalMethods/tests.yml?label=Tests)](https://github.com/saulohpm/NumericalMethods/actions/workflows/tests.yml)

Educational Python library implementing classical numerical methods from scratch, including root finding, numerical integration, differentiation, linear algebra, Fourier and Taylor series, with unit tests and performance benchmarks.

This started as a study project to practice numerical analysis concepts and has since been reorganized as an installable Python package, with unit tests and docstrings, as a way to also practice good project structure.

## Implemented Methods

- **Root finding:** bisection, Newton-Raphson, Ridders
- **Integration:** trapezoidal, midpoint, rectangle, Simpson (1st and 2nd), Gauss-Legendre, Monte Carlo
- **Differentiation:** finite differences, Richardson extrapolation
- **Series:** Taylor, Fourier
- **Linear algebra:** Gaussian elimination, LU decomposition, Cholesky decomposition, determinant, eigenvalues, Jacobian, linear systems

## Installation

```bash
gh repo clone saulohpm/NumericalMethods
cd NumericalMethods
pip install -e ".[dev]"
```

This installs the package in editable mode, along with `pytest` for running
the tests.

## Usage

```python
import numerical_methods as nm

f = lambda x: x ** 2
result = nm.trapezoidal_integrate(f, 0, 1, n = 1000)
print(result)
# 0.33333349999999995
```

For more detailed examples, including comparisons between methods and
plots, see **[Demonstration Notebook](examples/demonstration.ipynb)**.

## Benchmarks

Performance benchmarks comparing the implemented methods are available in **[Benchmark Notebook](examples/benchmark.ipynb)**

The notebook compares execution time, numerical error, and convergence against SciPy and numpy reference implementations.

## Notes and limitations

This project is meant for learning and experimentation, not for
production-grade numerical computing (for that, `numpy`/`scipy` are the
better choice). Some things worth knowing if you look at the code:

* The Taylor series approximation builds higher-order derivatives by
  repeatedly applying finite differences, which is numerically fragile, the default step size was chosen to keep results stable for a reasonable range of inputs, but it isn't a general-purpose solution.
* Methods are implemented directly from their mathematical definitions, favoring clarity over performance.

## Project structure

```text
NumericalMethods/
│
├── pyproject.toml
├── LICENSE.txt
├── README.md
├── .gitattributes
├── .gitignore
│
├── examples/
│   ├── benchmark.ipynb
│   └── demonstration.ipynb
│
├── src/
│   └── numerical_methods/
│       ├── __init__.py
│       │
│       ├── differentiation/
│       │   ├── __init__.py
│       │   ├── finite_differences.py
│       │   └── richardson.py
│       │
│       ├── integration/
│       │   ├── __init__.py
│       │   ├── gauss_legendre.py
│       │   ├── midpoint.py
│       │   ├── monte_carlo.py
│       │   ├── rectangle.py
│       │   ├── simpson1.py
│       │   ├── simpson2.py
│       │   └── trapezoidal.py
│       │
│       ├── linear_algebra/
│       │   ├── __init__.py
│       │   ├── decomposition.py
│       │   ├── determinant.py
│       │   ├── eigenvalues.py
│       │   ├── elimination.py
│       │   ├── jacobian.py
│       │   └── linear_system.py
│       │
│       ├── roots/
│       │   ├── __init__.py
│       │   ├── bisection.py
│       │   ├── newton_raphson.py
│       │   └── ridders.py
│       │
│       ├── series/
│       │   ├── __init__.py
│       │   ├── fourier.py
│       │   └── taylor.py
│       │
│       ├── utilities/
│       │   ├── __init__.py
│       │   └── errors.py
│       │
│       └── visualization/
│           ├── __init__.py
│           └── plotter.py
│
└── tests/
    ├── __init__.py
    ├── test_differentiation.py
    ├── test_integration.py
    ├── test_linear_algebra.py
    ├── test_roots.py
    └── test_series.py
```

## References
- Linge, S., & Langtangen, H. P. *Programming for Computations – Python: A Gentle Introduction to Numerical Simulations with Python 3.6*. 2nd ed. Springer, 2019. Available at: https://link.springer.com/book/10.1007/978-3-030-16877-3

- Cardoso, E. L. *Fundamentos de Cálculo Numérico: Notas de Aula*. PPGEM. Available at: <https://github.com/CodeLenz/Notas-de-aula/blob/main/Fundamentos%20de%20Matem%C3%A1tica/Fundamentos_de_Algebra_Computacional.pdf>.

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.