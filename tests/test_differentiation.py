from numerical_methods.differentiation.finite_differences import forward, backward, central
from numerical_methods.differentiation import richardson
from numerical_methods.differentiation.derivative_function import derivative_function

import numpy as np

EXPECTED = 6

def quadratic(x):
    return x ** 2


def test_forward():
    assert abs(forward(quadratic, 3) - EXPECTED) < 1e-4


def test_backward():
    assert abs(backward(quadratic, 3) - EXPECTED) < 1e-4


def test_central():
    assert abs(central(quadratic, 3) - EXPECTED) < 1e-6


def test_richardson():
    assert abs(richardson.calculate(quadratic, 3, 1e-2) - EXPECTED) < 1e-10

def test_derivative_function_forward():
    df = derivative_function(quadratic, method="forward")
    assert abs(df(3) - EXPECTED) < 1e-4


def test_derivative_function_backward():
    df = derivative_function(quadratic, method="backward")
    assert abs(df(3) - EXPECTED) < 1e-4


def test_derivative_function_central():
    df = derivative_function(quadratic, method="central")
    assert abs(df(3) - EXPECTED) < 1e-6


def test_derivative_function_default_method_is_central():
    df = derivative_function(quadratic)
    assert abs(df(3) - EXPECTED) < 1e-6


def test_derivative_function_on_array():
    df = derivative_function(quadratic, method="central")
    x = np.array([1.0, 2.0, 3.0])
    expected = 2 * x
    assert np.allclose(df(x), expected, atol=1e-6)