import pytest
from numerical_methods.utilities.errors import error_calculate, objective_functions

# ---------- error_calculate ----------

def test_error_calculate_abs():
    result = error_calculate(vv=10.0, va=8.0, type="abs")
    expected = 2.0

    assert abs(result - expected) < 1e-9


def test_error_calculate_rel_abs():
    result = error_calculate(vv=10.0, va=8.0, type="rel_abs")
    expected = 0.2

    assert abs(result - expected) < 1e-9


def test_error_calculate_rel_frac_positive():
    result = error_calculate(vv=10.0, va=8.0, type="rel_frac")
    expected = 0.2

    assert abs(result - expected) < 1e-9


def test_error_calculate_rel_frac_negative():
    result = error_calculate(vv=8.0, va=10.0, type="rel_frac")
    expected = -0.25

    assert abs(result - expected) < 1e-9


def test_error_calculate_case_insensitive_type():
    result = error_calculate(vv=10.0, va=8.0, type="ABS")
    expected = 2.0

    assert abs(result - expected) < 1e-9


def test_error_calculate_rel_error_with_zero_reference_raises():
    with pytest.raises(ValueError):
        error_calculate(vv=0.0, va=5.0, type="rel_abs")


def test_error_calculate_invalid_type_raises():
    with pytest.raises(ValueError):
        error_calculate(vv=10.0, va=8.0, type="not_a_type")


# ---------- objective_functions ----------

def test_objective_functions_mse():
    Qsim = [1.0, 2.0, 3.0]
    Qobs = [1.5, 2.5, 2.5]

    result = objective_functions(Qsim, Qobs, N=3, type="mse")
    expected = ((0.5) ** 2 + (0.5) ** 2 + (0.5) ** 2) / 3

    assert abs(result - expected) < 1e-9


def test_objective_functions_rmse():
    Qsim = [1.0, 2.0, 3.0]
    Qobs = [1.5, 2.5, 2.5]

    result = objective_functions(Qsim, Qobs, N=3, type="rmse")
    expected_mse = ((0.5) ** 2 + (0.5) ** 2 + (0.5) ** 2) / 3
    expected = expected_mse ** 0.5

    assert abs(result - expected) < 1e-9


def test_objective_functions_mae():
    Qsim = [1.0, 2.0, 3.0]
    Qobs = [1.5, 2.5, 2.5]

    result = objective_functions(Qsim, Qobs, N=3, type="mae")
    expected = (0.5 + 0.5 + 0.5) / 3

    assert abs(result - expected) < 1e-9


def test_objective_functions_mbe():
    Qsim = [1.0, 2.0, 3.0]
    Qobs = [1.5, 2.5, 2.5]

    result = objective_functions(Qsim, Qobs, N=3, type="mbe")
    expected = ((-0.5) + (-0.5) + (0.5)) / 3

    assert abs(result - expected) < 1e-9


def test_objective_functions_invalid_type_raises():
    Qsim = [1.0, 2.0, 3.0]
    Qobs = [1.5, 2.5, 2.5]

    with pytest.raises(ValueError):
        objective_functions(Qsim, Qobs, N=3, type="not_a_metric")