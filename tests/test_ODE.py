from numerical_methods.ODE.euler import euler_explicit
import math

def decay(t, y):
    return -2 * y

expected = math.exp(-2)
error = 1e-2


def test_euler_explicit():
    t, y = euler_explicit(decay, 0, 1, 1, 1000)

    assert abs(y[-1] - expected) < error