from numerical_methods.ODE import euler, runge_kutta
import math

def decay(t, y):
    return -2 * y

expected = math.exp(-2)
error = 1e-2

n = 1000
y0 = 1
t0 = 0
tf = 1

def test_euler_explicit():
    _ , y = euler.explicit(decay, y0, t0, tf, n)
    assert abs(y[-1] - expected) < error


def test_rk2():
    _ , y = euler.explicit(decay, y0, t0, tf, n)
    assert abs(y[-1] - expected) < error


def test_rk4():
    _ , y = euler.explicit(decay, y0, t0, tf, n)
    assert abs(y[-1] - expected) < error