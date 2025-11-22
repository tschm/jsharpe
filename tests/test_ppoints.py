import numpy as np
import pytest

from jsharpe.sharpe import ppoints


def test_ppoints_default_large_n():
    n = 20
    # default a = 0.5 when n > 10
    expected = np.linspace(1 - 0.5, n - 0.5, n) / (n + 1 - 2 * 0.5)
    x = ppoints(n)
    assert np.allclose(x, expected)
    assert np.all(x > 0) and np.all(x < 1)
    # uniform spacing
    diffs = np.diff(x)
    assert np.allclose(diffs, diffs[0])


def test_ppoints_default_small_n():
    n = 10
    # default a = 3/8 when n <= 10
    a = 3 / 8
    expected = np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)
    x = ppoints(n)
    assert np.allclose(x, expected)
    # uniform spacing
    diffs = np.diff(x)
    assert np.allclose(diffs, diffs[0])


def test_ppoints_custom_a_zero():
    n = 5
    a = 0.0
    expected = np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)
    x = ppoints(n, a=a)
    assert np.allclose(x, expected)
    # should exclude 0 and 1 for a=0
    assert x[0] == pytest.approx(1 / (n + 1))
    assert x[-1] == pytest.approx(n / (n + 1))


def test_ppoints_custom_a_one_includes_boundaries():
    n = 5
    a = 1.0
    # This includes both boundaries 0 and 1 by design of the formula
    expected = np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)
    x = ppoints(n, a=a)
    assert np.allclose(x, expected)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)


def test_ppoints_invalid_a_raises():
    n = 7
    with pytest.raises(AssertionError):
        ppoints(n, a=-0.01)
    with pytest.raises(AssertionError):
        ppoints(n, a=1.01)
