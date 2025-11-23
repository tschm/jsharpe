"""Sharpe-related utilities, including probability points generation (ppoints)."""

import numpy as np


def ppoints(n, a=None):
    """Equidistant points in [0,1], to be used as arguments of the pdf or icdf of distributions.
    Boundaries are excluded.
    See the documentation of the corresponding R function for more details.

    Inputs: n: integer, desired number of points
            a: offset
    Output: numpy array, with n equidistant points

    Example:
        ppoints(20)
    """
    if a is None:
        a = 0.5 if n > 10 else 3 / 8
    assert 0 <= a <= 1, f"the offset should be in [0,1], got {a}"
    return np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)


if __name__ == "__main__":
    x = ppoints(20)
    print(x)
    print(len(x))

    y = np.linspace(0, 1, 10 + 2)
    print(y)
    print(len(y))
