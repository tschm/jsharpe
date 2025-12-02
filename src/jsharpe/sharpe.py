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


def robust_covariance_inverse( V: np.ndarray ) -> np.ndarray:
    r"""
    Inverse of a constant-correlation covariance matrix, using the Sherman–Morrison formula

    Assume $V = \rho \sigma \sigma' + (1-\rho) \text{diag}(\sigma^2)$ (variance matrix, with constant correlations).
    Its inverse is $V^{-1} = A^{-1} - \dfrac{ A^{01} \rho \sigma \sigma' A^{-1} }{ 1 + \rho \sigma' A^{-1} \sigma }$.

    Input:
    - V: np.ndarray, variance matrix
    Output:
    - np.ndarray, inverse of the variance matrix
    """
    sigma = np.sqrt( np.diag(V) )
    C = (V.T/sigma).T/sigma
    rho = np.mean( C[ np.triu_indices_from(C,1) ] )
    A = np.diag( 1 / sigma**2 ) / (1-rho)
    sigma = sigma.reshape( -1, 1 )
    return A - ( rho * A @ sigma @ sigma.T @ A ) / ( 1 + rho * sigma.T @ A @ sigma )


def minimum_variance_weights_for_correlated_assets(V: np.ndarray) -> np.ndarray:
    """
    Weights of the minimum variance portfolio, for correlated assets, assuming a constant-correlation covariance matrix

    Input:
    - V: np.ndarray, variance matrix, shape (n,n)
    Output:
    - np.ndarray, weights of the minimum variance portfolio, shape (n,)
    """
    ones = np.ones( shape = V.shape[0] )
    S = robust_covariance_inverse(V)
    w = S @ ones
    w = w / np.sum(w)
    return w


if __name__ == "__main__":
    x = ppoints(20)
    print(x)
    print(len(x))

    y = np.linspace(0, 1, 10 + 2)
    print(y)
    print(len(y))
