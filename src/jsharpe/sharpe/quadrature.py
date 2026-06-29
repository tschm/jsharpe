"""Gaussian-expectation quadrature and order-statistic moments.

This module groups the Gauss-Hermite expectation helper and the moments
of the maximum of k standard normals, which together form the
expectation core relied upon by the Sharpe-ratio variance routines.
"""
# ruff: noqa: N802, N806

from collections.abc import Callable

import numpy as np
import scipy


def make_expectation_gh(
    n_nodes: int = 200,
) -> Callable[[Callable[[np.ndarray], np.ndarray]], float]:
    """Create an expectation function using Gauss-Hermite quadrature.

    Returns a function that computes E[g(Z)] where Z ~ N(0,1) using
    Gauss-Hermite quadrature with the specified number of nodes.

    The approximation is: E[g(Z)] ≈ (1/√π) Σ w_i g(√2 t_i)
    where (t_i, w_i) are Gauss-Hermite nodes and weights.

    Args:
        n_nodes: Number of quadrature nodes. Higher values increase accuracy.
            Default 200.

    Returns:
        Function E(g) that computes E[g(Z)] for Z ~ N(0,1).

    Example:
        >>> E = make_expectation_gh(n_nodes=100)
        >>> # E[Z^2] should be 1 for standard normal
        >>> bool(abs(E(lambda z: z**2) - 1.0) < 1e-6)
        True
        >>> # E[Z] should be 0 for standard normal
        >>> bool(abs(E(lambda z: z)) < 1e-10)
        True
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    scale = np.sqrt(2.0)
    norm = 1.0 / np.sqrt(np.pi)
    x = scale * nodes

    def E(g: Callable[[np.ndarray], np.ndarray]) -> float:
        """Compute E[g(Z)] for Z ~ N(0,1) via Gauss-Hermite quadrature.

        Args:
            g: Function to integrate against the standard normal density.

        Returns:
            Approximation of E[g(Z)].
        """
        vals = g(x)
        return float(norm * np.dot(weights, vals))

    return E


E_under_normal = make_expectation_gh(n_nodes=200)


def moments_Mk(k: int, *, rho: float = 0) -> tuple[float, float, float]:
    """Compute moments of the maximum of k standard normal random variables.

    Computes E[M_k], E[M_k^2], and Var[M_k] where M_k = max(Z_1, ..., Z_k)
    and Z_i are standard normal. Supports both independent (rho=0) and
    equi-correlated (rho > 0) cases.

    For the correlated case with equi-correlation rho:
        Z_i = sqrt(rho) * X + sqrt(1-rho) * Y_i
        M = sqrt(rho) * X + sqrt(1-rho) * max(Y_i)

    Args:
        k: Number of standard normal random variables.
        rho: Equi-correlation coefficient in [0, 1). Default is 0 (independent).

    Returns:
        Tuple of (Ez, Ez2, var):
            - Ez: Expected value E[M_k]
            - Ez2: Second moment E[M_k^2]
            - var: Variance Var[M_k]

    Example:
        >>> Ez, Ez2, var = moments_Mk(1)
        >>> bool(abs(Ez) < 1e-10)  # E[max of one N(0,1)] = 0
        True
        >>> bool(abs(var - 1.0) < 1e-10)  # Var[max of one N(0,1)] = 1
        True
    """
    Phi = scipy.stats.norm.cdf
    Ez = E_under_normal(lambda z: k * z * (Phi(z) ** (k - 1)))
    Ez2 = E_under_normal(lambda z: k * z**2 * (Phi(z) ** (k - 1)))
    var = Ez2 - Ez**2

    Ez = (1 - rho) * Ez
    var = rho + (1 - rho) * var
    Ez2 = var + Ez**2

    return Ez, Ez2, var
