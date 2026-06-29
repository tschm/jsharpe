"""Linear-algebra and probability-plotting helpers.

This module groups low-level numerical utilities used throughout the
package: probability plotting positions and constant-correlation
covariance routines (inversion and minimum-variance weights).
"""
# ruff: noqa: N803, N806, TRY003

import numpy as np


def ppoints(n: int, a: float | None = None) -> np.ndarray:
    """Generate probability points for Q-Q plots and distribution fitting.

    Creates n equidistant points in the interval (0, 1), suitable for
    probability plotting positions. The boundaries 0 and 1 are excluded.
    This function mirrors the behavior of R's `ppoints()` function.

    Args:
        n: Number of probability points to generate.
        a: Plotting position parameter in [0, 1]. Defaults to 0.5 if n > 10,
            otherwise 3/8 (following R's convention).

    Returns:
        Array of n equidistant probability points in (0, 1).

    Raises:
        ValueError: If a is not in [0, 1].

    Example:
        >>> import numpy as np
        >>> pts = ppoints(5)
        >>> len(pts)
        5
        >>> np.all((pts > 0) & (pts < 1))
        np.True_
        >>> ppoints(5, a=0.5)
        array([0.1, 0.3, 0.5, 0.7, 0.9])
    """
    if a is None:
        a = 0.5 if n > 10 else 3 / 8
    if not (0 <= a <= 1):
        raise ValueError(f"the offset should be in [0,1], got {a}")
    return np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)


def robust_covariance_inverse(V: np.ndarray) -> np.ndarray:
    r"""Compute inverse of a constant-correlation covariance matrix.

    Uses the Sherman-Morrison formula for efficient computation.
    Assumes the variance matrix has the form:
    $V = \rho \sigma \sigma' + (1-\rho) \text{diag}(\sigma^2)$
    (constant correlations across all pairs).

    Its inverse is computed as:
    $V^{-1} = A^{-1} - \dfrac{ A^{-1} \rho \sigma \sigma' A^{-1} }
    { 1 + \rho \sigma' A^{-1} \sigma }$

    Args:
        V: Covariance matrix with constant off-diagonal correlations.
            Shape (n, n).

    Returns:
        Inverse of the covariance matrix. Shape (n, n).

    Example:
        >>> import numpy as np
        >>> # Create a constant-correlation covariance matrix
        >>> rho = 0.5
        >>> sigma = np.array([1.0, 2.0, 1.5])
        >>> C = rho * np.ones((3, 3))
        >>> np.fill_diagonal(C, 1)
        >>> V = (C * sigma.reshape(-1, 1)).T * sigma.reshape(-1, 1)
        >>> V_inv = robust_covariance_inverse(V)
        >>> np.allclose(V @ V_inv, np.eye(3), atol=1e-10)
        True
    """
    sigma = np.sqrt(np.diag(V))
    C = (V.T / sigma).T / sigma
    rho = np.mean(C[np.triu_indices_from(C, 1)])
    A = np.diag(1 / sigma**2) / (1 - rho)
    sigma = sigma.reshape(-1, 1)
    result: np.ndarray = A - (rho * A @ sigma @ sigma.T @ A) / (1 + rho * sigma.T @ A @ sigma)
    return result


def minimum_variance_weights_for_correlated_assets(V: np.ndarray) -> np.ndarray:
    """Compute weights of the minimum variance portfolio for correlated assets.

    Computes the portfolio weights that minimize portfolio variance subject
    to the constraint that weights sum to 1. Assumes a constant-correlation
    covariance structure for efficient computation.

    Args:
        V: Covariance matrix of asset returns. Shape (n, n).

    Returns:
        Portfolio weights that minimize variance. Shape (n,).
        Weights sum to 1.

    Example:
        >>> import numpy as np
        >>> # Create a simple covariance matrix
        >>> V = np.array([[0.04, 0.01], [0.01, 0.09]])
        >>> w = minimum_variance_weights_for_correlated_assets(V)
        >>> np.isclose(w.sum(), 1.0)
        np.True_
    """
    ones = np.ones(shape=V.shape[0])
    S = robust_covariance_inverse(V)
    w = S @ ones
    w = w / np.sum(w)
    return w
