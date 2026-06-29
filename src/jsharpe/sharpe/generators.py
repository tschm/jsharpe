"""Synthetic return-data generators and autocorrelation estimation.

This module groups the simulation helpers used to generate
(autocorrelated) non-Gaussian return series and block-structured random
correlation matrices, plus the mean first-order autocorrelation estimator.
"""
# ruff: noqa: N802, N803, N806, S101

import numpy as np
import scipy

from .linalg import ppoints


def generate_autocorrelated_non_gaussian_data(
    N: int,
    n: int,
    SR0: float = 0,
    name: str = "gaussian",
    rho: float | None = None,
    gaussian_autocorrelation: float = 0,
) -> np.ndarray:
    """Generate autocorrelated non-Gaussian return data for simulation.

    Creates a matrix of simulated returns with specified autocorrelation
    and marginal distribution characteristics (skewness/kurtosis).

    Uses a copula-like approach:
        1. Generate AR(1) Gaussian processes
        2. Transform to uniform via Gaussian CDF
        3. Transform to target marginals via inverse CDF

    Args:
        N: Number of time periods (rows).
        n: Number of assets/strategies (columns).
        SR0: Target Sharpe ratio. Default 0.
        name: Distribution type. One of "gaussian", "mild", "moderate",
            "severe". Default "gaussian".
        rho: Autocorrelation coefficient. If None, uses gaussian_autocorrelation.
        gaussian_autocorrelation: Autocorrelation for Gaussian case. Default 0.

    Returns:
        Array of shape (N, n) containing simulated returns.

    Example:
        >>> np.random.seed(42)
        >>> X = generate_autocorrelated_non_gaussian_data(100, 2, SR0=0.1, name="mild")
        >>> X.shape
        (100, 2)
    """
    if rho is None:
        # With the distributions we consider the autocorrelation is almost the same.
        rho = gaussian_autocorrelation

    shape = (N, n)

    # Marginal distribution: ppf
    R = 10_000
    marginal = generate_non_gaussian_data(R, 1, SR0=SR0, name=name)[:, 0]
    ppf = scipy.interpolate.interp1d(ppoints(R), sorted(marginal), fill_value="extrapolate")

    # AR(1) processes
    X = np.random.normal(size=shape)
    for i in range(1, shape[0]):
        X[i, :] = rho * X[i - 1, :] + np.sqrt(1 - rho**2) * X[i, :]

    # Convert the margins to uniform, with the Gaussian cdf
    X = scipy.stats.norm.cdf(X)

    # Convert the uniforms to the target margins, using the ppf
    result: np.ndarray = ppf(X)

    return result


def get_random_correlation_matrix(
    number_of_trials: int = 100,
    effective_number_of_trials: int = 10,
    number_of_observations: int = 200,
    noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random correlation matrix with block structure.

    Creates a correlation matrix representing clustered strategies, where
    strategies within the same cluster are highly correlated and strategies
    across clusters have lower correlation.

    Args:
        number_of_trials: Number of time series (strategies). Default 100.
        effective_number_of_trials: Number of clusters. Default 10.
        number_of_observations: Number of time periods to simulate. Default 200.
        noise: Noise level added to each series. Default 0.1.

    Returns:
        Tuple of (C, X, clusters):
            - C: Correlation matrix of shape (number_of_trials, number_of_trials)
            - X: Data matrix of shape (number_of_observations, number_of_trials)
            - clusters: Cluster assignment for each strategy

    Example:
        >>> np.random.seed(42)
        >>> C, X, clusters = get_random_correlation_matrix(
        ...     number_of_trials=20, effective_number_of_trials=4
        ... )
        >>> C.shape
        (20, 20)
        >>> np.allclose(np.diag(C), 1)  # Diagonal is all ones
        True
    """
    while True:
        block_positions = [
            0,
            *sorted(np.random.choice(number_of_trials, effective_number_of_trials - 1, replace=True)),
            number_of_trials,
        ]
        block_sizes = np.diff(block_positions)
        if np.all(block_sizes > 0):
            break

    clusters = np.array([block_number for block_number, size in enumerate(block_sizes) for _ in range(size)])
    X0 = np.random.normal(size=(number_of_observations, effective_number_of_trials))
    X = np.zeros(shape=(number_of_observations, number_of_trials))
    for i, cluster in enumerate(clusters):
        X[:, i] = X0[:, cluster] + noise * np.random.normal(size=number_of_observations)
    C = np.asarray(np.corrcoef(X, rowvar=False))
    np.fill_diagonal(C, 1)  # rounding errors
    C = np.clip(C, -1, 1)
    return C, X, clusters


def generate_non_gaussian_data(
    nr: int,
    nc: int,
    *,
    SR0: float = 0,
    name: str = "severe",
) -> np.ndarray:
    """Generate non-Gaussian return data with specified characteristics.

    Creates a matrix of simulated returns from a mixture distribution that
    exhibits the specified skewness and kurtosis characteristics while
    maintaining the target Sharpe ratio.

    Args:
        nr: Number of rows (observations/time periods).
        nc: Number of columns (assets/strategies).
        SR0: Target Sharpe ratio. Default 0.
        name: Distribution severity. One of:
            - "gaussian": No skewness or kurtosis
            - "mild": Slight negative skew and excess kurtosis
            - "moderate": Moderate negative skew and excess kurtosis
            - "severe": Strong negative skew and excess kurtosis
            Default "severe".

    Returns:
        Array of shape (nr, nc) containing simulated returns.

    Raises:
        AssertionError: If name is not a valid distribution type.

    Example:
        >>> np.random.seed(42)
        >>> X = generate_non_gaussian_data(1000, 1, SR0=0.2, name="mild")
        >>> X.shape
        (1000, 1)
    """
    configs = {
        "gaussian": (0, 0, 0.015, 0.010),
        "mild": (0.04, -0.03, 0.015, 0.010),
        "moderate": (0.03, -0.045, 0.020, 0.010),
        "severe": (0.02, -0.060, 0.025, 0.010),
    }
    assert name in configs

    def mixture_variance(
        p_tail: float,
        mu_tail: float,
        sigma_tail: float,
        mu_core: float,
        sigma_core: float,
    ) -> float:
        """Compute the variance of a two-component Gaussian mixture.

        Args:
            p_tail: Mixing weight of the tail component.
            mu_tail: Mean of the tail component.
            sigma_tail: Standard deviation of the tail component.
            mu_core: Mean of the core component.
            sigma_core: Standard deviation of the core component.

        Returns:
            Variance of the mixture distribution.
        """
        w = 1.0 - p_tail
        mu = w * mu_core + p_tail * mu_tail
        m2 = w * (sigma_core**2 + mu_core**2) + p_tail * (sigma_tail**2 + mu_tail**2)
        return float(m2 - mu**2)

    def gen_with_true_SR0(reps: int, T: int, cfg: tuple[float, float, float, float], SR0: float) -> np.ndarray:
        """Generate mixture returns scaled to a target population Sharpe ratio.

        Args:
            reps: Number of independent return series to generate.
            T: Length of each return series.
            cfg: Mixture config tuple (p_tail, mu_tail, sigma_tail, sigma_core).
            SR0: Target population Sharpe ratio.

        Returns:
            Array of shape (reps, T) with non-Gaussian returns at the given Sharpe ratio.
        """
        p, mu_tail, sig_tail, sig_core = cfg
        # Zero-mean baseline mixture (choose mu_core so mean=0)
        mu_core0 = -p * mu_tail / (1.0 - p)
        std0 = np.sqrt(mixture_variance(p, mu_tail, sig_tail, mu_core0, sig_core))
        mu_shift = SR0 * std0  # sets population Sharpe to SR0, preserves skew/kurt
        mask = np.random.uniform(size=(reps, T)) < p
        X = np.random.normal(mu_core0 + mu_shift, sig_core, size=(reps, T))
        X[mask] = np.random.normal(mu_tail + mu_shift, sig_tail, size=mask.sum())
        return X

    return gen_with_true_SR0(nr, nc, configs[name], SR0)


def autocorrelation(X: np.ndarray) -> float:
    """Compute mean first-order autocorrelation across columns.

    Calculates the lag-1 autocorrelation for each column of the input
    matrix and returns the mean across all columns.

    Args:
        X: Data matrix of shape (n_observations, n_series).

    Returns:
        Mean autocorrelation coefficient across all columns.

    Example:
        >>> np.random.seed(42)
        >>> # Generate AR(1) process with rho=0.5
        >>> n = 1000
        >>> X = np.zeros((n, 1))
        >>> X[0] = np.random.normal()
        >>> for i in range(1, n):
        ...     X[i] = 0.5 * X[i-1] + np.sqrt(1-0.25) * np.random.normal()
        >>> ac = autocorrelation(X)
        >>> bool(0.4 < ac < 0.6)  # Should be close to 0.5
        True
    """
    _nr, nc = X.shape
    ac = np.zeros(nc)
    for i in range(nc):
        ac[i] = np.corrcoef(X[1:, i], X[:-1, i])[0, 1]
    return float(ac.mean())
