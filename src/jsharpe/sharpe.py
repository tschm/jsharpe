"""Sharpe-related utilities for statistical analysis and hypothesis testing.

This module provides comprehensive tools for Sharpe ratio analysis, including:
    - Variance estimation under non-Gaussian returns
    - Statistical significance testing
    - Multiple testing corrections (FDR, FWER)
    - Portfolio optimization utilities
"""

import math
import warnings
from collections.abc import Callable

import numpy as np
import scipy  # type: ignore[import-untyped]


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
        AssertionError: If a is not in [0, 1].

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
    assert 0 <= a <= 1, f"the offset should be in [0,1], got {a}"
    return np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)


def robust_covariance_inverse(V: np.ndarray) -> np.ndarray:
    r"""Compute inverse of a constant-correlation covariance matrix.

    Uses the Sherman–Morrison formula for efficient computation.
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
    result: np.ndarray = A - (rho * A @ sigma @ sigma.T @ A) / (
        1 + rho * sigma.T @ A @ sigma
    )
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


def effective_rank(C: np.ndarray) -> float:
    """Compute the effective rank of a positive semi-definite matrix.

    The effective rank measures the "effective dimensionality" of a matrix
    by computing the exponential of the entropy of its normalized eigenvalues.
    This provides a continuous measure between 1 (perfectly correlated) and
    n (perfectly uncorrelated/identity matrix).

    Algorithm:
        1. Compute eigenvalues (non-negative for PSD matrices)
        2. Discard zero eigenvalues
        3. Normalize to form a probability distribution
        4. Compute entropy H = -Σ p_i log(p_i)
        5. Return exp(H)

    Args:
        C: Positive semi-definite matrix (e.g., correlation matrix).
            Shape (n, n).

    Returns:
        Effective rank, a value in [1, n] where n is the matrix dimension.

    References:
        Roy, O. and Vetterli, M. (2007). "The effective rank: a measure of
        effective dimensionality." EURASIP Journal on Advances in Signal
        Processing. http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.177.2721

    Example:
        >>> import numpy as np
        >>> # Identity matrix has effective rank equal to its dimension
        >>> abs(effective_rank(np.eye(3)) - 3.0) < 1e-10
        True
        >>> # Perfectly correlated matrix has effective rank 1
        >>> C = np.ones((3, 3))
        >>> abs(effective_rank(C) - 1.0) < 1e-10
        True
    """
    p = np.linalg.eigvalsh(C)
    p = p[p > 0]
    p = p / sum(p)
    H = np.sum(-p * np.log(p))
    return math.exp(H)


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


def sharpe_ratio_variance(
    SR: float,
    T: int,
    *,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    K: int = 1,
) -> float:
    """Compute the asymptotic variance of the Sharpe ratio estimator.

    Accounts for non-Gaussian returns (skewness, kurtosis) and autocorrelation.
    Under Gaussian iid returns, reduces to (1 + SR^2/2) / T.

    Args:
        SR: Sharpe ratio value (annualized or per-period).
        T: Number of observations (time periods).
        gamma3: Skewness of returns. Default 0 (symmetric).
        gamma4: Kurtosis of returns (non-excess). Default 3 (Gaussian).
        rho: First-order autocorrelation of returns. Default 0 (iid).
        K: Number of strategies for multiple testing adjustment. Default 1.

    Returns:
        Variance of the Sharpe ratio estimator.

    Example:
        >>> # Variance under Gaussian iid assumptions
        >>> var_gaussian = sharpe_ratio_variance(SR=0.5, T=24)
        >>> # Variance with non-Gaussian returns (higher kurtosis)
        >>> var_nongauss = sharpe_ratio_variance(SR=0.5, T=24, gamma4=6.0)
        >>> bool(var_nongauss > var_gaussian)  # Higher kurtosis increases variance
        True
    """
    A = 1
    B = rho / (1 - rho)
    C = rho**2 / (1 - rho**2)
    a = A + 2 * B
    b = A + B + C
    c = A + 2 * C
    V = (a * 1 - b * gamma3 * SR + c * (gamma4 - 1) / 4 * SR**2) / T
    return float(V * moments_Mk(K)[2])


def variance_of_the_maximum_of_k_Sharpe_ratios(
    number_of_trials: int, variance: float
) -> float:
    """Compute the variance of the maximum Sharpe ratio across K strategies.

    Selection across a larger pool increases the uncertainty of the selected
    (maximum) Sharpe ratio estimate. This function applies a logarithmic
    variance inflation factor that increases monotonically with K.

    Args:
        number_of_trials: Number of strategies (K >= 1).
        variance: Base variance of individual Sharpe ratio estimates.

    Returns:
        Inflated variance accounting for selection from K strategies.

    Example:
        >>> # More trials increases variance due to selection bias
        >>> v1 = variance_of_the_maximum_of_k_Sharpe_ratios(1, 0.1)
        >>> v10 = variance_of_the_maximum_of_k_Sharpe_ratios(10, 0.1)
        >>> bool(v10 > v1)
        True
    """
    # Monotone increasing variance inflation with K (K>=1)
    inflation = 1.0 + np.log(max(1, int(number_of_trials)))
    return float(variance * inflation)


def control_for_FDR(
    q: float,
    *,
    SR0: float = 0,
    SR1: float = 0.5,
    p_H1: float = 0.05,
    T: int = 24,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    K: int = 1,
) -> tuple[float, float, float, float]:
    """Compute critical value to test multiple Sharpe ratios controlling FDR.

    Determines the critical Sharpe ratio threshold and associated error rates
    to control the False Discovery Rate at level q when testing multiple
    strategies.

    Args:
        q: Desired False Discovery Rate level in (0, 1).
        SR0: Sharpe ratio under null hypothesis H0. Default 0.
        SR1: Sharpe ratio under alternative hypothesis H1. Default 0.5.
        p_H1: Prior probability that H1 is true. Default 0.05.
        T: Number of observations. Default 24.
        gamma3: Skewness of returns. Default 0.
        gamma4: Kurtosis of returns (non-excess). Default 3 (Gaussian).
        rho: Autocorrelation of returns. Default 0.
        K: Number of strategies (K=1 for FDR; K>1 for FWER-FDR). Default 1.

    Returns:
        Tuple of (alpha, beta, SR_c, q_hat):
            - alpha: Significance level P[SR > SR_c | H0]
            - beta: Type II error P[SR <= SR_c | H1]; power is 1 - beta
            - SR_c: Critical Sharpe ratio threshold
            - q_hat: Estimated FDR (should be close to q)

    Example:
        >>> alpha, beta, SR_c, q_hat = control_for_FDR(q=0.25, T=24)
        >>> bool(0 < alpha < 1)
        True
        >>> bool(SR_c > 0)  # Critical value is positive
        True
    """
    Z = scipy.stats.norm.cdf

    s0 = math.sqrt(
        sharpe_ratio_variance(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    )
    s1 = math.sqrt(
        sharpe_ratio_variance(SR1, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    )
    SRc = FDR_critical_value(q, SR0, SR1, s0, s1, p_H1)

    beta = Z((SRc - SR1) / s1)
    alpha = q / (1 - q) * p_H1 / (1 - p_H1) * (1 - beta)
    q_hat = 1 / (1 + (1 - beta) / alpha * p_H1 / (1 - p_H1))

    return alpha, beta, SRc, q_hat


def expected_maximum_sharpe_ratio(
    number_of_trials: int, variance: float, SR0: float = 0
) -> float:
    """Compute expected maximum Sharpe ratio under multiple testing.

    Estimates E[max(SR_1, ..., SR_K)] assuming K independent Sharpe ratio
    estimates, each with the same variance. Uses the Gumbel approximation
    for the expected maximum of normals.

    Args:
        number_of_trials: Number of strategies (K).
        variance: Variance of individual Sharpe ratio estimates.
        SR0: Baseline Sharpe ratio to add. Default 0.

    Returns:
        Expected value of the maximum Sharpe ratio across K strategies.

    Example:
        >>> # Expected maximum increases with number of trials
        >>> e1 = expected_maximum_sharpe_ratio(1, 0.1)
        >>> e10 = expected_maximum_sharpe_ratio(10, 0.1)
        >>> bool(e10 > e1)
        True
    """
    return float(
        SR0
        + (
            np.sqrt(variance)
            * (
                (1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / number_of_trials)
                + np.euler_gamma
                * scipy.stats.norm.ppf(1 - 1 / number_of_trials / np.exp(1))
            )
        )
    )


def minimum_track_record_length(
    SR: float,
    SR0: float,
    *,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Compute minimum track record length for statistical significance.

    Determines the minimum number of observations T required for the observed
    Sharpe ratio SR to be significantly greater than SR0 at confidence level
    1 - alpha.

    Args:
        SR: Observed Sharpe ratio.
        SR0: Sharpe ratio under null hypothesis H0.
        gamma3: Skewness of returns. Default 0.
        gamma4: Kurtosis of returns (non-excess). Default 3 (Gaussian).
        rho: Autocorrelation of returns. Default 0.
        alpha: Significance level. Default 0.05.

    Returns:
        Minimum track record length (number of periods) required.

    Example:
        >>> # Higher Sharpe ratio needs shorter track record
        >>> mtrl_high = minimum_track_record_length(SR=1.0, SR0=0)
        >>> mtrl_low = minimum_track_record_length(SR=0.5, SR0=0)
        >>> bool(mtrl_high < mtrl_low)
        True
    """
    var = sharpe_ratio_variance(SR0, T=1, gamma3=gamma3, gamma4=gamma4, rho=rho, K=1)
    return float(var * (scipy.stats.norm.ppf(1 - alpha) / (SR - SR0)) ** 2)


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
        vals = g(x)
        return float(norm * np.dot(weights, vals))

    return E


E_under_normal = make_expectation_gh(n_nodes=200)


def adjusted_p_values_bonferroni(ps: np.ndarray) -> np.ndarray:
    """Adjust p-values using Bonferroni correction for FWER control.

    Multiplies each p-value by the number of tests M, capping at 1.
    This is the most conservative multiple testing correction.

    Args:
        ps: Array of unadjusted p-values.

    Returns:
        Array of Bonferroni-adjusted p-values, each in [0, 1].

    Example:
        >>> p_vals = np.array([0.01, 0.03, 0.05])
        >>> adj_p = adjusted_p_values_bonferroni(p_vals)
        >>> np.allclose(adj_p, [0.03, 0.09, 0.15])
        True
    """
    M = len(ps)
    result: np.ndarray = np.minimum(1, M * ps)
    return result


def adjusted_p_values_sidak(ps: np.ndarray) -> np.ndarray:
    """Adjust p-values using Šidák correction for FWER control.

    Uses the formula: 1 - (1 - p)^M, which is slightly less conservative
    than Bonferroni when tests are independent.

    Args:
        ps: Array of unadjusted p-values.

    Returns:
        Array of Šidák-adjusted p-values, each in [0, 1].

    Example:
        >>> p_vals = np.array([0.01, 0.03, 0.05])
        >>> adj_p = adjusted_p_values_sidak(p_vals)
        >>> np.all(adj_p <= adjusted_p_values_bonferroni(p_vals))
        np.True_
    """
    M = len(ps)
    return 1 - (1 - ps) ** M


def adjusted_p_values_holm(
    ps: np.ndarray, *, variant: str = "bonferroni"
) -> np.ndarray:
    """Adjust p-values using Holm's step-down procedure for FWER control.

    A step-down procedure that is uniformly more powerful than Bonferroni
    while still controlling the Family-Wise Error Rate.

    Args:
        ps: Array of unadjusted p-values.
        variant: Correction method for each step. Either "bonferroni"
            (default) or "sidak".

    Returns:
        Array of Holm-adjusted p-values, each in [0, 1].

    Raises:
        AssertionError: If variant is not "bonferroni" or "sidak".

    Example:
        >>> p_vals = np.array([0.01, 0.04, 0.03])
        >>> adj_p = adjusted_p_values_holm(p_vals)
        >>> float(adj_p[0])  # Smallest p-value adjusted most
        0.03
    """
    assert variant in ["bonferroni", "sidak"]
    i = np.argsort(ps)
    M = len(ps)
    p_adjusted = np.zeros(M)
    previous = 0
    for j, idx in enumerate(i):
        if variant == "bonferroni":
            candidate = min(1, ps[idx] * (M - j))
        else:
            candidate = 1 - (1 - ps[idx]) ** (M - j)
        p_adjusted[idx] = max(previous, candidate)
        previous = p_adjusted[idx]
    return p_adjusted


def FDR_critical_value(
    q: float, SR0: float, SR1: float, sigma0: float, sigma1: float, p_H1: float
) -> float:
    """Compute critical value for FDR control in hypothesis testing.

    Given a mixture model where H ~ Bernoulli(p_H1) determines whether
    X follows N(SR0, sigma0^2) or N(SR1, sigma1^2), finds the critical
    value c such that P[H=0 | X > c] = q.

    Args:
        q: Desired False Discovery Rate in (0, 1).
        SR0: Mean under null hypothesis (must be < SR1).
        SR1: Mean under alternative hypothesis.
        sigma0: Standard deviation under null hypothesis (must be > 0).
        sigma1: Standard deviation under alternative hypothesis (must be > 0).
        p_H1: Prior probability of alternative hypothesis in (0, 1).

    Returns:
        Critical value c. Returns -inf if solution is outside [-10, 10],
        or nan if no solution exists.

    Raises:
        AssertionError: If parameters are out of valid ranges.

    Example:
        >>> c = FDR_critical_value(q=0.2, SR0=0, SR1=0.5, sigma0=0.2, sigma1=0.3, p_H1=0.1)
        >>> c > 0  # Critical value should be positive
        True
    """
    assert SR0 < SR1
    assert 0 < q < 1
    assert 0 < p_H1 < 1
    assert 0 < sigma0
    assert 0 < sigma1

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in scalar divide"
        )
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in scalar divide"
        )

        def f(c: float) -> float:
            a = 1 / (
                1
                + scipy.stats.norm.sf((c - SR1) / sigma1)
                / scipy.stats.norm.sf((c - SR0) / sigma0)
                * p_H1
                / (1 - p_H1)
            )
            return float(np.where(np.isfinite(a), a, 0))

        if f(-10) < q:  # Solution outside of the search interval
            return float(-np.inf)

        if (f(-10) - q) * (
            f(10) - q
        ) > 0:  # No solution, for instance if σ₀≫σ₁ and q small
            return float(np.nan)

        return float(scipy.optimize.brentq(lambda c: f(c) - q, -10, 10))


def critical_sharpe_ratio(
    SR0: float,
    T: int,
    *,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    alpha: float = 0.05,
    K: int = 1,
) -> float:
    """Compute critical Sharpe ratio for hypothesis testing.

    Determines the threshold SR_c for the one-sided test:
        H0: SR = SR0  vs  H1: SR > SR0
    at significance level alpha.

    Args:
        SR0: Sharpe ratio under null hypothesis.
        T: Number of observations.
        gamma3: Skewness of returns. Default 0.
        gamma4: Kurtosis of returns (non-excess). Default 3 (Gaussian).
        rho: Autocorrelation of returns. Default 0.
        alpha: Significance level. Default 0.05.
        K: Number of strategies for variance adjustment. Default 1.

    Returns:
        Critical Sharpe ratio threshold. Reject H0 if observed SR > SR_c.

    Example:
        >>> SR_c = critical_sharpe_ratio(SR0=0, T=24, alpha=0.05)
        >>> bool(SR_c > 0)  # Need positive SR to reject H0: SR=0
        True
    """
    variance = sharpe_ratio_variance(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    return float(SR0 + scipy.stats.norm.ppf(1 - alpha) * math.sqrt(variance))


def probabilistic_sharpe_ratio(
    SR: float,
    SR0: float,
    *,
    variance: float | None = None,
    T: int | None = None,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    K: int = 1,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR).

    The PSR is 1 - p, where p is the p-value of testing H0: SR = SR0 vs
    H1: SR > SR0. It can be interpreted as a "Sharpe ratio on a probability
    scale", i.e., mapping the SR to [0, 1].

    Args:
        SR: Observed Sharpe ratio.
        SR0: Sharpe ratio under null hypothesis.
        variance: Variance of SR estimator. Provide this OR (T, gamma3, ...).
        T: Number of observations (if variance not provided).
        gamma3: Skewness of returns. Default 0.
        gamma4: Kurtosis of returns (non-excess). Default 3 (Gaussian).
        rho: Autocorrelation of returns. Default 0.
        K: Number of strategies for variance adjustment. Default 1.

    Returns:
        Probabilistic Sharpe Ratio in [0, 1]. Values near 1 indicate
        strong evidence that the true SR exceeds SR0.

    Raises:
        AssertionError: If both variance and T are provided.

    Example:
        >>> psr = probabilistic_sharpe_ratio(SR=0.5, SR0=0, T=24)
        >>> bool(0 < psr < 1)
        True
        >>> # Higher observed SR gives higher PSR
        >>> psr_high = probabilistic_sharpe_ratio(SR=1.0, SR0=0, T=24)
        >>> bool(psr_high > psr)
        True
    """
    if variance is None:
        assert T is not None, "T must be provided if variance is not provided"
        variance = sharpe_ratio_variance(
            SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K
        )
    else:
        assert T is None, "Provide either the variance or (T, gamma3, gamma4, rho)"
    return float(scipy.stats.norm.cdf((SR - SR0) / math.sqrt(variance)))


def sharpe_ratio_power(
    SR0: float,
    SR1: float,
    T: int,
    *,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    alpha: float = 0.05,
    K: int = 1,
) -> float:
    """Compute statistical power for Sharpe ratio hypothesis test.

    Calculates the power (1 - β) of the test H0: SR = SR0 vs H1: SR = SR1,
    which is the probability of correctly rejecting H0 when the true
    Sharpe ratio is SR1.

    Note: Power is equivalent to recall in classification:
        Power = P[reject H0 | H1] = TP / (TP + FN)

    Args:
        SR0: Sharpe ratio under null hypothesis.
        SR1: Sharpe ratio under alternative hypothesis (should be > SR0).
        T: Number of observations.
        gamma3: Skewness of returns. Default 0.
        gamma4: Kurtosis of returns (non-excess). Default 3 (Gaussian).
        rho: Autocorrelation of returns. Default 0.
        alpha: Significance level. Default 0.05.
        K: Number of strategies for variance adjustment. Default 1.

    Returns:
        Statistical power in [0, 1].

    Example:
        >>> # More observations increases power
        >>> power_short = sharpe_ratio_power(SR0=0, SR1=0.5, T=12)
        >>> power_long = sharpe_ratio_power(SR0=0, SR1=0.5, T=48)
        >>> bool(power_long > power_short)
        True
    """
    critical_SR = critical_sharpe_ratio(
        SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, alpha=alpha
    )
    variance = sharpe_ratio_variance(SR1, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    beta = scipy.stats.norm.cdf((critical_SR - SR1) / math.sqrt(variance))
    return float(1 - beta)


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
    ppf = scipy.interpolate.interp1d(
        ppoints(R), sorted(marginal), fill_value="extrapolate"
    )

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
        block_positions = (
            [0]
            + sorted(
                np.random.choice(
                    number_of_trials, effective_number_of_trials - 1, replace=True
                )
            )
            + [number_of_trials]
        )
        block_sizes = np.diff(block_positions)
        if np.all(block_sizes > 0):
            break

    clusters = np.array(
        [
            block_number
            for block_number, size in enumerate(block_sizes)
            for _ in range(size)
        ]
    )
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
        w = 1.0 - p_tail
        mu = w * mu_core + p_tail * mu_tail
        m2 = w * (sigma_core**2 + mu_core**2) + p_tail * (sigma_tail**2 + mu_tail**2)
        return float(m2 - mu**2)

    def gen_with_true_SR0(
        reps: int, T: int, cfg: tuple[float, float, float, float], SR0: float
    ) -> np.ndarray:
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
    nr, nc = X.shape
    ac = np.zeros(nc)
    for i in range(nc):
        ac[i] = np.corrcoef(X[1:, i], X[:-1, i])[0, 1]
    return float(ac.mean())


def pFDR(
    p_H1: float,
    alpha: float,
    beta: float,
) -> float:
    """Compute posterior FDR given test outcome exceeds critical value.

    Calculates P[H0 | SR > SR_c], the probability that the null hypothesis
    is true given that the observed Sharpe ratio exceeds the critical value.
    This is the "predictive" FDR based on the critical value, not the
    observed value.

    Args:
        p_H1: Prior probability that H1 is true.
        alpha: Significance level (Type I error rate).
        beta: Type II error rate (1 - power).

    Returns:
        Posterior probability of H0 given rejection.

    Example:
        >>> # With 5% prior on H1 and 5% significance
        >>> fdr = pFDR(p_H1=0.05, alpha=0.05, beta=0.3)
        >>> 0 < fdr < 1
        True
    """
    p_H0 = 1 - p_H1
    return 1 / (1 + (1 - beta) * p_H1 / alpha / p_H0)


def oFDR(
    SR: float,
    SR0: float,
    SR1: float,
    T: int,
    p_H1: float,
    *,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    K: int = 1,
) -> float:
    """Compute observed FDR given the observed Sharpe ratio.

    Calculates P[H0 | SR > SR_obs], the probability that the null hypothesis
    is true given the observed Sharpe ratio value. This is the "observed"
    FDR which conditions on the actual observation rather than just the
    critical value.

    Args:
        SR: Observed Sharpe ratio.
        SR0: Sharpe ratio under null hypothesis.
        SR1: Sharpe ratio under alternative hypothesis.
        T: Number of observations.
        p_H1: Prior probability that H1 is true.
        gamma3: Skewness of returns. Default 0.
        gamma4: Kurtosis of returns (non-excess). Default 3 (Gaussian).
        rho: Autocorrelation of returns. Default 0.
        K: Number of strategies for variance adjustment. Default 1.

    Returns:
        Posterior probability of H0 given the observed SR.

    Example:
        >>> # Higher observed SR should give lower probability of H0
        >>> fdr_low = oFDR(SR=0.3, SR0=0, SR1=0.5, T=24, p_H1=0.1)
        >>> fdr_high = oFDR(SR=0.8, SR0=0, SR1=0.5, T=24, p_H1=0.1)
        >>> bool(fdr_high < fdr_low)
        True
    """
    p0 = 1 - probabilistic_sharpe_ratio(
        SR, SR0, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K
    )
    p1 = 1 - probabilistic_sharpe_ratio(
        SR, SR1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K
    )
    p_H0 = 1 - p_H1
    return p0 * p_H0 / (p0 * p_H0 + p1 * p_H1)
