"""Sharpe-ratio variance, track record, and probabilistic Sharpe ratio.

This module groups the core statistical-inference routines for the Sharpe
ratio: its asymptotic variance, the variance/expectation of the maximum
across many trials, the minimum track record length, the critical Sharpe
ratio, the Probabilistic Sharpe Ratio (PSR), and the associated power.
"""
# ruff: noqa: N802, N803, N806, S101

import math

import numpy as np
import scipy

from .quadrature import moments_Mk


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


def variance_of_the_maximum_of_k_Sharpe_ratios(number_of_trials: int, variance: float) -> float:
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


def expected_maximum_sharpe_ratio(number_of_trials: int, variance: float, SR0: float = 0) -> float:
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
                + np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / number_of_trials / np.exp(1))
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
        variance = sharpe_ratio_variance(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
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
    critical_SR = critical_sharpe_ratio(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, alpha=alpha)
    variance = sharpe_ratio_variance(SR1, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    beta = scipy.stats.norm.cdf((critical_SR - SR1) / math.sqrt(variance))
    return float(1 - beta)
