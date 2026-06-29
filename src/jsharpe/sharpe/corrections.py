"""Multiple-testing corrections and false-discovery-rate control.

This module groups the family-wise error rate corrections (Bonferroni,
Šidák, Holm) and the false-discovery-rate routines (critical value, FDR
control, predictive pFDR and observed oFDR) used when screening many
candidate strategies.
"""
# ruff: noqa: N802, N803, N806, S101

import math
import warnings

import numpy as np
import scipy

from .psr import probabilistic_sharpe_ratio, sharpe_ratio_variance


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


def adjusted_p_values_holm(ps: np.ndarray, *, variant: str = "bonferroni") -> np.ndarray:
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
        candidate = min(1, ps[idx] * (M - j)) if variant == "bonferroni" else 1 - (1 - ps[idx]) ** (M - j)
        p_adjusted[idx] = max(previous, candidate)
        previous = p_adjusted[idx]
    return p_adjusted


def FDR_critical_value(q: float, SR0: float, SR1: float, sigma0: float, sigma1: float, p_H1: float) -> float:
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
    assert sigma0 > 0
    assert sigma1 > 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")

        def f(c: float) -> float:
            """Compute the posterior probability P[H=0 | X > c].

            Args:
                c: Candidate critical value.

            Returns:
                Posterior false discovery probability at threshold c.
            """
            a = 1 / (
                1
                + scipy.stats.norm.sf((c - SR1) / sigma1) / scipy.stats.norm.sf((c - SR0) / sigma0) * p_H1 / (1 - p_H1)
            )
            return float(np.where(np.isfinite(a), a, 0))

        if f(-10) < q:  # Solution outside of the search interval
            return float(-np.inf)

        if (f(-10) - q) * (f(10) - q) > 0:  # No solution, for instance if σ₀≫σ₁ and q small
            return float(np.nan)

        return float(scipy.optimize.brentq(lambda c: f(c) - q, -10, 10))


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

    s0 = math.sqrt(sharpe_ratio_variance(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K))
    s1 = math.sqrt(sharpe_ratio_variance(SR1, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K))
    SRc = FDR_critical_value(q, SR0, SR1, s0, s1, p_H1)

    beta = Z((SRc - SR1) / s1)
    alpha = q / (1 - q) * p_H1 / (1 - p_H1) * (1 - beta)
    q_hat = 1 / (1 + (1 - beta) / alpha * p_H1 / (1 - p_H1))

    return alpha, beta, SRc, q_hat


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
    p0 = 1 - probabilistic_sharpe_ratio(SR, SR0, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    p1 = 1 - probabilistic_sharpe_ratio(SR, SR1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    p_H0 = 1 - p_H1
    return p0 * p_H0 / (p0 * p_H0 + p1 * p_H1)
