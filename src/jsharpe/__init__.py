"""JSharpe: Sharpe Ratio Analysis and Statistical Testing.

This package provides comprehensive tools for Sharpe ratio analysis,
including statistical significance testing, multiple testing corrections,
and portfolio optimization utilities.

Key features:
    - Sharpe ratio variance estimation under non-Gaussian returns
    - Minimum track record length computation
    - Probabilistic Sharpe Ratio (PSR) calculation
    - False Discovery Rate (FDR) control for multiple strategy testing
    - Family-Wise Error Rate (FWER) corrections (Bonferroni, Šidák, Holm)
    - Minimum variance portfolio optimization

Example:
    >>> import numpy as np
    >>> from jsharpe.sharpe import sharpe_ratio_variance, probabilistic_sharpe_ratio
    >>> # Compute variance of a Sharpe ratio estimate
    >>> var = sharpe_ratio_variance(SR=0.5, T=24)
    >>> # Compute the Probabilistic Sharpe Ratio
    >>> psr = probabilistic_sharpe_ratio(SR=0.5, SR0=0, T=24)
"""

from .sharpe import (
    FDR_critical_value,
    adjusted_p_values_bonferroni,
    adjusted_p_values_holm,
    adjusted_p_values_sidak,
    autocorrelation,
    control_for_FDR,
    critical_sharpe_ratio,
    effective_rank,
    expected_maximum_sharpe_ratio,
    generate_autocorrelated_non_gaussian_data,
    generate_non_gaussian_data,
    get_random_correlation_matrix,
    make_expectation_gh,
    minimum_track_record_length,
    minimum_variance_weights_for_correlated_assets,
    oFDR,
    pFDR,
    ppoints,
    probabilistic_sharpe_ratio,
    robust_covariance_inverse,
    sharpe_ratio_power,
    sharpe_ratio_variance,
    variance_of_the_maximum_of_k_Sharpe_ratios,
)

__all__ = [
    "FDR_critical_value",
    "adjusted_p_values_bonferroni",
    "adjusted_p_values_holm",
    "adjusted_p_values_sidak",
    "autocorrelation",
    "control_for_FDR",
    "critical_sharpe_ratio",
    "effective_rank",
    "expected_maximum_sharpe_ratio",
    "generate_autocorrelated_non_gaussian_data",
    "generate_non_gaussian_data",
    "get_random_correlation_matrix",
    "make_expectation_gh",
    "minimum_track_record_length",
    "minimum_variance_weights_for_correlated_assets",
    "oFDR",
    "pFDR",
    "ppoints",
    "probabilistic_sharpe_ratio",
    "robust_covariance_inverse",
    "sharpe_ratio_power",
    "sharpe_ratio_variance",
    "variance_of_the_maximum_of_k_Sharpe_ratios",
]
