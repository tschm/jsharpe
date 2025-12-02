"""Top-level package for jsharpe.

This module exposes the primary public functions from ``jsharpe.sharpe`` for
convenient imports, e.g. ``from jsharpe import probabilistic_sharpe_ratio``.
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
