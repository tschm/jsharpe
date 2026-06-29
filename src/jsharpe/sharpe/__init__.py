"""Sharpe-related utilities for statistical analysis and hypothesis testing.

This package provides comprehensive tools for Sharpe ratio analysis, including:
    - Variance estimation under non-Gaussian returns
    - Statistical significance testing
    - Multiple testing corrections (FDR, FWER)
    - Portfolio optimization utilities

The implementation is split across topical sub-modules:
    - :mod:`jsharpe.sharpe.linalg`: probability points and covariance helpers
    - :mod:`jsharpe.sharpe.clustering`: effective rank and clustering
    - :mod:`jsharpe.sharpe.quadrature`: Gauss-Hermite expectation and moments
    - :mod:`jsharpe.sharpe.psr`: Sharpe variance, track record and PSR core
    - :mod:`jsharpe.sharpe.corrections`: FWER/FDR multiple-testing corrections
    - :mod:`jsharpe.sharpe.generators`: synthetic data and autocorrelation

The full public API is re-exported here so existing imports such as
``from jsharpe.sharpe import sharpe_ratio_variance`` keep resolving unchanged.
"""

from .clustering import effective_rank, number_of_clusters
from .corrections import (
    FDR_critical_value,
    adjusted_p_values_bonferroni,
    adjusted_p_values_holm,
    adjusted_p_values_sidak,
    control_for_FDR,
    oFDR,
    pFDR,
)
from .generators import (
    autocorrelation,
    generate_autocorrelated_non_gaussian_data,
    generate_non_gaussian_data,
    get_random_correlation_matrix,
)
from .linalg import (
    minimum_variance_weights_for_correlated_assets,
    ppoints,
    robust_covariance_inverse,
)
from .psr import (
    critical_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio_power,
    sharpe_ratio_variance,
    variance_of_the_maximum_of_k_Sharpe_ratios,
)
from .quadrature import E_under_normal, make_expectation_gh, moments_Mk

__all__ = [
    "E_under_normal",
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
    "moments_Mk",
    "number_of_clusters",
    "oFDR",
    "pFDR",
    "ppoints",
    "probabilistic_sharpe_ratio",
    "robust_covariance_inverse",
    "sharpe_ratio_power",
    "sharpe_ratio_variance",
    "variance_of_the_maximum_of_k_Sharpe_ratios",
]
