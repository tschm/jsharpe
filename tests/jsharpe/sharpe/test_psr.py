"""Tests for :mod:`jsharpe.sharpe.psr` (Sharpe variance, track record, PSR, power).

Combines pinned reference values (documented numeric examples) with behavioural
invariants — bounds and monotonicity in each argument — that must hold for any
correct implementation.
"""
# ruff: noqa: N806

import math

import pytest

from jsharpe import (
    critical_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio_power,
    sharpe_ratio_variance,
    variance_of_the_maximum_of_k_Sharpe_ratios,
)

# ---- pinned numeric contracts ----------------------------------------------


def test_sharpe_ratio_variance_pinned_values():
    """Std error of the Sharpe ratio matches the documented Gaussian/non-Gaussian values."""
    SR = 0.036 / 0.079
    var_ng = sharpe_ratio_variance(SR=SR, gamma3=-2.448, gamma4=10.164, T=24)
    var_g = sharpe_ratio_variance(SR=SR, gamma3=0, gamma4=3, T=24)
    assert round(math.sqrt(var_ng), 3) == 0.329
    assert round(math.sqrt(var_g), 3) == 0.214


def test_minimum_track_record_length_pinned_value():
    """MinTRL matches the documented worked example."""
    mtrl = minimum_track_record_length(SR=0.036 / 0.079, SR0=0, gamma3=-2.448, gamma4=10.164, alpha=0.05)
    assert round(mtrl, 3) == 13.029


def test_probabilistic_sharpe_ratio_pinned_values():
    """PSR matches the documented values for SR0=0 and SR0=0.1."""
    psr0 = probabilistic_sharpe_ratio(SR=0.036 / 0.079, SR0=0, T=24, gamma3=-2.448, gamma4=10.164)
    psr1 = probabilistic_sharpe_ratio(SR=0.036 / 0.079, SR0=0.1, T=24, gamma3=-2.448, gamma4=10.164)
    assert round(psr0, 3) == 0.987
    assert round(psr1, 3) == 0.939


def test_sharpe_ratio_power_pinned_value():
    """The type-II error (1 - power) matches the documented value."""
    power = sharpe_ratio_power(SR0=0, SR1=0.5, T=24, gamma3=-2.448, gamma4=10.164)
    assert round(1 - power, 3) == 0.315


def test_probabilistic_sharpe_ratio_with_variance_and_t_conflict_raises():
    """Providing both variance and T should raise an assertion error."""
    with pytest.raises(AssertionError):
        probabilistic_sharpe_ratio(SR=0.5, SR0=0.0, variance=0.04, T=24)


def test_probabilistic_sharpe_ratio_accepts_explicit_variance():
    """PSR can be computed from an explicit variance instead of (T, moments)."""
    psr = probabilistic_sharpe_ratio(SR=0.5, SR0=0.0, variance=0.04)
    assert 0.0 < psr < 1.0


# ---- behavioural invariants -------------------------------------------------


def test_sharpe_ratio_variance_positive_and_monotone_in_trials():
    """Variance is strictly positive and shrinks with the multiple-testing factor K."""
    var_single = sharpe_ratio_variance(SR=0.5, T=24)
    var_many = sharpe_ratio_variance(SR=0.5, T=24, K=20)
    assert var_single > 0
    assert var_many > 0
    assert var_many < var_single


def test_sharpe_ratio_variance_shrinks_with_more_observations():
    """More observations reduce the estimator variance (proportional to 1/T)."""
    assert sharpe_ratio_variance(SR=0.5, T=120) < sharpe_ratio_variance(SR=0.5, T=12)


def test_sharpe_ratio_variance_grows_with_kurtosis():
    """Fatter tails (higher kurtosis) inflate the estimator variance."""
    assert sharpe_ratio_variance(SR=0.5, T=24, gamma4=6.0) > sharpe_ratio_variance(SR=0.5, T=24)


def test_variance_of_maximum_monotonic_in_k():
    """Variance of the maximum Sharpe ratio increases with the number of trials."""
    assert variance_of_the_maximum_of_k_Sharpe_ratios(5, 0.1) > variance_of_the_maximum_of_k_Sharpe_ratios(1, 0.1)


def test_probabilistic_sharpe_ratio_bounds_and_monotonicity():
    """PSR lies in (0, 1), rises with the observed SR, and falls as the benchmark SR0 rises."""
    psr = probabilistic_sharpe_ratio(SR=0.5, SR0=0.0, T=24)
    assert 0.0 < psr < 1.0
    assert probabilistic_sharpe_ratio(SR=1.0, SR0=0.0, T=24) > psr
    assert probabilistic_sharpe_ratio(SR=0.5, SR0=0.3, T=24) < psr


def test_minimum_track_record_length_positive_and_stricter_alpha_needs_more_data():
    """MinTRL is positive and grows as the confidence requirement tightens (smaller alpha)."""
    mtrl_loose = minimum_track_record_length(SR=0.5, SR0=0.0, alpha=0.10)
    mtrl_strict = minimum_track_record_length(SR=0.5, SR0=0.0, alpha=0.01)
    assert mtrl_loose > 0
    assert mtrl_strict > mtrl_loose


def test_critical_sharpe_ratio_positive_and_falls_as_alpha_relaxes():
    """SR_c to reject H0: SR=0 is positive and decreases as alpha is relaxed."""
    sr_c_strict = critical_sharpe_ratio(SR0=0.0, T=24, alpha=0.01)
    sr_c_loose = critical_sharpe_ratio(SR0=0.0, T=24, alpha=0.10)
    assert sr_c_strict > 0
    assert sr_c_loose < sr_c_strict


def test_sharpe_ratio_power_bounds_and_grows_with_sample_and_effect():
    """Power is in [0, 1] and increases with both sample size T and the alternative SR1."""
    power = sharpe_ratio_power(SR0=0.0, SR1=0.5, T=24)
    assert 0.0 <= power <= 1.0
    assert sharpe_ratio_power(SR0=0.0, SR1=0.5, T=96) > power
    assert sharpe_ratio_power(SR0=0.0, SR1=0.8, T=24) > power


def test_expected_maximum_sharpe_ratio_grows_with_trials():
    """The expected maximum Sharpe ratio increases with the number of trials."""
    assert expected_maximum_sharpe_ratio(number_of_trials=50, variance=0.1) > expected_maximum_sharpe_ratio(
        number_of_trials=2, variance=0.1
    )
