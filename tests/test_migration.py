# ruff: noqa: N802, N803, N806
"""Parametrized migration tests comparing jsharpe functions against zoonek/2025-sharpe-ratio."""

import numpy as np
import pytest

import jsharpe
from jsharpe.sharpe import moments_Mk

# ---------------------------------------------------------------------------
# Scalar-output functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "args", "kwargs", "tol"),
    [
        # sharpe_ratio_variance
        ("sharpe_ratio_variance", (0.5, 24), {}, 1e-12),
        ("sharpe_ratio_variance", (0.5, 24), {"gamma3": -2.448, "gamma4": 10.164}, 1e-12),
        ("sharpe_ratio_variance", (0.5, 24), {"rho": 0.2}, 1e-12),
        ("sharpe_ratio_variance", (0.5, 24), {"K": 5}, 1e-12),
        ("sharpe_ratio_variance", (0.036 / 0.079, 24), {"gamma3": -2.448, "gamma4": 10.164}, 1e-12),
        # minimum_track_record_length
        ("minimum_track_record_length", (0.5, 0.0), {}, 1e-12),
        (
            "minimum_track_record_length",
            (0.036 / 0.079, 0.0),
            {"gamma3": -2.448, "gamma4": 10.164, "alpha": 0.05},
            1e-12,
        ),
        ("minimum_track_record_length", (0.5, 0.1), {"rho": 0.2}, 1e-12),
        # probabilistic_sharpe_ratio
        ("probabilistic_sharpe_ratio", (0.5, 0.0), {"T": 24}, 1e-12),
        ("probabilistic_sharpe_ratio", (0.5, 0.1), {"T": 24, "gamma3": -2.448, "gamma4": 10.164}, 1e-12),
        ("probabilistic_sharpe_ratio", (0.5, 0.0), {"T": 24, "rho": 0.2}, 1e-12),
        ("probabilistic_sharpe_ratio", (0.036 / 0.079, 0.0), {"T": 24, "gamma3": -2.448, "gamma4": 10.164}, 1e-12),
        # critical_sharpe_ratio
        ("critical_sharpe_ratio", (0.0, 24), {}, 1e-12),
        ("critical_sharpe_ratio", (0.0, 24), {"gamma3": -2.448, "gamma4": 10.164}, 1e-12),
        ("critical_sharpe_ratio", (0.0, 24), {"rho": 0.2, "alpha": 0.10}, 1e-12),
        # sharpe_ratio_power
        ("sharpe_ratio_power", (0.0, 0.5, 24), {}, 1e-12),
        ("sharpe_ratio_power", (0.0, 0.5, 24), {"gamma3": -2.448, "gamma4": 10.164}, 1e-12),
        ("sharpe_ratio_power", (0.0, 0.5, 24), {"rho": 0.2, "alpha": 0.10}, 1e-12),
        # pFDR
        ("pFDR", (0.05, 0.05, 0.315), {}, 1e-12),
        ("pFDR", (0.10, 0.10, 0.30), {}, 1e-12),
        # oFDR
        ("oFDR", (0.5, 0.0, 0.5, 24, 0.05), {}, 1e-12),
        ("oFDR", (0.036 / 0.079, 0.0, 0.5, 24, 0.05), {"gamma3": -2.448, "gamma4": 10.164}, 1e-12),
        ("oFDR", (0.5, 0.0, 0.5, 24, 0.05), {"rho": 0.2, "K": 3}, 1e-12),
        # expected_maximum_sharpe_ratio
        ("expected_maximum_sharpe_ratio", (10, 0.1), {}, 1e-12),
        ("expected_maximum_sharpe_ratio", (10, 0.1, 0.5), {}, 1e-12),
        ("expected_maximum_sharpe_ratio", (5, 0.05), {}, 1e-12),
        # FDR_critical_value
        ("FDR_critical_value", (0.20, 0.0, 0.5, 0.2, 0.3, 0.1), {}, 1e-12),
        ("FDR_critical_value", (0.25, 0.0, 0.5, 0.3, 0.4, 0.05), {}, 1e-12),
    ],
)
def test_migration(original, method, args, kwargs, tol):
    """Verify each function produces the same result as the original implementation."""
    x = getattr(jsharpe, method)(*args, **kwargs)
    y = getattr(original, method)(*args, **kwargs)
    assert x == pytest.approx(y, abs=tol)


# ---------------------------------------------------------------------------
# ppoints — array output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "kwargs"),
    [
        (5, {}),
        (20, {}),
        (5, {"a": 0.5}),
        (10, {"a": 0.0}),
        (10, {"a": 3 / 8}),
    ],
)
def test_migration_ppoints(original, n, kwargs):
    """Ppoints should produce identical arrays."""
    assert np.allclose(jsharpe.ppoints(n, **kwargs), original.ppoints(n, **kwargs))


# ---------------------------------------------------------------------------
# moments_Mk — tuple output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "kwargs"),
    [
        (1, {}),
        (5, {}),
        (10, {}),
        (5, {"rho": 0.3}),
        (10, {"rho": 0.5}),
    ],
)
def test_migration_moments_Mk(original, k, kwargs):
    """moments_Mk should return identical (Ez, Ez2, var) tuples."""
    x = moments_Mk(k, **kwargs)
    y = original.moments_Mk(k, **kwargs)
    assert x == pytest.approx(y, abs=1e-12)


# ---------------------------------------------------------------------------
# effective_rank — scalar output, array input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C",
    [
        np.eye(3),
        np.eye(5),
        np.array([[10, 1, 7], [1, 10, 8], [7, 8, 10]]) / 10,
    ],
)
def test_migration_effective_rank(original, C):
    """effective_rank should return identical values."""
    assert jsharpe.effective_rank(C) == pytest.approx(original.effective_rank(C), abs=1e-12)


# ---------------------------------------------------------------------------
# control_for_FDR — tuple output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "kwargs"),
    [
        (0.25, {}),
        (0.25, {"SR0": 0.0, "SR1": 0.5, "T": 24, "gamma3": -2.448, "gamma4": 10.164}),
        (0.20, {"rho": 0.2}),
    ],
)
def test_migration_control_for_FDR(original, q, kwargs):
    """control_for_FDR should return identical (alpha, beta, SR_c, q_hat) tuples."""
    x = jsharpe.control_for_FDR(q, **kwargs)
    y = original.control_for_FDR(q, **kwargs)
    assert x == pytest.approx(y, abs=1e-12)


# ---------------------------------------------------------------------------
# Adjusted p-values — array output
# ---------------------------------------------------------------------------

_PS = np.array([0.01, 0.03, 0.05, 0.10, 0.50])


@pytest.mark.parametrize("method", ["adjusted_p_values_bonferroni", "adjusted_p_values_sidak"])
def test_migration_adjusted_p_values(original, method):
    """Bonferroni and Šidák corrections should produce identical arrays."""
    assert np.allclose(getattr(jsharpe, method)(_PS), getattr(original, method)(_PS))


@pytest.mark.parametrize("variant", ["bonferroni", "sidak"])
def test_migration_adjusted_p_values_holm(original, variant):
    """Holm correction should produce identical arrays for both variants."""
    x = jsharpe.adjusted_p_values_holm(_PS, variant=variant)
    y = original.adjusted_p_values_holm(_PS, variant=variant)
    assert np.allclose(x, y)


# ---------------------------------------------------------------------------
# robust_covariance_inverse and minimum_variance_weights — array output
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def constant_corr_cov():
    """Constant-correlation covariance matrix for array comparison tests."""
    np.random.seed(0)
    rho = 0.5
    C = rho * np.ones((10, 10))
    np.fill_diagonal(C, 1)
    sigma = np.random.lognormal(size=10).reshape(-1, 1)
    return (C * sigma).T * sigma


def test_migration_robust_covariance_inverse(original, constant_corr_cov):
    """robust_covariance_inverse should return identical matrices."""
    V = constant_corr_cov
    assert np.allclose(jsharpe.robust_covariance_inverse(V), original.robust_covariance_inverse(V))


def test_migration_minimum_variance_weights(original, constant_corr_cov):
    """minimum_variance_weights_for_correlated_assets should return identical weights."""
    V = constant_corr_cov
    assert np.allclose(
        jsharpe.minimum_variance_weights_for_correlated_assets(V),
        original.minimum_variance_weights_for_correlated_assets(V),
    )


# ---------------------------------------------------------------------------
# autocorrelation — scalar output, matrix input
# ---------------------------------------------------------------------------


def test_migration_autocorrelation(original):
    """Autocorrelation should return identical mean lag-1 autocorrelation."""
    np.random.seed(42)
    X = np.random.normal(size=(100, 3))
    assert jsharpe.autocorrelation(X) == pytest.approx(original.autocorrelation(X), abs=1e-12)


# ---------------------------------------------------------------------------
# make_expectation_gh — returns a callable
# ---------------------------------------------------------------------------


def test_migration_make_expectation_gh(original):
    """make_expectation_gh should produce callables that integrate identically."""
    E_js = jsharpe.make_expectation_gh(n_nodes=50)
    E_orig = original.make_expectation_gh(n_nodes=50)
    for g in [lambda z: z**2, lambda z: np.exp(-(z**2) / 2), lambda z: np.ones_like(z)]:
        assert E_js(g) == pytest.approx(E_orig(g), abs=1e-12)


# ---------------------------------------------------------------------------
# Stochastic data-generation functions — compared with matching seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gaussian", "mild", "moderate", "severe"])
def test_migration_generate_non_gaussian_data(original, name):
    """generate_non_gaussian_data should produce identical arrays given the same seed."""
    np.random.seed(42)
    x = jsharpe.generate_non_gaussian_data(100, 2, SR0=0.1, name=name)
    np.random.seed(42)
    y = original.generate_non_gaussian_data(100, 2, SR0=0.1, name=name)
    assert np.allclose(x, y)


def test_migration_generate_autocorrelated_non_gaussian_data(original):
    """generate_autocorrelated_non_gaussian_data should produce identical arrays given the same seed."""
    np.random.seed(42)
    x = jsharpe.generate_autocorrelated_non_gaussian_data(100, 2, SR0=0.0, name="mild", rho=0.3)
    np.random.seed(42)
    y = original.generate_autocorrelated_non_gaussian_data(100, 2, SR0=0.0, name="mild", rho=0.3)
    assert np.allclose(x, y)


def test_migration_get_random_correlation_matrix(original):
    """get_random_correlation_matrix should produce identical (C, X, clusters) given the same seed."""
    kwargs = {"number_of_trials": 20, "effective_number_of_trials": 4, "number_of_observations": 50}
    np.random.seed(42)
    C_js, X_js, clusters_js = jsharpe.get_random_correlation_matrix(**kwargs)
    np.random.seed(42)
    C_orig, X_orig, clusters_orig = original.get_random_correlation_matrix(**kwargs)
    assert np.allclose(C_js, C_orig)
    assert np.allclose(X_js, X_orig)
    assert np.array_equal(clusters_js, clusters_orig)
