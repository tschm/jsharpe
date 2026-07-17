"""Tests for :mod:`jsharpe.sharpe.generators` (synthetic data, autocorrelation).

Assertions target the statistical properties the generators promise: the sample
Sharpe ratio of generated data, recovered autocorrelation, and the shape /
symmetry / clustering structure of random correlation matrices.
"""
# ruff: noqa: N806

import numpy as np
import pytest

from jsharpe import (
    autocorrelation,
    generate_autocorrelated_non_gaussian_data,
    generate_non_gaussian_data,
    get_random_correlation_matrix,
)


@pytest.mark.parametrize("name", ["gaussian", "mild", "moderate", "severe"])
def test_generate_non_gaussian_data_sr0_shift(name):
    """Generated non-Gaussian data should have a sample SR close to the requested SR0."""
    np.random.seed(0)
    X = generate_non_gaussian_data(4000, 1, SR0=0.2, name=name)
    sr = float(X.mean()) / float(X.std(ddof=0))
    assert sr == pytest.approx(0.2, abs=0.05)


def test_generate_non_gaussian_data_rejects_unknown_name():
    """An unknown distribution name is rejected."""
    with pytest.raises(AssertionError):
        generate_non_gaussian_data(10, 1, name="does-not-exist")


def test_generate_autocorrelated_non_gaussian_data_recovers_rho():
    """AR(1) non-Gaussian data has mean lag-1 autocorrelation close to the target rho."""
    np.random.seed(0)
    N, n, rho = 800, 4, 0.3
    X = generate_autocorrelated_non_gaussian_data(N, n, SR0=0.0, name="mild", rho=rho, gaussian_autocorrelation=0.0)
    assert X.shape == (N, n)
    assert autocorrelation(X) == pytest.approx(rho, abs=0.08)


def test_generate_autocorrelated_non_gaussian_data_rho_none_uses_gaussian_autocorrelation():
    """When rho is None the gaussian_autocorrelation argument is used instead."""
    np.random.seed(42)
    X = generate_autocorrelated_non_gaussian_data(200, 2, SR0=0.0, name="gaussian", gaussian_autocorrelation=0.2)
    assert X.shape == (200, 2)


def test_get_random_correlation_matrix_structure():
    """Random correlation matrix is symmetric with unit diagonal and in-range cluster labels."""
    np.random.seed(1)
    C, X, clusters = get_random_correlation_matrix(
        number_of_trials=30, effective_number_of_trials=5, number_of_observations=200, noise=0.05
    )
    assert C.shape == (30, 30)
    assert X.shape == (200, 30)
    assert clusters.shape == (30,)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 1)
    assert np.all(np.abs(C) <= 1)
    assert clusters.min() >= 0
    assert clusters.max() < 5


def test_get_random_correlation_matrix_blocks_are_non_empty():
    """Every one of the requested clusters is populated (no empty block)."""
    np.random.seed(3)
    _, _, clusters = get_random_correlation_matrix(
        number_of_trials=40, effective_number_of_trials=6, number_of_observations=100, noise=0.1
    )
    assert len(np.unique(clusters)) == 6


def test_autocorrelation_of_iid_data_is_near_zero():
    """Independent Gaussian columns have mean lag-1 autocorrelation near zero."""
    np.random.seed(42)
    X = np.random.normal(size=(2000, 5))
    assert abs(autocorrelation(X)) < 0.1
