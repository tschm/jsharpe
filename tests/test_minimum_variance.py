"""Tests for minimum variance portfolio functions."""

import cvxpy as cp
import numpy as np

from jsharpe.sharpe import minimum_variance_weights_for_correlated_assets
from src.jsharpe.sharpe import robust_covariance_inverse


def test_minimum_variance_weights_for_correlated_assets():
    """Test minimum variance portfolio weights computation."""
    np.random.seed(0)
    rho = 0.5
    C = rho * np.ones(shape=(10, 10))
    np.fill_diagonal(C, 1)
    sigma = np.random.lognormal(size=10).reshape(-1, 1)
    V = (C * sigma).T * sigma
    w = minimum_variance_weights_for_correlated_assets(V)

    W = cp.Variable(shape=V.shape[0])
    problem = cp.Problem(cp.Minimize(cp.quad_form(W, V)), [W.sum() == 1])
    problem.solve()
    assert np.all(np.abs(W.value - w) < 1e-10)


def test_robust_covariance_inverse():
    """Test robust covariance inverse computation."""
    np.random.seed(0)
    rho = 0.5
    C = rho * np.ones(shape=(10, 10))
    np.fill_diagonal(C, 1)
    sigma = np.random.lognormal(size=10).reshape(-1, 1)
    V = (C * sigma).T * sigma
    assert np.all(np.abs(np.linalg.inv(V) - robust_covariance_inverse(V)) < 1e-12)
