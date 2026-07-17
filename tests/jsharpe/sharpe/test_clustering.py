"""Tests for :mod:`jsharpe.sharpe.clustering` (effective rank, clustering).

Assertions target the documented behaviour and known limiting cases of the
effective rank and the optimal-cluster-count routine.
"""
# ruff: noqa: N806

import numpy as np
import pytest

from jsharpe import effective_rank, get_random_correlation_matrix, number_of_clusters

# ---- effective_rank ---------------------------------------------------------


def test_effective_rank_limiting_cases():
    """Identity -> full rank, perfectly correlated -> rank 1, plus a pinned intermediate value."""
    assert abs(effective_rank(np.eye(3)) - 3) < 1e-12
    assert abs(effective_rank(np.ones((3, 3))) - 1.0) < 1e-10
    C = np.array([[10, 1, 7], [1, 10, 8], [7, 8, 10]]) / 10
    assert abs(effective_rank(C[:2, :2]) - 2) < 0.02
    assert abs(effective_rank(C) - 1.84) < 0.01


def test_effective_rank_within_bounds():
    """Effective rank of a correlation matrix lies between 1 and its dimension."""
    np.random.seed(1)
    C, _, _ = get_random_correlation_matrix(
        number_of_trials=30, effective_number_of_trials=5, number_of_observations=200, noise=0.05
    )
    assert 1 <= effective_rank(C) <= 30


# ---- number_of_clusters -----------------------------------------------------


def test_number_of_clusters_basic():
    """Returns a valid cluster count, a per-k quality dict, and a labels array."""
    np.random.seed(7)
    C, _, _ = get_random_correlation_matrix(
        number_of_trials=30, effective_number_of_trials=5, number_of_observations=200, noise=0.05
    )
    n, qualities, labels = number_of_clusters(C, retries=3, max_clusters=10)
    assert 2 <= n <= 10
    assert set(qualities.keys()) == set(range(2, 11))
    assert all(isinstance(q, float) for q in qualities.values())
    assert n == max(qualities, key=lambda x: qualities[x])
    assert labels.shape == (30,)
    assert labels.min() >= 0
    assert labels.max() < n


def test_number_of_clusters_identity_matrix():
    """On an identity correlation matrix a valid k in [2, n-1] is still returned."""
    n = 5
    k, _qualities, labels = number_of_clusters(np.eye(n), retries=2, max_clusters=n - 1)
    assert 2 <= k <= n - 1
    assert labels.shape == (n,)


def test_number_of_clusters_block_structure():
    """A low-noise block-structured matrix recovers approximately the true cluster count."""
    np.random.seed(0)
    C, _, true_clusters = get_random_correlation_matrix(
        number_of_trials=20, effective_number_of_trials=4, number_of_observations=500, noise=0.01
    )
    true_k = len(np.unique(true_clusters))
    k, _qualities, _labels = number_of_clusters(C, retries=5, max_clusters=8)
    assert abs(k - true_k) <= 2


def test_number_of_clusters_skips_zero_std_quality(monkeypatch):
    """A k whose silhouette scores have zero spread (std == 0) is skipped, not selected."""

    def fake_kmeans2(data, k, **_kwargs):
        """Return a deterministic cluster assignment: symmetric for k=2, uneven otherwise."""
        labels = np.array([0, 0, 1, 1]) if k == 2 else np.array([0, 1, 2, 0])
        return np.zeros((k, data.shape[1])), labels

    monkeypatch.setattr("scipy.cluster.vq.kmeans2", fake_kmeans2)

    r = 0.2
    C = np.array([[1, 1, r, r], [1, 1, r, r], [r, r, 1, 1], [r, r, 1, 1]], dtype=float)
    n, qualities, labels = number_of_clusters(C, retries=2, max_clusters=3)

    assert qualities[2] == -np.inf  # zero-spread k=2 skipped
    assert n == 3
    assert labels.shape == (4,)


def test_number_of_clusters_raises_when_no_valid_solution():
    """A 2x2 matrix admits no k in [2, n-1], so no clustering is found and it raises."""
    with pytest.raises(RuntimeError, match="No valid clustering solution"):
        number_of_clusters(np.eye(2))


def test_number_of_clusters_rejects_non_correlation_matrix():
    """A matrix with out-of-range entries fails the correlation-matrix precondition."""
    with pytest.raises(AssertionError):
        number_of_clusters(np.array([[1.0, 2.0], [2.0, 1.0]]))
