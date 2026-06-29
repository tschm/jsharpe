"""Clustering and effective-dimensionality utilities.

This module groups the routines used to assess the structure of a
correlation matrix: the effective rank and the optimal number of
clusters (with silhouette-based quality scoring).
"""
# ruff: noqa: N803, N806, S101, TRY003

import math
import warnings

import numpy as np
import scipy


def effective_rank(C: np.ndarray) -> float:
    """Compute the effective rank of a positive semi-definite matrix.

    The effective rank measures the "effective dimensionality" of a matrix
    by computing the exponential of the entropy of its normalized eigenvalues.
    This provides a continuous measure between 1 (perfectly correlated) and
    n (perfectly uncorrelated/identity matrix).

    Algorithm:
        1. Compute eigenvalues (non-negative for PSD matrices)
        2. Discard zero eigenvalues
        3. Normalize to form a probability distribution
        4. Compute entropy H = -Σ p_i log(p_i)
        5. Return exp(H)

    Args:
        C: Positive semi-definite matrix (e.g., correlation matrix).
            Shape (n, n).

    Returns:
        Effective rank, a value in [1, n] where n is the matrix dimension.

    References:
        Roy, O. and Vetterli, M. (2007). "The effective rank: a measure of
        effective dimensionality." EURASIP Journal on Advances in Signal
        Processing. http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.177.2721

    Example:
        >>> import numpy as np
        >>> # Identity matrix has effective rank equal to its dimension
        >>> abs(effective_rank(np.eye(3)) - 3.0) < 1e-10
        True
        >>> # Perfectly correlated matrix has effective rank 1
        >>> C = np.ones((3, 3))
        >>> abs(effective_rank(C) - 1.0) < 1e-10
        True
    """
    p = np.linalg.eigvalsh(C)
    p = p[p > 0]
    p = p / sum(p)
    H = np.sum(-p * np.log(p))
    return math.exp(H)


def _silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute silhouette coefficients for each sample.

    For each sample i:
        a(i) = mean distance from i to all other samples in the same cluster
        b(i) = min over other clusters c of mean distance from i to samples in c
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        labels: Cluster labels of shape (n_samples,).

    Returns:
        Silhouette coefficients of shape (n_samples,).
    """
    n = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Pairwise Euclidean distances: dist[i,j] = ||X[i] - X[j]||
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    # For each cluster c, compute mean distance from every sample to cluster c.
    # For samples IN cluster c, the self-distance (zero) is excluded.
    label_to_idx = {c: idx for idx, c in enumerate(unique_labels)}
    cluster_mean_dist = np.zeros((n, n_clusters))
    for c_idx, c in enumerate(unique_labels):
        mask = labels == c
        count = int(mask.sum())
        dist_to_cluster = dist[:, mask].sum(axis=1)
        cluster_mean_dist[:, c_idx] = dist_to_cluster / max(count, 1)
        # For members of cluster c, exclude self-distance (which is 0)
        in_cluster = np.where(mask)[0]
        if count > 1:
            cluster_mean_dist[in_cluster, c_idx] = dist_to_cluster[in_cluster] / (count - 1)
        else:
            cluster_mean_dist[in_cluster, c_idx] = 0.0

    # a(i): mean distance to same cluster
    own_cluster_idx = np.array([label_to_idx[lbl] for lbl in labels])
    a = cluster_mean_dist[np.arange(n), own_cluster_idx]

    # b(i): min mean distance to any other cluster
    b_mat = cluster_mean_dist.copy()
    b_mat[np.arange(n), own_cluster_idx] = np.inf
    b = b_mat.min(axis=1)
    b = np.where(b == np.inf, 0.0, b)

    max_ab = np.maximum(a, b)
    return np.where(max_ab > 0, (b - a) / max_ab, 0.0)


def number_of_clusters(
    C: np.ndarray,
    *,
    retries: int = 10,
    max_clusters: int = 100,
) -> tuple[int, dict[int, float], np.ndarray]:
    """Compute the optimal number of clusters from a correlation matrix.

    Implements the algorithm from section 8.1 of Lopez de Prado (2018):
        1. Convert the correlation matrix into a distance matrix.
        2. Using the columns of the distance matrix as features, run the
           k-means algorithm for each k and compute the quality of the
           clustering.
        3. Return the clustering with the highest quality.

    Quality is defined as the mean of the silhouette scores divided by
    their standard deviation.

    Args:
        C: Correlation matrix. Must be square, symmetric, finite, with ones
            on the diagonal and all entries in [-1, 1].
        retries: Number of times to run k-means for each k to reduce the
            impact of random initialisation. Default 10.
        max_clusters: Maximum number of clusters to evaluate. Capped at
            ``C.shape[0] - 1``. Default 100.

    Returns:
        Tuple of (n_clusters, qualities, labels):
            - n_clusters: Optimal number of clusters.
            - qualities: Dict mapping k to its quality score.
            - labels: Cluster assignment for each observation (shape (n,)).

    References:
        Lopez de Prado, M. (2018). "Detection of false investment strategies
        using unsupervised learning methods." SSRN 3167017.
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017

    Example:
        >>> from jsharpe.sharpe.generators import get_random_correlation_matrix
        >>> np.random.seed(42)
        >>> C, _, _ = get_random_correlation_matrix(
        ...     number_of_trials=20, effective_number_of_trials=4
        ... )
        >>> n, qualities, labels = number_of_clusters(C, retries=3, max_clusters=8)
        >>> 2 <= n <= 8
        True
        >>> labels.shape
        (20,)
    """
    assert isinstance(C, np.ndarray)
    assert np.all(C >= -1)
    assert np.all(C <= 1)
    assert np.all(np.diag(C) == 1)
    assert np.all(np.isfinite(C))

    max_clusters = min(max_clusters, C.shape[0] - 1)

    # Convert correlations to distances
    D = np.sqrt((1 - C) / 2)
    assert np.all(np.isfinite(D))

    qualities: dict[int, float] = {}
    best_labels: dict[int, np.ndarray] = {}
    for k in range(2, max_clusters + 1):
        qualities[k] = -np.inf
        for _ in range(retries):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _centroids, labels = scipy.cluster.vq.kmeans2(D, k, minit="points", iter=300)
            # Skip degenerate solutions with empty clusters
            if len(np.unique(labels)) < k:
                continue
            silhouette_vals = _silhouette_samples(D, labels)
            std = silhouette_vals.std()
            if std == 0:
                continue
            q = float(silhouette_vals.mean() / std)
            if q > qualities[k]:
                qualities[k] = q
                best_labels[k] = labels.copy()

    # Select the best k among those for which a valid solution was found
    valid_k = {k: q for k, q in qualities.items() if k in best_labels}
    if not valid_k:
        raise RuntimeError("No valid clustering solution found; try increasing retries or reducing max_clusters.")
    best_k = max(valid_k, key=lambda x: valid_k[x])
    return best_k, qualities, best_labels[best_k]
