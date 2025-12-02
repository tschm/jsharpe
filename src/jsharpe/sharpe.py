"""Sharpe-related utilities, including probability points generation (ppoints)."""

import math
import warnings

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def ppoints(n, a=None):
    """Equidistant points in [0,1], to be used as arguments of the pdf or icdf of distributions.

    Boundaries are excluded.
    See the documentation of the corresponding R function for more details.

    Inputs: n: integer, desired number of points
            a: offset
    Output: numpy array, with n equidistant points

    Example:
        ppoints(20)
    """
    if a is None:
        a = 0.5 if n > 10 else 3 / 8
    assert 0 <= a <= 1, f"the offset should be in [0,1], got {a}"
    return np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)


def robust_covariance_inverse(V: np.ndarray) -> np.ndarray:
    r"""Inverse of a constant-correlation covariance matrix, using the Sherman–Morrison formula.

    Assume $V = \rho \sigma \sigma' + (1-\rho) \text{diag}(\sigma^2)$
    (variance matrix, with constant correlations).
    Its inverse is $V^{-1} = A^{-1} - \dfrac{ A^{01} \rho \sigma \sigma' A^{-1} }
    { 1 + \rho \sigma' A^{-1} \sigma }$.

    Args:
        V: np.ndarray, variance matrix

    Returns:
        np.ndarray, inverse of the variance matrix
    """
    sigma = np.sqrt(np.diag(V))
    C = (V.T / sigma).T / sigma
    rho = np.mean(C[np.triu_indices_from(C, 1)])
    A = np.diag(1 / sigma**2) / (1 - rho)
    sigma = sigma.reshape(-1, 1)
    return A - (rho * A @ sigma @ sigma.T @ A) / (1 + rho * sigma.T @ A @ sigma)


def minimum_variance_weights_for_correlated_assets(V: np.ndarray) -> np.ndarray:
    """Compute weights of the minimum variance portfolio for correlated assets.

    Assumes a constant-correlation covariance matrix.

    Args:
        V: np.ndarray, variance matrix, shape (n,n)

    Returns:
        np.ndarray, weights of the minimum variance portfolio, shape (n,)
    """
    ones = np.ones(shape=V.shape[0])
    S = robust_covariance_inverse(V)
    w = S @ ones
    w = w / np.sum(w)
    return w


def effective_rank(C: np.ndarray) -> float:
    """Compute the effective rank of a correlation matrix.

    The effective rank of a positive semi-definite matrix is computed as follows:
    - Compute the eigenvalues; they are non-negative
    - Discard zeros
    - Normalize the remaining eigenvalues to sum to 1 (probability distribution)
    - Compute its entropy
    - "Invert" the entropy to have an effective number of items
      (the number of items for which the uniform distribution has the same entropy)

    Args:
        C: np.ndarray, square, a positive semi-definite matrix

    Returns:
        float, effective rank

    References:
        [1] The effective rank: a measure of effective dimensionality
            O. Roy and M. Vetterli (2007)
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.177.2721
    """
    p = np.linalg.eigvalsh(C)
    p = p[p > 0]
    p = p / sum(p)
    H = np.sum(-p * np.log(p))
    return math.exp(H)


def variance_of_the_clustered_trials(X: np.ndarray, clusters: np.ndarray) -> tuple[float, np.ndarray, pd.DataFrame]:
    """Compute variance of Sharpe ratios across cluster portfolios.

    Computes the returns of a minimum variance portfolio in each cluster,
    the corresponding Sharpe ratios, and then the variance of those Sharpe ratios.

    Args:
        X: numpy array of returns, one column per strategy
        clusters: cluster assignment, list (or array) with one element per strategy

    Returns:
        Tuple containing:
        - the variance of the Sharpe ratios of the cluster portfolios
        - The Sharpe ratios of the cluster portfolios
        - The time series of returns of the cluster portfolios
    """
    assert X.shape[1] == len(clusters)
    # Minimum variance portfolio in each cluster (assuming constant correlation)
    y = {}
    for i in np.unique(clusters):
        j = clusters == i
        if j.sum() == 1:
            y[i] = X[:, j][:, 0]
        else:
            Y = X[:, j]
            V = np.cov(Y.T)
            w = minimum_variance_weights_for_correlated_assets(V)
            y[i] = np.sum(Y * w, axis=1)
    y = pd.DataFrame(y)

    # Sharpe ratios
    SRs = y.mean() / y.std()

    return SRs.var(), SRs, y


def moments_Mk(k, *, rho=0):
    """Compute moments of M_k = Max(Z_1, Z_2, ..., Z_k), where Z_i are i.i.d. N(0,1).

    The density is f_M(x) = k φ(x) Φ(x)^(k-1), so the moments are
    E[M_k^r] = k * E[ Z^r * Φ(Z)^(k-1) ].

    For the correlated case, we assume equi-correlation:
    Zᵢ = √ρX + √(1-ρ)Yᵢ, M = √ρX + √(1-ρ)Max(Yᵢ)

    Args:
        k: int, number of variables
        rho: float, correlation coefficient

    Returns:
        Tuple of (Ez, Ez2, var):
        - Ez: float, expectation of M_k
        - Ez2: float, expectation of M_k^2
        - var: float, variance of M_k
    """
    Phi = scipy.stats.norm.cdf
    Ez = E_under_normal(lambda z: k * z * (Phi(z) ** (k - 1)))
    Ez2 = E_under_normal(lambda z: k * z * z * (Phi(z) ** (k - 1)))
    var = Ez2 - Ez**2

    Ez = (1 - rho) * Ez
    var = rho + (1 - rho) * var
    Ez2 = var + Ez**2

    return Ez, Ez2, var


def sharpe_ratio_variance(
    SR: float,
    T: int,
    *,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    K: int = 1,
) -> float:
    """Compute the asymptotic variance of the Sharpe ratio.

    Args:
        SR: float, Sharpe ratio
        T: int, number of observations
        gamma3: float, skewness
        gamma4: float, (non-excess) kurtosis
        rho: float, autocorrelation
        K: int, number of strategies whose Sharpe ratios we take the maximum of

    Returns:
        float, the variance of the Sharpe ratio
    """
    A = 1
    B = rho / (1 - rho)
    C = rho**2 / (1 - rho**2)
    a = A + 2 * B
    b = A + B + C
    c = A + 2 * C
    V = (a * 1 - b * gamma3 * SR + c * (gamma4 - 1) / 4 * SR**2) / T
    return V * moments_Mk(K)[2]


def variance_of_the_maximum_of_k_Sharpe_ratios(number_of_trials: int, variance: float) -> float:
    """Compute the variance of the maximum of K Sharpe ratios."""
    return variance * moments_Mk(number_of_trials)[2]


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

    See FDR_critical_value for the computations.

    Args:
        q: float, FDR level
        SR0: float, Sharpe ratio under H0
        SR1: float, Sharpe ratio under H1
        p_H1: float, probability that H1 is true
        T: int, number of observations
        gamma3: float, skewness
        gamma4: float, (non-excess) kurtosis
        rho: float, autocorrelation
        K: int, number of strategies (K=1 for FDR control; K>1 for FWER-FDR)

    Returns:
        Tuple of (alpha, beta, SR_c, q_hat):
        - alpha: float, significance level, P[SR>SR_c|H0]
        - beta: float, type II error, P[SR<=SR_c|H1]; power is 1-beta
        - SR_c: float, critical value
        - q_hat: float, estimated FDR; should be close to q
    """
    # Z_inv = scipy.stats.norm.ppf
    Z = scipy.stats.norm.cdf

    s0 = math.sqrt(sharpe_ratio_variance(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K))
    s1 = math.sqrt(sharpe_ratio_variance(SR1, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K))
    SRc = FDR_critical_value(q, SR0, SR1, s0, s1, p_H1)

    beta = Z((SRc - SR1) / s1)
    alpha = q / (1 - q) * p_H1 / (1 - p_H1) * (1 - beta)
    q_hat = 1 / (1 + (1 - beta) / alpha * p_H1 / (1 - p_H1))

    return alpha, beta, SRc, q_hat


def expected_maximum_sharpe_ratio(number_of_trials: int, variance: float, SR0: float = 0) -> float:
    """Compute expected maximum Sharpe ratio.

    Args:
        number_of_trials: int, number of trials
        variance: float, variance of the Sharpe ratios
        SR0: float, baseline Sharpe ratio to add

    Returns:
        float, expected maximum Sharpe ratio
    """
    return SR0 + (
        np.sqrt(variance)
        * (
            (1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / number_of_trials)
            + np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / number_of_trials / np.exp(1))
        )
    )


def minimum_track_record_length(
    SR: float,
    SR0: float,
    *,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Compute minimum track record length for significance.

    Computes the minimum track record length for the Sharpe ratio to be
    significantly greater than SR0, at the confidence level alpha.

    Args:
        SR: float, observed Sharpe ratio
        SR0: float, Sharpe ratio under H0
        gamma3: float, skewness
        gamma4: float, (non-excess) kurtosis
        rho: float, autocorrelation
        alpha: float, confidence level

    Returns:
        float, minimum track record length
    """
    var = sharpe_ratio_variance(SR0, T=1, gamma3=gamma3, gamma4=gamma4, rho=rho, K=1)
    return var * (scipy.stats.norm.ppf(1 - alpha) / (SR - SR0)) ** 2


def make_expectation_gh(n_nodes=200):
    """Create expectation function via Gauss–Hermite quadrature.

    Computes E[g(Z)] = (1/√π) * Σ w_i * g(√2 * t_i) where Z ~ N(0,1)
    and (t_i, w_i) are GH nodes/weights.

    Args:
        n_nodes: int, number of nodes

    Returns:
        Function to compute the expectation, f↦E[f(X)]
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    scale = np.sqrt(2.0)
    norm = 1.0 / np.sqrt(np.pi)
    x = scale * nodes

    def E(g):
        vals = g(x)
        return norm * np.dot(weights, vals)

    return E


E_under_normal = make_expectation_gh(n_nodes=200)


def adjusted_p_values_bonferroni(ps: np.ndarray) -> np.ndarray:
    """Adjust p-values using the Bonferroni correction to control FWER.

    Args:
        ps: np.ndarray, p-values

    Returns:
        np.ndarray, adjusted p-values
    """
    M = len(ps)
    return np.minimum(1, M * ps)


def adjusted_p_values_sidak(ps: np.ndarray) -> np.ndarray:
    """Adjust p-values using the Šidák correction to control FWER.

    Args:
        ps: np.ndarray, p-values

    Returns:
        np.ndarray, adjusted p-values
    """
    M = len(ps)
    return 1 - (1 - ps) ** M


def adjusted_p_values_holm(ps: np.ndarray, *, variant: str = "bonferroni") -> np.ndarray:
    """Adjust p-values using the Holm correction to control FWER.

    Args:
        ps: np.ndarray, p-values
        variant: str, variant of the Holm correction (bonferroni or sidak)

    Returns:
        np.ndarray, adjusted p-values
    """
    assert variant in ["bonferroni", "sidak"]
    i = np.argsort(ps)
    M = len(ps)
    p_adjusted = np.zeros(M)
    previous = 0
    for j, idx in enumerate(i):
        if variant == "bonferroni":
            candidate = min(1, ps[idx] * (M - j))
        else:
            candidate = 1 - (1 - ps[idx]) ** (M - j)
        p_adjusted[idx] = max(previous, candidate)
        previous = p_adjusted[idx]
    return p_adjusted


def FDR_critical_value(q: float, SR0: float, SR1: float, sigma0: float, sigma1: float, p_H1: float) -> float:
    """Compute critical value c such that P[H=0|X_H>c] = q.

    Given H ~ Bern(p₁), X₀ ~ N(μ₀,σ₀²), X₁ ~ N(μ₁,σ₁²),
    compute c such that P[H=0|X_H>c] = q.

    Args:
        q: float, desired FDR
        SR0: float, mean of X0
        SR1: float, mean of X1
        sigma0: float, standard deviation of X0
        sigma1: float, standard deviation of X1
        p_H1: float, probability of H=1

    Returns:
        c: float, critical value
    """
    assert SR0 < SR1
    assert 0 < q < 1
    assert 0 < p_H1 < 1
    assert 0 < sigma0
    assert 0 < sigma1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")

        def f(c):
            a = 1 / (
                1
                + scipy.stats.norm.sf((c - SR1) / sigma1) / scipy.stats.norm.sf((c - SR0) / sigma0) * p_H1 / (1 - p_H1)
            )
            return np.where(np.isfinite(a), a, 0)

        if f(-10) < q:  # Solution outside of the search interval
            return -np.inf

        if (f(-10) - q) * (f(10) - q) > 0:  # No solution, for instance if σ₀≫σ₁ and q small
            return np.nan

        return scipy.optimize.brentq(lambda c: f(c) - q, -10, 10)


def critical_sharpe_ratio(
    SR0: float,
    T: int,
    *,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    alpha: float = 0.05,
    K: int = 1,
) -> float:
    """Compute critical value for the test H0: SR=SR0 vs H1: SR>SR0.

    Args:
        SR0: float, Sharpe ratio under H0
        T: int, number of observations
        gamma3: float, skewness
        gamma4: float, (non-excess) kurtosis
        rho: float, autocorrelation
        alpha: float, confidence level
        K: int, number of strategies (larger K means smaller variance)

    Returns:
        float, critical value
    """
    variance = sharpe_ratio_variance(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    return SR0 + scipy.stats.norm.ppf(1 - alpha) * math.sqrt(variance)


def probabilistic_sharpe_ratio(
    SR: float,
    SR0: float,
    *,
    variance: float = None,
    T: int = None,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    K: int = 1,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR).

    This is 1-p, where p is the p-value of the test H0: SR=SR0 vs H1: SR>SR0.
    It can be interpreted as a Sharpe ratio "on a probability scale", i.e., in [0,1].

    Note: In case of multiple testing, SR0 is expected to be already adjusted;
    this function will only adjust the variance.

    Args:
        SR: float, observed Sharpe ratio
        SR0: float, Sharpe ratio under H0
        variance: float, optional variance (provide instead of T, gamma3, etc.)
        T: int, number of observations
        gamma3: float, skewness
        gamma4: float, (non-excess) kurtosis
        rho: float, autocorrelation
        K: int, number of strategies (larger K means smaller variance)

    Returns:
        float, probabilistic Sharpe ratio
    """
    if variance is None:
        variance = sharpe_ratio_variance(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    else:
        assert T is None, "Provide either the variance or (T, gamma3, gamma4, rho)"
    return scipy.stats.norm.cdf((SR - SR0) / math.sqrt(variance))


def sharpe_ratio_power(
    SR0: float,
    SR1: float,
    T: int,
    *,
    gamma3: float = 0.0,
    gamma4: float = 3.0,
    rho: float = 0.0,
    alpha: float = 0.05,
    K: int = 1,
) -> float:
    """Compute power (1-β) of the test H0: SR=SR0 vs H1: SR=SR1.

    Remarks:
    - To compute the power, we need to know more about the alternative hypothesis:
      SR1 could be the average Sharpe ratio of strategies with positive excess returns
    - "Power" is the same thing as "recall" in classification:
      Power = P[reject H0 | H1] = TP / (TP + FN) = recall

    Args:
        SR0: float, Sharpe ratio under H0
        SR1: float, Sharpe ratio under H1
        T: int, number of observations
        gamma3: float, skewness
        gamma4: float, (non-excess) kurtosis
        rho: float, autocorrelation
        alpha: float, confidence level
        K: int, number of strategies (larger K means smaller variance)

    Returns:
        float, power
    """
    critical_SR = critical_sharpe_ratio(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, alpha=alpha)
    variance = sharpe_ratio_variance(SR1, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    beta = scipy.stats.norm.cdf((critical_SR - SR1) / math.sqrt(variance))
    return 1 - beta


def generate_autocorrelated_non_gaussian_data(N, n, SR0=0, name="gaussian", rho=None, gaussian_autocorrelation=0):
    """Generate autocorrelated non-Gaussian data.

    Args:
        N: int, number of rows
        n: int, number of columns
        SR0: float, target Sharpe ratio
        name: str, distribution type (gaussian, mild, moderate, severe)
        rho: float, autocorrelation coefficient (optional)
        gaussian_autocorrelation: float, autocorrelation for Gaussian case

    Returns:
        np.ndarray, generated data matrix
    """
    if rho is None:
        # With the distributions we consider the autocorrelation is almost the same.
        rho = gaussian_autocorrelation

    shape = (N, n)

    # Marginal distribution: ppf
    R = 10_000
    marginal = generate_non_gaussian_data(R, 1, SR0=SR0, name=name)[:, 0]
    ppf = scipy.interpolate.interp1d(ppoints(R), sorted(marginal), fill_value="extrapolate")

    # AR(1) processes
    X = np.random.normal(size=shape)
    for i in range(1, shape[0]):
        X[i, :] = rho * X[i - 1, :] + np.sqrt(1 - rho**2) * X[i, :]

    # Convert the margins to uniform, with the Gaussian cdf
    X = scipy.stats.norm.cdf(X)

    # Convert the uniforms to the target margins, using the ppf
    X = ppf(X)

    return X


def get_random_correlation_matrix(
    number_of_trials: int = 100,
    effective_number_of_trials: int = 10,
    number_of_observations: int = 200,
    noise: float = 0.1,
):
    """Generate a correlation matrix with a block structure.

    Args:
        number_of_trials: int, number of time series to generate
        effective_number_of_trials: int, number of clusters
        number_of_observations: int, number of observations to generate
        noise: float, noise level

    Returns:
        Tuple of (C, X, clusters):
        - C: np.ndarray, square, a correlation matrix
        - X: np.ndarray, matrix of observations
        - clusters: np.ndarray, cluster assignment for each trial (column)
    """
    while True:
        block_positions = (
            [0]
            + sorted(np.random.choice(number_of_trials, effective_number_of_trials - 1, replace=True))
            + [number_of_trials]
        )
        block_sizes = np.diff(block_positions)
        if np.all(block_sizes > 0):
            break

    clusters = np.array([block_number for block_number, size in enumerate(block_sizes) for _ in range(size)])
    X0 = np.random.normal(size=(number_of_observations, effective_number_of_trials))
    X = np.zeros(shape=(number_of_observations, number_of_trials))
    for i, cluster in enumerate(clusters):
        X[:, i] = X0[:, cluster] + noise * np.random.normal(size=number_of_observations)
    C = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(C, 1)  # rounding errors
    C = np.clip(C, -1, 1)
    return C, X, clusters


def generate_non_gaussian_data(
    nr: int,
    nc: int,
    *,
    SR0: float = 0,
    name: str = "severe",
) -> np.ndarray:
    """Generate non-Gaussian data.

    Args:
        nr: int, number of rows
        nc: int, number of columns
        SR0: float, the target Sharpe ratio
        name: str, distribution (gaussian, mild, moderate, severe)

    Returns:
        X: np.ndarray, matrix of observations, shape (nr, nc)
    """
    configs = {
        "gaussian": (0, 0, 0.015, 0.010),
        "mild": (0.04, -0.03, 0.015, 0.010),
        "moderate": (0.03, -0.045, 0.020, 0.010),
        "severe": (0.02, -0.060, 0.025, 0.010),
    }
    assert name in configs

    def mixture_variance(p_tail, mu_tail, sigma_tail, mu_core, sigma_core):
        w = 1.0 - p_tail
        mu = w * mu_core + p_tail * mu_tail
        m2 = w * (sigma_core**2 + mu_core**2) + p_tail * (sigma_tail**2 + mu_tail**2)
        return m2 - mu**2

    def gen_with_true_SR0(reps, T, cfg, SR0):
        p, mu_tail, sig_tail, sig_core = cfg
        # Zero-mean baseline mixture (choose mu_core so mean=0)
        mu_core0 = -p * mu_tail / (1.0 - p)
        std0 = np.sqrt(mixture_variance(p, mu_tail, sig_tail, mu_core0, sig_core))
        mu_shift = SR0 * std0  # sets population Sharpe to SR0, preserves skew/kurt
        mask = np.random.uniform(size=(reps, T)) < p
        X = np.random.normal(mu_core0 + mu_shift, sig_core, size=(reps, T))
        X[mask] = np.random.normal(mu_tail + mu_shift, sig_tail, size=mask.sum())
        return X

    return gen_with_true_SR0(nr, nc, configs[name], SR0)


def autocorrelation(X):
    """Compute mean autocorrelation across columns.

    Args:
        X: np.ndarray, data matrix

    Returns:
        float, mean autocorrelation
    """
    nr, nc = X.shape
    ac = np.zeros(nc)
    for i in range(nc):
        ac[i] = np.corrcoef(X[1:, i], X[:-1, i])[0, 1]
    return ac.mean()


def pFDR(
    p_H1: float,
    alpha: float,
    beta: float,
) -> float:
    """Compute posterior probability of H0, given that SR>SR_c.

    Remarks:
    - Needs beta=1-power (which needs SR1), and p[H1]
    - This does not use the observed Sharpe ratio, only the critical value

    Args:
        p_H1: float, probability that H1 is true
        alpha: float, confidence level
        beta: float, 1 - power, i.e., type II error

    Returns:
        float, posterior probability of H0
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
    """Compute posterior probability of H0, given that SR>SR_obs.

    In case of multiple testing, SR0 and SR1 are expected to be already adjusted;
    this function will only adjust the variance.

    Args:
        SR: float, observed Sharpe ratio
        SR0: float, Sharpe ratio under H0
        SR1: float, Sharpe ratio under H1
        T: int, number of observations
        p_H1: float, probability that H1 is true
        gamma3: float, skewness
        gamma4: float, (non-excess) kurtosis
        rho: float, autocorrelation
        K: int, number of strategies (larger K means smaller variance)

    Returns:
        float, posterior probability of H0
    """
    p0 = 1 - probabilistic_sharpe_ratio(SR, SR0, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    p1 = 1 - probabilistic_sharpe_ratio(SR, SR1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    p_H0 = 1 - p_H1
    return p0 * p_H0 / (p0 * p_H0 + p1 * p_H1)


def number_of_clusters(
    C: np.ndarray,
    *,
    retries: int = 10,
    max_clusters: int = 100,
    plot: bool = False,
) -> tuple[int, pd.Series, np.ndarray]:
    """Compute the optimal number of clusters from a correlation matrix.

    Algorithm in section 8.1 of [1], without recursive re-clustering:
    - Convert the correlation matrix into a distance matrix
    - Using the columns of the distance matrix as features, run k-means
      for all k, and compute the "quality" of the clustering
    - Keep the clustering with the highest quality
    The quality is computed as mean of silhouette scores / std deviation.

    References:
        [1] Detection of false investment strategies using unsupervised learning
            M. Lopez de Prado (2018)
            https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017

    Args:
        C: np.ndarray, square, a correlation matrix
        retries: int, number of times to run the k-means algorithm
        max_clusters: int, maximum number of clusters to consider
        plot: bool, whether to plot the quality of the clustering

    Returns:
        Tuple of (number_of_clusters, qualities, clusters):
        - number_of_clusters: int, the optimal number of clusters
        - qualities: pd.Series, quality of clusterings by number of clusters
        - clusters: np.ndarray, cluster assignment for optimal clustering
    """
    # Check this looks like a correlation matrix
    assert isinstance(C, np.ndarray)
    assert np.all(-1 <= C)
    assert np.all(C <= 1)
    assert np.all(np.diag(C) == 1)
    assert np.all(np.isfinite(C))

    max_clusters = min(max_clusters, C.shape[0] - 1)

    # Compute the distances
    D = np.sqrt((1 - C) / 2)
    assert np.all(np.isfinite(D))

    # For all values of k:
    # - run the k-means algorithm on D
    # - compute the silhouette score of each observation, S[i],
    # - compute the quality of the clustering, q = E[S]/Std[S]
    # - Do that several times and keep the maximum quality
    qualities = {}
    clusters = {}
    for k in range(2, max_clusters + 1):
        qualities[k] = -np.inf
        for _ in range(retries):
            kmeans = KMeans(n_clusters=k)
            # Use the distances as features
            kmeans.fit(D)
            labels = kmeans.labels_
            silhouette_vals = silhouette_samples(D, labels)
            q = silhouette_vals.mean() / silhouette_vals.std()
            if q > qualities[k]:
                qualities[k] = max(qualities[k], q)
                clusters[k] = labels
    qualities = pd.Series(qualities)
    number_of_clusters = qualities.idxmax()
    clusters = clusters[number_of_clusters]

    if plot:
        fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
        ax.plot(qualities)
        i = np.argmax(qualities)
        x, y = qualities.index[i], qualities.iloc[i]
        ax.scatter(x, y)
        ax.text(x, y, f"  {qualities.index[i]}", va="center", ha="left")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Quality")
        plt.show()

    return number_of_clusters, qualities, clusters
