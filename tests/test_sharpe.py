"""Tests for Sharpe ratio functions.

This test module now uses a dedicated logger to provide detailed diagnostics
when running under pytest. Logging is kept lightweight and added mainly to
complex/numerically sensitive test paths to aid debugging without affecting
assertions.
"""

import logging
import math

import cvxpy as cp
import numpy as np
import pytest

try:
    from jsharpe import (
        FDR_critical_value,
        # additional imports for full coverage
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
        # merged imports
        ppoints,
        probabilistic_sharpe_ratio,
        robust_covariance_inverse,
        sharpe_ratio_power,
        sharpe_ratio_variance,
        variance_of_the_maximum_of_k_Sharpe_ratios,
    )
except ModuleNotFoundError:
    # Allow running tests with `uv run pytest` which uses an ephemeral env
    # that doesn't include the local project by default. Fallback to src/.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from jsharpe import (
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

# Set up module-level logger for detailed diagnostics
logger = logging.getLogger(__name__)

# Route any legacy print calls in this module to the logger for consistency
print = logger.debug  # type: ignore[assignment]


def test_effective_rank():
    """Test effective_rank computation."""
    np.random.seed(1)
    x = np.random.normal(size=(10, 3))
    C = np.corrcoef(x.T)
    effective_rank(C)  # Almost 3
    assert abs(effective_rank(np.eye(3)) - 3) < 1e-12
    C = np.array([[10, 1, 7], [1, 10, 8], [7, 8, 10]]) / 10
    assert abs(effective_rank(C[:2, :2]) - 2) < 0.02
    assert abs(effective_rank(C) - 1.84) < 0.01


def test_sharpe_ratio_variance():
    """Test sharpe_ratio_variance computation."""
    SR = 0.036 / 0.079
    var1 = sharpe_ratio_variance(SR=SR, gamma3=-2.448, gamma4=10.164, T=24)
    var2 = sharpe_ratio_variance(SR=SR, gamma3=0, gamma4=3, T=24)
    assert round(math.sqrt(var1), 3) == 0.329
    assert round(math.sqrt(var2), 3) == 0.214


def test_minimum_track_record_length():
    """Test minimum_track_record_length computation."""
    mtrl = minimum_track_record_length(SR=0.036 / 0.079, SR0=0, gamma3=-2.448, gamma4=10.164, alpha=0.05)
    assert round(mtrl, 3) == 13.029


def test_probabilistic_sharpe_ratio():
    """Test probabilistic_sharpe_ratio computation."""
    psr0 = probabilistic_sharpe_ratio(SR=0.036 / 0.079, SR0=0, T=24, gamma3=-2.448, gamma4=10.164)
    psr1 = probabilistic_sharpe_ratio(SR=0.036 / 0.079, SR0=0.1, T=24, gamma3=-2.448, gamma4=10.164)
    assert round(psr0, 3) == 0.987
    assert round(psr1, 3) == 0.939


def test_sharpe_ratio_power():
    """Test sharpe_ratio_power computation."""
    power = sharpe_ratio_power(SR0=0, SR1=0.5, T=24, gamma3=-2.448, gamma4=10.164)
    assert round(1 - power, 3) == 0.315


def test_pFDR():
    """Test pFDR computation."""
    assert round(pFDR(0.05, 0.05, 0.315), 3) == 0.581


def test_oFDR():
    """Test oFDR computation."""
    result = oFDR(SR=0.036 / 0.079, SR0=0, SR1=0.5, T=24, p_H1=0.05, gamma3=-2.448, gamma4=10.164)
    assert round(result, 3) == 0.306


def test_FDR_critical_value():
    """Test FDR_critical_value computation."""
    np.random.seed(0)
    r = dict()
    for _ in range(100):
        q = np.random.uniform()
        mu0 = np.random.uniform()
        mu1 = np.random.uniform()
        mu0, mu1 = min(mu0, mu1), max(mu0, mu1)
        sigma0 = np.random.uniform()
        sigma1 = np.random.uniform()
        p = np.random.uniform()

        R = 100_000
        H = np.random.uniform(size=R) < p
        X0 = np.random.normal(mu0, sigma0, size=R)
        X1 = np.random.normal(mu1, sigma1, size=R)
        X = np.where(H, X1, X0)
        c = FDR_critical_value(q, mu0, mu1, sigma0, sigma1, p)
        r["q"] = q
        r["mu0"] = mu0
        r["mu1"] = mu1
        r["sigma0"] = sigma0
        r["sigma1"] = sigma1
        r["p"] = p
        r["c"] = c
        r["FDP"] = np.sum((H == 0) & (X > c)) / (1e-100 + np.sum(X > c))

    logger.debug("FDR simulation snapshot: %s", r)
    np.isfinite(r["c"]) & (r["FDP"] > 0)
    assert np.abs(r["q"] - r["FDP"]).mean() < 1e-2


def test_FDR_critical_value_edge_returns():
    """Explicitly trigger -inf and NaN branches in FDR_critical_value."""
    # For c -> -inf, f(-10) ≈ 1 - p. If q > 1 - p then function returns -inf
    mu0, mu1 = 0.0, 1.0
    sigma0, sigma1 = 1.0, 1.0
    p = 0.2
    q = 0.85  # > 1 - p = 0.8
    c = FDR_critical_value(q, mu0, mu1, sigma0, sigma1, p)
    assert c == -np.inf

    # If q < 1 - p then no root in [-10, 10] and function returns NaN.
    # To ensure the no-root case, choose highly imbalanced variances (sigma0 >> sigma1)
    # so that f(c) stays near 1 across the interval.
    p2 = 0.9
    q2 = 0.05  # < 1 - p2 = 0.1
    sigma0b, sigma1b = 10.0, 0.1
    c2 = FDR_critical_value(q2, mu0, mu1, sigma0b, sigma1b, p2)
    assert np.isnan(c2)


def test_numeric_example():
    """Test numeric example with various Sharpe ratio computations."""
    for rho in [0, 0.2]:
        logger.debug("----------")
        mu = 0.036
        sigma = 0.079
        T = 24
        gamma3 = -2.448
        gamma4 = 10.164
        SR0 = 0
        SR1 = 0.5
        p_H1 = 0.10
        alpha = 0.10
        SR = mu / sigma
        print(f"SR0                    = {SR0:.3f}")
        print(f"SR1                    = {SR1:.3f}")
        print(f"μ                      = {mu:.3f}")
        print(f"σ                      = {sigma:.3f}")
        print(f"γ3                     = {gamma3:.3f}")
        print(f"γ4                     = {gamma4:.3f}")
        print(f"ρ                      = {rho:.3f}")
        print(f"T                      = {T}")
        print(f"SR                     = {SR:.3f}")

        var_ng = sharpe_ratio_variance(SR=mu / sigma, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T)
        var_g = sharpe_ratio_variance(SR=mu / sigma, gamma3=0, gamma4=3, T=T)
        print(f"σ_SR                   = {math.sqrt(var_ng):.3f} (non-Gaussian)")
        print(f"σ_SR                   = {math.sqrt(var_g):.3f} (Gaussian, iid)")

        mtrl = minimum_track_record_length(SR=mu / sigma, SR0=0, gamma3=gamma3, gamma4=gamma4, rho=rho, alpha=alpha)
        mtrl_01 = minimum_track_record_length(
            SR=mu / sigma, SR0=0.1, gamma3=gamma3, gamma4=gamma4, rho=rho, alpha=alpha
        )
        print(f"MinTRL                 = {mtrl:.3f}")
        print(f"MinTRL(SR0=.1)         = {mtrl_01:.3f}")

        psr_0 = probabilistic_sharpe_ratio(SR=mu / sigma, SR0=0, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho)
        psr_01 = probabilistic_sharpe_ratio(SR=mu / sigma, SR0=0.1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho)
        print(f"p = 1 - PSR(SR0=0)     = {1 - psr_0:.3f}")
        print(f"PSR(SR0=0)             = {psr_0:.3f}")
        print(f"PSR(SR0=.1)            = {psr_01:.3f}")
        print(f"SR0                    = {SR0:.3f}")

        sr_c_g = critical_sharpe_ratio(SR0, T, gamma3=0.0, gamma4=3.0, rho=0, alpha=alpha)
        sr_c_ng = critical_sharpe_ratio(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, alpha=alpha)
        print(f"SR_c                   = {sr_c_g:.3f} (Gaussian, iid)")
        # Note: unchanged if iid and SR0=0
        print(f"SR_c                   = {sr_c_ng:.3f} (non-Gaussian)")
        print(f"SR1                    = {SR1:.3f}")

        power = sharpe_ratio_power(SR0=SR0, SR1=SR1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, alpha=alpha)
        print(f"Power = 1 - β          = {power:.3f}")
        print(f"β                      = {1 - power:.3f}")
        print(f"P[H1]                  = {p_H1:.3f}")

        pfdr_val = pFDR(p_H1, alpha, 1 - power)
        ofdr_val = oFDR(SR=mu / sigma, SR0=SR0, SR1=SR1, T=T, p_H1=p_H1, gamma3=gamma3, gamma4=gamma4, rho=rho)
        print(f"pFDR = P[H0|SR>SR_c]   = {pfdr_val:.3f}")
        print(f"oFDR = P[H0|SR>SR_obs] = {ofdr_val:.3f}")

        print("\nFWER")
        number_of_trials = 10
        variance = 0.1
        E_max_SR = expected_maximum_sharpe_ratio(number_of_trials, variance)
        SR0_adj = SR0 + E_max_SR
        SR1_adj = SR1 + E_max_SR

        sigma_SR0_adj_single = math.sqrt(sharpe_ratio_variance(SR=SR0_adj, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T))
        sigma_SR0_adj = math.sqrt(
            sharpe_ratio_variance(SR=SR0_adj, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T, K=number_of_trials)
        )
        sigma_SR1_adj_single = math.sqrt(sharpe_ratio_variance(SR=SR1_adj, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T))
        sigma_SR1_adj = math.sqrt(
            sharpe_ratio_variance(SR=SR1 + SR0_adj, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T, K=number_of_trials)
        )
        DSR_single = probabilistic_sharpe_ratio(SR, SR0=SR0_adj, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho)
        DSR = probabilistic_sharpe_ratio(
            SR, SR0=SR0_adj, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=number_of_trials
        )
        print(f"K                         = {number_of_trials}")
        print(f"Var[SR_k]                 = {variance:.3f}  (only used to compute E[max SR])")
        print(f"E[max SR]                 = {E_max_SR:.3f}")
        print(f"SR0_adj = SR0 + E[max SR] = {SR0_adj:.3f}")
        print(f"SR1_adj = SR1 + E[max SR] = {SR1_adj:.3f}")
        print(f"σ_SR0_adj                 = {sigma_SR0_adj:.3f}")
        print(f"σ_SR1_adj                 = {sigma_SR1_adj:.3f}")
        print(f"DSR                       = {DSR:.3f}")

        ofdr_fwer = oFDR(
            SR=mu / sigma,
            SR0=SR0_adj,
            SR1=SR1 + SR0_adj,
            T=T,
            p_H1=p_H1,
            gamma3=gamma3,
            gamma4=gamma4,
            rho=rho,
            K=number_of_trials,
        )
        print(f"oFDR = P[H0|SR>SR_obs]    = {ofdr_fwer:.3f}")

        print(f"σ_SR0_adj (K=1)              = {sigma_SR0_adj_single:.3f}")
        print(f"σ_SR1_adj (K=1)              = {sigma_SR1_adj_single:.3f}")
        print(f"DSR (K=1)                    = {DSR_single:.3f}")

        ofdr_k1 = oFDR(
            SR=mu / sigma, SR0=SR0_adj, SR1=SR1 + SR0_adj, T=T, p_H1=p_H1, gamma3=gamma3, gamma4=gamma4, rho=rho
        )
        print(f"oFDR = P[H0|SR>SR_obs] (K=1) = {ofdr_k1:.3f}")

        print("\nFDR")
        q = 0.25
        alpha_, beta_, SR_c, q_hat = control_for_FDR(
            q, SR0=SR0, SR1=SR1, p_H1=p_H1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho
        )
        print(f"P[H1]                  = {p_H1:.3f}")
        print(f"q                      = {q:.3f}")
        print(f"α                      = {alpha_:.4f}")
        print(f"β                      = {beta_:.3f}")
        print(f"SR_c                   = {SR_c:.3f}")

        var_sr0 = sharpe_ratio_variance(SR=SR0, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T)
        var_sr1 = sharpe_ratio_variance(SR=SR1, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T)
        print(f"σ_SR0                  = {math.sqrt(var_sr0):.3f}")
        print(f"σ_SR1                  = {math.sqrt(var_sr1):.3f}")

        print("\nFWER-FDR")
        alpha_W, beta_W, SR_c_W, q_hat_W = control_for_FDR(
            q, SR0=SR0 + SR0_adj, SR1=SR1 + SR0_adj, p_H1=p_H1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho
        )
        alpha_, beta_, SR_c, q_hat = control_for_FDR(
            q,
            SR0=SR0 + SR0_adj,
            SR1=SR1 + SR0_adj,
            p_H1=p_H1,
            T=T,
            gamma3=gamma3,
            gamma4=gamma4,
            rho=rho,
            K=number_of_trials,
        )

        var_sr0_fwer = sharpe_ratio_variance(
            SR=SR0 + SR0_adj, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T, K=number_of_trials
        )
        var_sr1_fwer = sharpe_ratio_variance(
            SR=SR1 + SR0_adj, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T, K=number_of_trials
        )
        print(f"σ_SR0                  = {math.sqrt(var_sr0_fwer):.3f}")
        print(f"σ_SR1                  = {math.sqrt(var_sr1_fwer):.3f}")
        print(f"α                      = {alpha_:.5f}")
        print(f"β                      = {beta_:.3f}")
        print(f"SR_c                   = {SR_c:.3f}")

        var_sr0_k1 = sharpe_ratio_variance(SR=SR0 + SR0_adj, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T)
        var_sr1_k1 = sharpe_ratio_variance(SR=SR1 + SR0_adj, gamma3=gamma3, gamma4=gamma4, rho=rho, T=T)
        print(f"σ_SR0 (K=1)          = {math.sqrt(var_sr0_k1):.3f}")
        print(f"σ_SR1 (K=1)          = {math.sqrt(var_sr1_k1):.3f}")
        print(f"α (K=1)              = {alpha_W:.5f}")
        print(f"β (K=1)              = {beta_W:.3f}")
        print(f"SR_c (K=1)           = {SR_c_W:.3f}")


# ---- merged from test_ppoints.py ----


def test_ppoints_default_large_n():
    """Default behavior when n > 10 (a defaults to 0.5)."""
    n = 20
    # default a = 0.5 when n > 10
    expected = np.linspace(1 - 0.5, n - 0.5, n) / (n + 1 - 2 * 0.5)
    x = ppoints(n)
    assert np.allclose(x, expected)
    assert np.all(x > 0) and np.all(x < 1)
    # uniform spacing
    diffs = np.diff(x)
    assert np.allclose(diffs, diffs[0])


def test_ppoints_default_small_n():
    """Default behavior when n <= 10 (a defaults to 3/8)."""
    n = 10
    # default a = 3/8 when n <= 10
    a = 3 / 8
    expected = np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)
    x = ppoints(n)
    assert np.allclose(x, expected)
    # uniform spacing
    diffs = np.diff(x)
    assert np.allclose(diffs, diffs[0])


def test_ppoints_custom_a_zero():
    """Custom a=0.0 should exclude boundaries 0 and 1."""
    n = 5
    a = 0.0
    expected = np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)
    x = ppoints(n, a=a)
    assert np.allclose(x, expected)
    # should exclude 0 and 1 for a=0
    assert x[0] == pytest.approx(1 / (n + 1))
    assert x[-1] == pytest.approx(n / (n + 1))


def test_ppoints_custom_a_one_includes_boundaries():
    """Custom a=1.0 includes both boundaries 0 and 1 by formula design."""
    n = 5
    a = 1.0
    # This includes both boundaries 0 and 1 by design of the formula
    expected = np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)
    x = ppoints(n, a=a)
    assert np.allclose(x, expected)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)


def test_ppoints_invalid_a_raises():
    """Invalid a outside [0, 1] should raise AssertionError."""
    n = 7
    with pytest.raises(AssertionError):
        ppoints(n, a=-0.01)
    with pytest.raises(AssertionError):
        ppoints(n, a=1.01)


# ---- merged from test_minimum_variance.py ----


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


def test_make_expectation_gh_moments():
    """Gauss–Hermite expectation should reproduce standard normal moments."""
    E = make_expectation_gh(n_nodes=50)
    # E[1] = 1
    assert E(lambda x: np.ones_like(x)) == pytest.approx(1.0, rel=1e-10, abs=1e-10)
    # E[Z] = 0
    assert E(lambda x: x) == pytest.approx(0.0, abs=1e-8)
    # E[Z^2] = 1
    assert E(lambda x: x**2) == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_adjusted_p_values_methods():
    """Test Bonferroni/Šidák/Holm adjusted p-values for simple inputs."""
    ps = np.array([0.01, 0.02, 0.5, 0.9])
    M = len(ps)

    bonf = adjusted_p_values_bonferroni(ps)
    assert np.allclose(bonf, np.minimum(1, M * ps))

    sidak = adjusted_p_values_sidak(ps)
    assert np.allclose(sidak, 1 - (1 - ps) ** M)

    # Holm-Bonferroni expected values for this ordering
    holm_b = adjusted_p_values_holm(ps, variant="bonferroni")
    expected_holm_b = np.array([0.04, 0.06, 1.0, 1.0])
    assert np.allclose(holm_b, expected_holm_b)
    assert np.all((holm_b >= 0) & (holm_b <= 1))

    # Holm-Šidák variant: compute manually by definition for this simple case
    holm_s = adjusted_p_values_holm(ps, variant="sidak")
    order = np.argsort(ps)
    out = np.zeros_like(ps)
    prev = 0.0
    for j, idx in enumerate(order):
        cand = 1 - (1 - ps[idx]) ** (M - j)
        out[idx] = max(prev, cand)
        prev = out[idx]
    assert np.allclose(holm_s, out)


def test_variance_of_maximum_monotonic_in_k():
    """Variance of the maximum Sharpe ratio increases with number of trials."""
    base_var = 0.1
    v1 = variance_of_the_maximum_of_k_Sharpe_ratios(1, base_var)
    v5 = variance_of_the_maximum_of_k_Sharpe_ratios(5, base_var)
    assert v5 > v1


def test_get_random_correlation_matrix_and_effective_rank():
    """Generate a clustered correlation matrix and validate shapes, symmetry, labels, and effective rank."""
    np.random.seed(1)
    C, X, clusters = get_random_correlation_matrix(
        number_of_trials=30, effective_number_of_trials=5, number_of_observations=200, noise=0.05
    )
    # Shapes
    assert C.shape == (30, 30)
    assert X.shape == (200, 30)
    assert clusters.shape == (30,)
    # Symmetry and diagonal ones
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 1)
    # Cluster labels within range
    assert clusters.min() >= 0 and clusters.max() < 5
    # Effective rank between 1 and n
    er = effective_rank(C)
    assert 1 <= er <= 30


essr_tol = 0.03


def test_generate_non_gaussian_data_sr0_shift():
    """Generated non-Gaussian data should have sample SR close to requested SR0."""
    np.random.seed(0)
    for name in ["gaussian", "mild", "moderate", "severe"]:
        X = generate_non_gaussian_data(4000, 1, SR0=0.2, name=name)
        m = float(X.mean())
        s = float(X.std(ddof=0))
        sr = m / s
        assert sr == pytest.approx(0.2, abs=0.05)


def test_generate_autocorrelated_non_gaussian_data_and_autocorrelation():
    """Generate AR(1) non-Gaussian data and verify mean autocorrelation matches target rho."""
    np.random.seed(0)
    N, n = 800, 4
    rho = 0.3
    X = generate_autocorrelated_non_gaussian_data(N, n, SR0=0.0, name="mild", rho=rho, gaussian_autocorrelation=0.0)
    assert X.shape == (N, n)
    ac = autocorrelation(X)
    # Allow some tolerance due to finite sample
    assert ac == pytest.approx(rho, abs=0.08)


def test_probabilistic_sharpe_ratio_with_variance_and_T_conflict_raises():
    """Providing both variance and T should raise an assertion error."""
    with pytest.raises(AssertionError):
        probabilistic_sharpe_ratio(SR=0.5, SR0=0.0, variance=0.04, T=24)
