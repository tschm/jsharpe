"""Tests for :mod:`jsharpe.sharpe.corrections` (FWER/FDR multiple-testing).

Covers the multiple-testing corrections and false-discovery-rate routines via
pinned values, Monte-Carlo calibration of the critical value, and behavioural
bounds/monotonicity — no comparison against a bundled reference implementation.
"""
# ruff: noqa: N802, N806

import numpy as np
import pytest

from jsharpe import (
    FDR_critical_value,
    adjusted_p_values_bonferroni,
    adjusted_p_values_holm,
    adjusted_p_values_sidak,
    control_for_FDR,
    oFDR,
    pFDR,
)

# ---- FWER corrections -------------------------------------------------------


def test_adjusted_p_values_methods():
    """Bonferroni/Sidak/Holm corrections match their definitions and stay in [0, 1]."""
    ps = np.array([0.01, 0.02, 0.5, 0.9])
    M = len(ps)

    bonf = adjusted_p_values_bonferroni(ps)
    assert np.allclose(bonf, np.minimum(1, M * ps))

    sidak = adjusted_p_values_sidak(ps)
    assert np.allclose(sidak, 1 - (1 - ps) ** M)
    # Sidak is never more conservative than Bonferroni.
    assert np.all(sidak <= bonf + 1e-12)

    holm_b = adjusted_p_values_holm(ps, variant="bonferroni")
    assert np.allclose(holm_b, np.array([0.04, 0.06, 1.0, 1.0]))
    assert np.all((holm_b >= 0) & (holm_b <= 1))

    holm_s = adjusted_p_values_holm(ps, variant="sidak")
    order = np.argsort(ps)
    out = np.zeros_like(ps)
    prev = 0.0
    for j, idx in enumerate(order):
        cand = 1 - (1 - ps[idx]) ** (M - j)
        out[idx] = max(prev, cand)
        prev = out[idx]
    assert np.allclose(holm_s, out)


def test_adjusted_p_values_holm_invalid_variant_raises():
    """An unknown Holm variant is rejected."""
    with pytest.raises(AssertionError):
        adjusted_p_values_holm(np.array([0.1, 0.2]), variant="unknown")


# ---- FDR critical value -----------------------------------------------------


def test_FDR_critical_value_is_calibrated():
    """The critical value achieves the requested FDR in a Monte-Carlo mixture simulation."""
    np.random.seed(0)
    errors = []
    for _ in range(50):
        q = np.random.uniform(0.05, 0.95)
        mu0, mu1 = sorted(np.random.uniform(size=2))
        sigma0, sigma1 = np.random.uniform(0.1, 1.0, size=2)
        p = np.random.uniform(0.05, 0.95)

        c = FDR_critical_value(q, mu0, mu1, sigma0, sigma1, p)
        if not np.isfinite(c):
            continue

        R = 200_000
        H = np.random.uniform(size=R) < p
        X = np.where(H, np.random.normal(mu1, sigma1, R), np.random.normal(mu0, sigma0, R))
        discoveries = c < X
        if discoveries.sum() == 0:
            continue
        fdp = np.sum((~H) & discoveries) / discoveries.sum()
        errors.append(abs(q - fdp))

    assert errors, "expected at least one finite critical value"
    assert np.mean(errors) < 0.05


def test_FDR_critical_value_edge_returns():
    """The out-of-interval (-inf) and no-root (NaN) branches are reachable."""
    # For c -> -inf, f(-10) ~ 1 - p; if q > 1 - p the function returns -inf.
    c = FDR_critical_value(0.85, 0.0, 1.0, 1.0, 1.0, 0.2)
    assert c == -np.inf

    # Highly imbalanced variances (sigma0 >> sigma1) with small q leave no root in [-10, 10].
    c2 = FDR_critical_value(0.05, 0.0, 1.0, 10.0, 0.1, 0.9)
    assert np.isnan(c2)


def test_FDR_critical_value_finite_solution_is_positive():
    """A well-posed configuration yields a finite, positive critical value."""
    c = FDR_critical_value(q=0.2, SR0=0.0, SR1=0.5, sigma0=0.2, sigma1=0.3, p_H1=0.1)
    assert np.isfinite(c)
    assert c > 0


# ---- control_for_FDR --------------------------------------------------------


def test_control_for_FDR_returns_calibrated_threshold():
    """control_for_FDR yields valid error rates, a positive threshold, and q_hat close to q."""
    q = 0.25
    alpha, beta, SR_c, q_hat = control_for_FDR(q, SR0=0.0, SR1=0.5, p_H1=0.10, T=24)
    assert 0.0 < alpha < 1.0
    assert 0.0 < beta < 1.0
    assert SR_c > 0.0
    assert q_hat == pytest.approx(q, abs=0.05)


def test_control_for_FDR_autocorrelation_widens_threshold():
    """Positive autocorrelation increases estimator variance, raising the critical Sharpe ratio."""
    _, _, sr_c_iid, _ = control_for_FDR(0.25, SR0=0.0, SR1=0.5, p_H1=0.10, T=24, rho=0.0)
    _, _, sr_c_ac, _ = control_for_FDR(0.25, SR0=0.0, SR1=0.5, p_H1=0.10, T=24, rho=0.3)
    assert sr_c_ac > sr_c_iid


# ---- posterior FDR ----------------------------------------------------------


def test_pFDR_pinned_value():
    """Posterior FDR matches the documented worked example."""
    assert round(pFDR(0.05, 0.05, 0.315), 3) == 0.581


def test_pFDR_bounds_and_grows_with_alpha():
    """Posterior FDR is a probability in (0, 1) that increases as the Type-I error rate alpha grows."""
    fdr = pFDR(p_H1=0.05, alpha=0.05, beta=0.3)
    assert 0.0 < fdr < 1.0
    assert pFDR(p_H1=0.05, alpha=0.20, beta=0.3) > fdr


def test_oFDR_pinned_value():
    """Observed FDR matches the documented worked example."""
    result = oFDR(SR=0.036 / 0.079, SR0=0, SR1=0.5, T=24, p_H1=0.05, gamma3=-2.448, gamma4=10.164)
    assert round(result, 3) == 0.306


def test_oFDR_bounds_and_falls_with_stronger_evidence():
    """Observed FDR is a probability in (0, 1) that decreases as the observed SR strengthens."""
    fdr_weak = oFDR(SR=0.3, SR0=0.0, SR1=0.5, T=24, p_H1=0.1)
    fdr_strong = oFDR(SR=0.8, SR0=0.0, SR1=0.5, T=24, p_H1=0.1)
    assert 0.0 < fdr_weak < 1.0
    assert fdr_strong < fdr_weak
