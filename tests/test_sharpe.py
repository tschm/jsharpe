"""Tests for Sharpe ratio functions."""

import math

import numpy as np
import pandas as pd

from src.jsharpe.sharpe import (
    FDR_critical_value,
    control_for_FDR,
    critical_sharpe_ratio,
    effective_rank,
    expected_maximum_sharpe_ratio,
    minimum_track_record_length,
    oFDR,
    pFDR,
    probabilistic_sharpe_ratio,
    sharpe_ratio_power,
    sharpe_ratio_variance,
)


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
    r = []
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
        r.append(
            {
                "q": q,
                "mu0": mu0,
                "mu1": mu1,
                "sigma0": sigma0,
                "sigma1": sigma1,
                "p": p,
                "c": c,
                "FDP": np.sum((H == 0) & (X > c)) / (1e-100 + np.sum(X > c)),
            }
        )
    r = pd.DataFrame(r)
    i = np.isfinite(r["c"]) & (r["FDP"] > 0)
    assert np.abs(r["q"][i] - r["FDP"][i]).mean() < 1e-2
    # plt.scatter( r['q'][i], r['FDP'][i] )  # Straight line


def test_numeric_example():
    """Test numeric example with various Sharpe ratio computations."""
    for rho in [0, 0.2]:
        print("----------")
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
