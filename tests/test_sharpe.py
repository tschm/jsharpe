import math

import numpy as np
import pandas as pd

from src.jsharpe.sharpe import effective_rank, minimum_track_record_length, sharpe_ratio_variance, \
    probabilistic_sharpe_ratio, sharpe_ratio_power, pFDR, oFDR, FDR_critical_value, control_for_FDR, \
    critical_sharpe_ratio, expected_maximum_sharpe_ratio


def test_effective_rank():
    np.random.seed(1)
    x = np.random.normal( size=(10,3) )
    C = np.corrcoef(x.T)
    effective_rank(C)  # Almost 3
    assert abs( effective_rank( np.eye(3) ) - 3 ) < 1e-12
    C = np.array([[10,1,7],[1,10,8],[7,8,10]]) / 10
    assert abs( effective_rank(C[:2,:2]) - 2 ) < .02
    assert abs( effective_rank(C) - 1.84 ) < .01


def test_sharpe_ratio_variance():
    assert round( math.sqrt( sharpe_ratio_variance( SR = .036 / .079, gamma3 = -2.448, gamma4 = 10.164, T = 24 ) ), 3 ) == .329
    assert round( math.sqrt( sharpe_ratio_variance( SR = .036 / .079, gamma3 = 0,      gamma4 = 3,      T = 24 ) ), 3 ) == .214

def test_minimum_track_record_length():
    assert round( minimum_track_record_length( SR = .036 / .079, SR0 = 0, gamma3 = -2.448, gamma4 = 10.164, alpha = .05 ), 3 ) == 13.029


def test_probabilistic_sharpe_ratio():
    assert round( probabilistic_sharpe_ratio( SR = .036 / .079, SR0 = 0,  T = 24, gamma3 = -2.448, gamma4 = 10.164), 3 ) == .987
    assert round( probabilistic_sharpe_ratio( SR = .036 / .079, SR0 = .1, T = 24, gamma3 = -2.448, gamma4 = 10.164), 3 ) == .939

def test_sharpe_ratio_power():
    assert round( 1 - sharpe_ratio_power( SR0=0, SR1 = .5, T = 24, gamma3 = -2.448, gamma4 = 10.164 ), 3 ) == .315

def test_pFDR():
    assert round( pFDR( .05, .05, .315 ), 3 ) == .581

def test_oFDR():
    assert round( oFDR( SR = .036 / .079, SR0=0, SR1=.5, T=24, p_H1=.05, gamma3 = -2.448, gamma4 = 10.164 ), 3 ) == .306


def test_FDR_critical_value():
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
        H = np.random.uniform(size = R) < p
        X0 = np.random.normal(mu0, sigma0, size = R)
        X1 = np.random.normal(mu1, sigma1, size = R)
        X = np.where( H, X1, X0 )
        c = FDR_critical_value( q, mu0, mu1, sigma0, sigma1, p )
        r.append( {
            'q': q,
            'mu0': mu0,
            'mu1': mu1,
            'sigma0': sigma0,
            'sigma1': sigma1,
            'p': p,
            'c': c,
            'FDP': np.sum( ( H == 0 ) & ( X > c ) ) / ( 1e-100 + np.sum( X > c ) ),
        })
    r = pd.DataFrame( r )
    i = np.isfinite( r['c'] ) & (r['FDP'] > 0 )
    assert np.abs( r['q'][i] - r['FDP'][i] ).mean() < 1e-2
    #plt.scatter( r['q'][i], r['FDP'][i] )  # Straight line


def test_numeric_example():

    for rho in [0, 0.2]:
        print( "----------" )
        mu     = .036
        sigma  = .079
        T      = 24
        gamma3 = -2.448
        gamma4 = 10.164
        SR0    = 0
        SR1    = .5
        p_H1   = .10
        alpha  = .10
        SR = mu / sigma
        print( f"SR0                    = {SR0:.3f}" )
        print( f"SR1                    = {SR1:.3f}" )
        print( f"μ                      = {mu:.3f}" )
        print( f"σ                      = {sigma:.3f}" )
        print( f"γ3                     = {gamma3:.3f}" )
        print( f"γ4                     = {gamma4:.3f}" )
        print( f"ρ                      = {rho:.3f}" )
        print( f"T                      = {T}" )
        print( f"SR                     = {SR:.3f}" )
        print( f"σ_SR                   = {math.sqrt( sharpe_ratio_variance( SR = mu / sigma, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f} (non-Gaussian)" )
        print( f"σ_SR                   = {math.sqrt( sharpe_ratio_variance( SR = mu / sigma, gamma3 = 0,      gamma4 = 3,      T = T ) ):.3f} (Gaussian, iid)" )
        print( f"MinTRL                 = {minimum_track_record_length( SR = mu / sigma, SR0 = 0, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ):.3f}" )
        print( f"MinTRL(SR0=.1)         = {minimum_track_record_length( SR = mu / sigma, SR0 = .1, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ):.3f}" )
        print( f"p = 1 - PSR(SR0=0)     = {1-probabilistic_sharpe_ratio( SR = mu / sigma, SR0 = 0,  T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho):.3f}" )
        print( f"PSR(SR0=0)             = {  probabilistic_sharpe_ratio( SR = mu / sigma, SR0 = 0,  T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho):.3f}" )
        print( f"PSR(SR0=.1)            = {probabilistic_sharpe_ratio( SR = mu / sigma, SR0 = .1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho):.3f}" )
        print( f"SR0                    = {SR0:.3f}" )
        print( f"SR_c                   = {critical_sharpe_ratio(SR0, T, gamma3=0.,     gamma4=3.,     rho = 0,   alpha=alpha):.3f} (Gaussian, iid)" )
        print( f"SR_c                   = {critical_sharpe_ratio(SR0, T, gamma3=gamma3, gamma4=gamma4, rho = rho, alpha=alpha):.3f} (non-Gaussian -- unchanged if iid, SR0=0)" )
        print( f"SR1                    = {SR1:.3f}" )
        print( f"Power = 1 - β          = {sharpe_ratio_power( SR0=SR0, SR1 = SR1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ):.3f}" )
        print( f"β                      = {1-sharpe_ratio_power( SR0=SR0, SR1 = SR1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ):.3f}" )
        print( f"P[H1]                  = {p_H1:.3f}" )
        print( f"pFDR = P[H0|SR>SR_c]   = {pFDR( p_H1, alpha, 1 - sharpe_ratio_power( SR0=SR0, SR1 = SR1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ) ):.3f}" )
        print( f"oFDR = P[H0|SR>SR_obs] = {oFDR( SR = mu / sigma, SR0=SR0, SR1=SR1, T=T, p_H1=p_H1, gamma3 = gamma3, gamma4 = gamma4, rho = rho ):.3f}" )

        print( "\nFWER" )
        number_of_trials = 10
        variance = .1
        E_max_SR = expected_maximum_sharpe_ratio( number_of_trials, variance )
        SR0_adj = SR0 + E_max_SR
        SR1_adj = SR1 + E_max_SR
        sigma_SR0_adj_single = math.sqrt( sharpe_ratio_variance( SR = SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) )
        sigma_SR0_adj =  math.sqrt( sharpe_ratio_variance( SR = SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T, K = number_of_trials ) )
        sigma_SR1_adj_single = math.sqrt( sharpe_ratio_variance( SR = SR1_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) )
        sigma_SR1_adj =  math.sqrt( sharpe_ratio_variance( SR = SR1 + SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T, K = number_of_trials ) )
        DSR_single = probabilistic_sharpe_ratio(SR, SR0 = SR0_adj, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho )
        DSR        = probabilistic_sharpe_ratio(SR, SR0 = SR0_adj, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, K = number_of_trials )
        print( f"K                         = {number_of_trials}" )
        print( f"Var[SR_k]                 = {variance:.3f}  (only used to compute E[max SR])" )
        print( f"E[max SR]                 = {E_max_SR:.3f}" )
        print( f"SR0_adj = SR0 + E[max SR] = {SR0_adj:.3f}" )
        print( f"SR1_adj = SR1 + E[max SR] = {SR1_adj:.3f}" )
        print( f"σ_SR0_adj                 = {sigma_SR0_adj:.3f}" )
        print( f"σ_SR1_adj                 = {sigma_SR1_adj:.3f}" )
        print( f"DSR                       = {DSR:.3f}" )
        print( f"oFDR = P[H0|SR>SR_obs]    = {oFDR( SR = mu / sigma, SR0=SR0_adj, SR1=SR1+SR0_adj, T=T, p_H1=p_H1, gamma3 = gamma3, gamma4 = gamma4, rho = rho, K = number_of_trials ):.3f}" )

        print( f"σ_SR0_adj (K=1)              = {sigma_SR0_adj_single:.3f}" )
        print( f"σ_SR1_adj (K=1)              = {sigma_SR1_adj_single:.3f}" )
        print( f"DSR (K=1)                    = {DSR_single:.3f}" )
        print( f"oFDR = P[H0|SR>SR_obs] (K=1) = {oFDR( SR = mu / sigma, SR0=SR0_adj, SR1=SR1+SR0_adj, T=T, p_H1=p_H1, gamma3 = gamma3, gamma4 = gamma4, rho = rho ):.3f}" )

        print( "\nFDR" )
        q    = .25
        alpha_, beta_, SR_c, q_hat = control_for_FDR( q, SR0 = SR0, SR1 = SR1, p_H1 = p_H1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho )
        print( f"P[H1]                  = {p_H1:.3f}" )
        print( f"q                      = {q:.3f}" )
        print( f"α                      = {alpha_:.4f}" )
        print( f"β                      = {beta_:.3f}" )
        print( f"SR_c                   = {SR_c:.3f}" )
        print( f"σ_SR0                  = {math.sqrt( sharpe_ratio_variance( SR = SR0, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f}" )
        print( f"σ_SR1                  = {math.sqrt( sharpe_ratio_variance( SR = SR1, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f}" )

        print( "\nFWER-FDR" )
        alpha_W, beta_W, SR_c_W, q_hat_W = control_for_FDR( q, SR0 = SR0+SR0_adj, SR1 = SR1+SR0_adj, p_H1 = p_H1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho )
        alpha_, beta_, SR_c, q_hat = control_for_FDR( q, SR0 = SR0+SR0_adj, SR1 = SR1+SR0_adj, p_H1 = p_H1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, K = number_of_trials )
        print( f"σ_SR0                  = {math.sqrt( sharpe_ratio_variance( SR = SR0+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T, K = number_of_trials ) ):.3f}" )
        print( f"σ_SR1                  = {math.sqrt( sharpe_ratio_variance( SR = SR1+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T, K = number_of_trials ) ):.3f}" )
        print( f"α                      = {alpha_:.5f}" )
        print( f"β                      = {beta_:.3f}" )
        print( f"SR_c                   = {SR_c:.3f}" )

        print( f"σ_SR0 (K=1)          = {math.sqrt( sharpe_ratio_variance( SR = SR0+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f}" )
        print( f"σ_SR1 (K=1)          = {math.sqrt( sharpe_ratio_variance( SR = SR1+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f}" )
        print( f"α (K=1)              = {alpha_W:.5f}" )
        print( f"β (K=1)              = {beta_W:.3f}" )
        print( f"SR_c (K=1)           = {SR_c_W:.3f}" )
