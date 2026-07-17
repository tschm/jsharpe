"""Tests for :mod:`jsharpe.sharpe.quadrature` (Gauss-Hermite expectation, moments).

Assertions target closed-form contracts of the standard normal and the maximum
of k standard normals, plus invariants of the equi-correlated case. Also hosts
the public-API facade consistency guard.
"""
# ruff: noqa: N802, N806

import math
from itertools import pairwise

import numpy as np
import pytest

import jsharpe
import jsharpe.sharpe as sharpe_pkg
from jsharpe import make_expectation_gh

# moments_Mk / E_under_normal are internal helpers, reachable via full path only.
from jsharpe.sharpe.quadrature import E_under_normal, moments_Mk


def test_make_expectation_gh_reproduces_standard_normal_moments():
    """Gauss-Hermite expectation should reproduce the first standard-normal moments."""
    E = make_expectation_gh(n_nodes=50)
    assert E(lambda x: np.ones_like(x)) == pytest.approx(1.0, rel=1e-10, abs=1e-10)  # E[1] = 1
    assert E(lambda x: x) == pytest.approx(0.0, abs=1e-8)  # E[Z] = 0
    assert E(lambda x: x**2) == pytest.approx(1.0, rel=1e-6, abs=1e-6)  # E[Z^2] = 1


def test_E_under_normal_singleton_behaves_like_expectation():
    """The precomputed E_under_normal callable integrates the standard normal correctly."""
    assert E_under_normal(lambda x: np.ones_like(x)) == pytest.approx(1.0, abs=1e-10)
    assert E_under_normal(lambda x: x**2) == pytest.approx(1.0, abs=1e-8)


def test_moments_Mk_single_variable_is_standard_normal():
    """For k=1 the maximum is a single N(0,1): mean 0, variance 1."""
    Ez, Ez2, var = moments_Mk(1)
    assert Ez == pytest.approx(0.0, abs=1e-10)
    assert Ez2 == pytest.approx(1.0, abs=1e-10)
    assert var == pytest.approx(1.0, abs=1e-10)


def test_moments_Mk_pair_matches_closed_form():
    """E[max(Z1, Z2)] has the closed form 1/sqrt(pi) for two independent standard normals."""
    Ez, _Ez2, _var = moments_Mk(2)
    assert Ez == pytest.approx(1.0 / math.sqrt(math.pi), abs=1e-6)


def test_moments_Mk_expected_maximum_grows_with_k():
    """E[max of k standard normals] increases monotonically in k."""
    means = [moments_Mk(k)[0] for k in (1, 2, 5, 10)]
    assert all(later > earlier for earlier, later in pairwise(means))


def test_moments_Mk_correlation_relations():
    """Equi-correlation rescales the mean by (1-rho) and lifts the variance toward rho."""
    Ez0, _, var0 = moments_Mk(5, rho=0.0)
    rho = 0.3
    Ez_rho, Ez2_rho, var_rho = moments_Mk(5, rho=rho)
    assert Ez_rho == pytest.approx((1 - rho) * Ez0, abs=1e-10)
    assert var_rho == pytest.approx(rho + (1 - rho) * var0, abs=1e-10)
    assert Ez2_rho == pytest.approx(var_rho + Ez_rho**2, abs=1e-10)


def test_public_api_facades_are_consistent():
    """The top-level and subpackage facades re-export the identical public surface."""
    assert set(jsharpe.__all__) == set(sharpe_pkg.__all__)
    # Internal helpers must stay off the public surface (see issue #273).
    assert "moments_Mk" not in jsharpe.__all__
    assert "E_under_normal" not in sharpe_pkg.__all__
