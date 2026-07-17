"""Tests for :mod:`jsharpe.sharpe.linalg` (ppoints and covariance helpers).

These assert the public behaviour and contracts of the linear-algebra helpers
directly (values, bounds, invariants) rather than parity against a reference
implementation. The module also hosts the package-wide import-lint guard.
"""
# ruff: noqa: N806

import ast
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest

import jsharpe.sharpe as sharpe_pkg
from jsharpe import (
    minimum_variance_weights_for_correlated_assets,
    ppoints,
    robust_covariance_inverse,
)


def test_ppoints_default_large_n():
    """Default behaviour when n > 10 (a defaults to 0.5)."""
    n = 20
    expected = np.linspace(1 - 0.5, n - 0.5, n) / (n + 1 - 2 * 0.5)
    x = ppoints(n)
    assert np.allclose(x, expected)
    assert np.all(x > 0)
    assert np.all(x < 1)
    diffs = np.diff(x)
    assert np.allclose(diffs, diffs[0])  # uniform spacing


def test_ppoints_default_small_n():
    """Default behaviour when n <= 10 (a defaults to 3/8)."""
    n = 10
    a = 3 / 8
    expected = np.linspace(1 - a, n - a, n) / (n + 1 - 2 * a)
    x = ppoints(n)
    assert np.allclose(x, expected)
    diffs = np.diff(x)
    assert np.allclose(diffs, diffs[0])


def test_ppoints_custom_a_zero():
    """Custom a=0.0 should exclude the boundaries 0 and 1."""
    n = 5
    x = ppoints(n, a=0.0)
    assert x[0] == pytest.approx(1 / (n + 1))
    assert x[-1] == pytest.approx(n / (n + 1))


def test_ppoints_custom_a_one_includes_boundaries():
    """Custom a=1.0 includes both boundaries 0 and 1 by formula design."""
    n = 5
    x = ppoints(n, a=1.0)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)


def test_ppoints_invalid_a_raises():
    """Invalid a outside [0, 1] should raise ValueError."""
    with pytest.raises(ValueError, match="offset should be in"):
        ppoints(7, a=-0.01)
    with pytest.raises(ValueError, match="offset should be in"):
        ppoints(7, a=1.01)


def test_robust_covariance_inverse_matches_numpy_inverse():
    """The Sherman-Morrison inverse reproduces the true inverse of a constant-corr covariance."""
    np.random.seed(0)
    C = 0.5 * np.ones(shape=(10, 10))
    np.fill_diagonal(C, 1)
    sigma = np.random.lognormal(size=10).reshape(-1, 1)
    V = (C * sigma).T * sigma
    assert np.allclose(robust_covariance_inverse(V), np.linalg.inv(V), atol=1e-12)


def test_minimum_variance_weights_match_convex_solver():
    """Closed-form minimum-variance weights match a convex-optimiser solution and sum to one."""
    np.random.seed(0)
    C = 0.5 * np.ones(shape=(10, 10))
    np.fill_diagonal(C, 1)
    sigma = np.random.lognormal(size=10).reshape(-1, 1)
    V = (C * sigma).T * sigma
    w = minimum_variance_weights_for_correlated_assets(V)
    assert w.sum() == pytest.approx(1.0)

    solver_w = cp.Variable(shape=V.shape[0])
    problem = cp.Problem(cp.Minimize(cp.quad_form(solver_w, V)), [solver_w.sum() == 1])
    problem.solve()
    assert np.allclose(solver_w.value, w, atol=1e-10)


# --- package-wide architecture guard (see ARCHITECTURE.md) ------------------

# Layer index: a module may only import modules with a strictly lower index.
_LAYER = {
    "linalg": 0,
    "quadrature": 0,
    "clustering": 0,
    "psr": 1,
    "generators": 1,
    "corrections": 2,
}


def test_layering_is_acyclic_and_downward():
    """Intra-package imports only ever point downward, so the import graph stays an acyclic DAG."""
    pkg_dir = Path(sharpe_pkg.__file__).parent
    violations = []
    for module, idx in _LAYER.items():
        tree = ast.parse((pkg_dir / f"{module}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # Relative `from .<module> import ...` statements carry level == 1.
            is_intra_import = isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in _LAYER
            if is_intra_import and _LAYER[node.module] >= idx:
                violations.append(f"{module} (layer {idx}) imports {node.module} (layer {_LAYER[node.module]})")
    assert not violations, "Upward/same-layer intra-package imports found: " + "; ".join(violations)
