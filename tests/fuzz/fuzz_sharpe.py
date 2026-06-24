"""Fuzz the jsharpe linear-algebra helpers against arbitrary matrices/sizes.

``ppoints`` builds plotting positions from an integer count, while
``robust_covariance_inverse``, ``minimum_variance_weights_for_correlated_assets``
and ``effective_rank`` consume square covariance-shaped matrices. None of them
should crash with an unexpected exception on adversarial input — they should
return a result or raise a documented error (or numpy's ``LinAlgError``). This
harness exercises that contract with coverage-guided input.

Run locally:
    pip install atheris numpy scipy
    python tests/fuzz/fuzz_sharpe.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

# Pre-import the native dependencies OUTSIDE the instrumentation block so they
# load uninstrumented; atheris's import hook can miscompile C-accelerated
# libraries. Only the first-party package under test is instrumented.
import numpy as np
import scipy.linalg  # noqa: F401  # pre-imported uninstrumented

with atheris.instrument_imports():
    from jsharpe.sharpe import (
        effective_rank,
        minimum_variance_weights_for_correlated_assets,
        ppoints,
        robust_covariance_inverse,
    )

# Errors the routines may legitimately raise on adversarial input. Anything
# outside this set propagates and is recorded by Atheris as a crash.
_ALLOWED = (ValueError, ZeroDivisionError, np.linalg.LinAlgError, FloatingPointError)


def _matrix(fdp: atheris.FuzzedDataProvider) -> np.ndarray:
    """Build an n x n float64 matrix from fuzzed bytes (n in [0, 8])."""
    n = fdp.ConsumeIntInRange(0, 8)
    floats = [fdp.ConsumeFloat() for _ in range(n * n)]
    return np.array(floats, dtype=np.float64).reshape(n, n)


def test_one_input(data: bytes) -> None:
    """Run fuzzed sizes/matrices through the jsharpe helpers."""
    fdp = atheris.FuzzedDataProvider(data)

    with contextlib.suppress(_ALLOWED):
        ppoints(fdp.ConsumeIntInRange(0, 64))

    matrix = _matrix(fdp)
    for fn in (robust_covariance_inverse, minimum_variance_weights_for_correlated_assets, effective_rank):
        with contextlib.suppress(_ALLOWED):
            fn(matrix)


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
