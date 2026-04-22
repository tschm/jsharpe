"""Session fixture loading the original zoonek/2025-sharpe-ratio functions module."""

import importlib.util
import sys
import types
import urllib.request
from pathlib import Path

import pytest

_CACHE = Path(__file__).parent / "_zoonek_functions.py"
_URL = "https://raw.githubusercontent.com/zoonek/2025-sharpe-ratio/main/functions.py"


def _mock_unavailable_modules() -> None:
    """Mock modules the original file imports at the top level.

    These are not used by any of the functions under test.
    """
    deprecated_mod = types.ModuleType("deprecated")
    deprecated_mod.deprecated = lambda *a, **kw: lambda f: f

    plt_mod = types.ModuleType("matplotlib.pyplot")
    matplotlib_mod = types.ModuleType("matplotlib")
    matplotlib_mod.pyplot = plt_mod

    sklearn_mod = types.ModuleType("sklearn")
    sklearn_cluster_mod = types.ModuleType("sklearn.cluster")
    sklearn_cluster_mod.KMeans = type("KMeans", (), {})
    sklearn_metrics_mod = types.ModuleType("sklearn.metrics")
    sklearn_metrics_mod.silhouette_samples = lambda *a, **kw: None

    for name, mod in [
        ("deprecated", deprecated_mod),
        ("matplotlib", matplotlib_mod),
        ("matplotlib.pyplot", plt_mod),
        ("sklearn", sklearn_mod),
        ("sklearn.cluster", sklearn_cluster_mod),
        ("sklearn.metrics", sklearn_metrics_mod),
    ]:
        sys.modules.setdefault(name, mod)


@pytest.fixture(scope="session")
def original():
    """Original functions from zoonek/2025-sharpe-ratio, loaded once per session."""
    if not _CACHE.exists():
        assert _URL.startswith("https://"), _URL
        with urllib.request.urlopen(_URL) as resp:  # noqa: S310
            _CACHE.write_bytes(resp.read())
    _mock_unavailable_modules()
    spec = importlib.util.spec_from_file_location("_zoonek_functions", _CACHE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
