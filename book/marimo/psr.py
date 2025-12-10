"""Marimo Example: Probabilistic Sharpe Ratio (PSR).

A compact, lint-clean app demonstrating PSR computation using jsharpe.
Adjust SR, SR0, T, skew, kurtosis, autocorrelation, and trials. Results
react live. Ideal as a minimal template for financial analytics apps.
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App()

with app.setup:
    import importlib
    import math
    import subprocess
    from pathlib import Path

    import marimo as mo

    project_root = Path(__file__).resolve().parents[2]
    print(f"Project root: {project_root}")

    result = importlib.util.find_spec("jsharpe")
    print(result)

    if not result:
        # Run uv install and wait until fully finished
        subprocess.run(["uv", "pip", "install", "-e", str(project_root)], check=True)

        # Invalidate import caches to make newly installed package visible
        importlib.invalidate_caches()


@app.function
def fmt(x, precision: int = 3) -> str:
    """Format values for display."""
    if x is None:
        return "—"
    try:
        if math.isnan(x) or math.isinf(x):
            return str(x)
        return f"{x:.{precision}f}"
    except Exception:
        return str(x)


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Probabilistic Sharpe Ratio (PSR)
    Explore PSR interactively using jsharpe.
    """)
    return


@app.cell
def _inputs():
    """Create all widgets and return them in a dict for safe access."""
    # Basic
    sr = mo.ui.slider(0.0, 0.5, step=0.05, label="SR (observed)")
    sr0 = mo.ui.slider(0.0, 0.5, step=0.05, label="SR0 (benchmark)")
    T = mo.ui.slider(12, 500, value=24, step=12, label="T (observations)")

    # Advanced
    gamma3 = mo.ui.slider(-3.0, 3.0, value=0.0, step=0.1, label="gamma3 (skew)")
    gamma4 = mo.ui.slider(2.0, 8.0, step=0.1, label="gamma4 (kurtosis)")
    rho = mo.ui.slider(0.0, 0.9, value=0.0, step=0.05, label="rho (autocorr)")
    K = mo.ui.slider(1, 200, value=1, step=1, label="K (trials)")

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("### Basic"),
                    sr,
                    sr0,
                    T,
                ]
            ),
            mo.vstack(
                [
                    mo.md("### Advanced"),
                    gamma3,
                    gamma4,
                    rho,
                    K,
                ]
            ),
        ]
    )
    return K, T, gamma3, gamma4, rho, sr, sr0


@app.cell
def display(K, T, gamma3, gamma4, rho, sr, sr0):
    """Render the PSR result markdown.

    Args:
        K: Trials slider widget controlling multiple-testing adjustment.
        T: Observations slider widget.
        gamma3: Skewness (third standardized moment) widget.
        gamma4: Kurtosis (fourth standardized moment) widget.
        rho: Autocorrelation widget.
        sr: Observed Sharpe ratio widget.
        sr0: Benchmark Sharpe ratio widget.
    """
    from jsharpe import probabilistic_sharpe_ratio

    mo.md(f"""
    ### Result
    **PSR = {
        fmt(probabilistic_sharpe_ratio(sr.value, sr0.value, T=T.value, gamma3=gamma3.value, gamma4=gamma4.value, rho=rho.value, K=K.value), 4)
    }**
    SR = {fmt(sr.value)} vs SR0 = {fmt(sr0.value)}
    T={T.value}, γ₃={fmt(gamma3.value)}, γ₄={fmt(gamma4.value)}, ρ={fmt(rho.value)}, K={K.value}
    """)
    return


if __name__ == "__main__":
    app.run()
