"""Marimo Example: Probabilistic Sharpe Ratio (PSR).

A compact, lint-clean app demonstrating PSR computation using jsharpe.
Adjust SR, SR0, T, skew, kurtosis, autocorrelation, and trials. Results
react live. Ideal as a minimal template for financial analytics apps.
"""
import marimo

__generated_with = "0.18.3"
app = marimo.App()

with app.setup:
    import math
    import subprocess
    from pathlib import Path

    import marimo as mo

    project_root = Path(__file__).parent.parent.parent
    print(f"Project root: {project_root}")

    mo.md("Installing dependencies via `make install`...")
    result = subprocess.run(["make", "install"], cwd=project_root)

    from jsharpe import probabilistic_sharpe_ratio


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


@app.function
def psr_value(sr, sr0, T, gamma3, gamma4, rho, K):
    """Compute PSR from widget values and return the numeric result."""
    return probabilistic_sharpe_ratio(
        SR=float(sr.value),
        SR0=float(sr0.value),
        T=int(T.value),
        gamma3=float(gamma3.value),
        gamma4=float(gamma4.value),
        rho=float(rho.value),
        K=int(K.value),
    )


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
    mo.md(f"""
    ### Result
    **PSR = {fmt(psr_value(sr, sr0, T, gamma3, gamma4, rho, K), 4)}**
    SR = {fmt(sr.value)} vs SR0 = {fmt(sr0.value)}
    T={T.value}, γ₃={fmt(gamma3.value)}, γ₄={fmt(gamma4.value)}, ρ={fmt(rho.value)}, K={K.value}
    """)
    return


if __name__ == "__main__":
    app.run()
