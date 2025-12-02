# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.18.1",
# ]
# ///

"""Interactive marimo demo for the jsharpe package.

This notebook-style script provides a lightweight, self-contained UI to explore
core functions from jsharpe:
- Probabilistic Sharpe Ratio (PSR)
- Critical Sharpe Ratio and minimum track record length
- Power (1 - beta)
- Expected maximum Sharpe ratio and multiple-testing adjustments
- Posterior FDR metrics (pFDR / oFDR)

It intentionally avoids heavy plotting dependencies so it can run with the
package's default dependencies. Use `make marimo` or run this file directly.
"""

import marimo

app = marimo.App()

with app.setup:
    import sys

    import marimo as mo

    # Allow running from a checkout without installing the package
    # by adding the local src to sys.path if present.
    if "src" not in sys.path:
        sys.path.append("src")

    from jsharpe.sharpe import (
        control_for_FDR,
        critical_sharpe_ratio,
        expected_maximum_sharpe_ratio,
        minimum_track_record_length,
        oFDR,
        pFDR,
        probabilistic_sharpe_ratio,
        sharpe_ratio_power,
        sharpe_ratio_variance,
    )


@app.cell(hide_code=True)
def _():
    mo.md(
        """
        # jsharpe: interactive demo

        Adjust inputs to see how Sharpe-related metrics update in real time.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    mu = mo.ui.number(value=0.036, step=0.001, label="mu (mean)")
    sigma = mo.ui.number(value=0.079, step=0.001, label="sigma (std)")
    T = mo.ui.slider(12, 120, value=24, label="T (observations)")
    gamma3 = mo.ui.number(value=-2.448, step=0.1, label="gamma3 (skew)")
    gamma4 = mo.ui.number(value=10.164, step=0.1, label="gamma4 (kurtosis)")
    rho = mo.ui.slider(0.0, 0.9, value=0.0, step=0.05, label="rho (autocorr)")

    SR0 = mo.ui.number(value=0.0, step=0.05, label="SR0 (null)")
    SR1 = mo.ui.number(value=0.5, step=0.05, label="SR1 (alt)")
    p_H1 = mo.ui.slider(0.0, 0.9, value=0.10, step=0.01, label="P[H1]")
    alpha = mo.ui.slider(0.01, 0.5, value=0.10, step=0.01, label="alpha")

    K = mo.ui.slider(1, 50, value=1, step=1, label="K (number of trials)")

    return mu, sigma, T, gamma3, gamma4, rho, SR0, SR1, p_H1, alpha, K


@app.cell(hide_code=True)
def _(mu, sigma, T, gamma3, gamma4, rho, SR0, SR1, p_H1, alpha, K):
    import math

    import marimo as mo
    import numpy as np

    SR = mu.value / sigma.value if sigma.value != 0 else np.nan

    var_sr = sharpe_ratio_variance(
        SR=SR,
        T=T.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=K.value,
    )
    s_sr = math.sqrt(var_sr)

    psr = probabilistic_sharpe_ratio(
        SR=SR,
        SR0=SR0.value,
        T=T.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=K.value,
    )

    sr_c = critical_sharpe_ratio(
        SR0=SR0.value,
        T=T.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=K.value,
        alpha=alpha.value,
    )

    mtrl = minimum_track_record_length(
        SR=SR,
        SR0=SR0.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        alpha=alpha.value,
    )

    power = sharpe_ratio_power(
        SR0=SR0.value,
        SR1=SR1.value,
        T=T.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        alpha=alpha.value,
        K=K.value,
    )

    emsr = expected_maximum_sharpe_ratio(
        number_of_trials=max(1, int(K.value)),
        variance=var_sr,
        SR0=SR0.value,
    )

    alpha_fdr, beta_fdr, SR_c_fdr, q_hat = control_for_FDR(
        q=0.25,
        SR0=SR0.value,
        SR1=SR1.value,
        p_H1=p_H1.value,
        T=T.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=K.value,
    )

    pfdr_val = pFDR(p_H1.value, alpha.value, 1 - power)
    ofdr_val = oFDR(
        SR=SR,
        SR0=SR0.value,
        SR1=SR1.value,
        T=T.value,
        p_H1=p_H1.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=K.value,
    )

    mo.vstack(
        [
            mo.hstack([mu, sigma, T, gamma3, gamma4, rho]).callout("Inputs: returns distribution", color="neutral"),
            mo.hstack([SR0, SR1, p_H1, alpha, K]).callout("Inputs: hypothesis & multiple testing", color="neutral"),
            mo.md(
                f"""
                ### Results
                - SR = {SR:.3f}
                - sigma_SR = {s_sr:.3f} (variance={var_sr:.5f})
                - PSR(SR0) = {psr:.3f}
                - Critical SR (alpha={alpha.value:.2f}) = {sr_c:.3f}
                - Min track record length = {mtrl:.3f}
                - Power = {power:.3f} (beta={1 - power:.3f})
                - E[max SR] with K={int(K.value)} = {emsr:.3f}
                - pFDR (given SR_c) = {pfdr_val:.3f}
                - oFDR (given SR_obs) = {ofdr_val:.3f}
                - FDR control (q=0.25): alpha={alpha_fdr:.4f}, beta={beta_fdr:.3f}, SR_c={SR_c_fdr:.3f}, q_hat={q_hat:.3f}
                """
            ).callout("Computed metrics", color="green"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
