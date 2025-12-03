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

    import marimo as mo

    from jsharpe import (
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
    mu = mo.ui.number(value=0.036, step=0.001, label="mu (mean)")
    sigma = mo.ui.number(value=0.079, step=0.001, label="sigma (std)")
    t = mo.ui.slider(12, 120, value=24, label="T (observations)")
    gamma3 = mo.ui.number(value=-2.448, step=0.1, label="gamma3 (skew)")
    gamma4 = mo.ui.number(value=10.164, step=0.1, label="gamma4 (kurtosis)")
    rho = mo.ui.slider(0.0, 0.9, value=0.0, step=0.05, label="rho (autocorr)")

    sr0 = mo.ui.number(value=0.0, step=0.05, label="SR0 (null)")
    sr1 = mo.ui.number(value=0.5, step=0.05, label="SR1 (alt)")
    p_h1 = mo.ui.slider(0.0, 0.9, value=0.10, step=0.01, label="P[H1]")
    alpha = mo.ui.slider(0.01, 0.5, value=0.10, step=0.01, label="alpha")

    k = mo.ui.slider(1, 50, value=1, step=1, label="K (number of trials)")

    return mu, sigma, t, gamma3, gamma4, rho, sr0, sr1, p_h1, alpha, k


@app.cell(hide_code=True)
def _(mu, sigma, t, gamma3, gamma4, rho, sr0, sr1, p_h1, alpha, k):
    import math

    import numpy as np

    sr = mu.value / sigma.value if sigma.value != 0 else np.nan

    var_sr = sharpe_ratio_variance(
        SR=sr,
        T=t.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=k.value,
    )
    s_sr = math.sqrt(var_sr)

    psr = probabilistic_sharpe_ratio(
        SR=sr,
        SR0=sr0.value,
        T=t.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=k.value,
    )

    sr_c = critical_sharpe_ratio(
        SR0=sr0.value,
        T=t.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=k.value,
        alpha=alpha.value,
    )

    mtrl = minimum_track_record_length(
        SR=sr,
        SR0=sr0.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        alpha=alpha.value,
    )

    power = sharpe_ratio_power(
        SR0=sr0.value,
        SR1=sr1.value,
        T=t.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        alpha=alpha.value,
        K=k.value,
    )

    emsr = expected_maximum_sharpe_ratio(
        number_of_trials=max(1, int(k.value)),
        variance=var_sr,
        SR0=sr0.value,
    )

    alpha_fdr, beta_fdr, sr_c_fdr, q_hat = control_for_FDR(
        q=0.25,
        SR0=sr0.value,
        SR1=sr1.value,
        p_H1=p_h1.value,
        T=t.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=k.value,
    )

    pfdr_val = pFDR(p_h1.value, alpha.value, 1 - power)
    ofdr_val = oFDR(
        SR=sr,
        SR0=sr0.value,
        SR1=sr1.value,
        T=t.value,
        p_H1=p_h1.value,
        gamma3=gamma3.value,
        gamma4=gamma4.value,
        rho=rho.value,
        K=k.value,
    )

    mo.vstack(
        [
            mo.hstack([mu, sigma, t, gamma3, gamma4, rho]).callout("Inputs: returns distribution"),
            mo.hstack([sr0, sr1, p_h1, alpha, k]).callout("Inputs: hypothesis & multiple testing"),
            mo.md(
                f"""
                ### Results
                - SR = {sr:.3f}
                - sigma_SR = {s_sr:.3f} (variance={var_sr:.5f})
                - PSR(SR0) = {psr:.3f}
                - Critical SR (alpha={alpha.value:.2f}) = {sr_c:.3f}
                - Min track record length = {mtrl:.3f}
                - Power = {power:.3f} (beta={1 - power:.3f})
                - E[max SR] with K={int(k.value)} = {emsr:.3f}
                - pFDR (given SR_c) = {pfdr_val:.3f}
                - oFDR (given SR_obs) = {ofdr_val:.3f}
                - FDR control (q=0.25): alpha={alpha_fdr:.4f}, beta={beta_fdr:.3f}, SR_c={sr_c_fdr:.3f},
                  q_hat={q_hat:.3f}
                """
            ).callout("Computed metrics"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
