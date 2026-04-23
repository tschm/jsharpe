import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Precision and Recall of PSR
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from scipy import stats
    from tqdm.auto import tqdm

    from jsharpe import generate_autocorrelated_non_gaussian_data, generate_non_gaussian_data, sharpe_ratio_variance

    return (
        generate_autocorrelated_non_gaussian_data,
        generate_non_gaussian_data,
        np,
        pd,
        sharpe_ratio_variance,
        stats,
        tqdm,
    )


@app.cell
def _():
    import ray

    ray.init()
    return (ray,)


@app.cell
def _(np, sharpe_ratio_variance, stats):
    REPS = 10_000
    SR0_annual_list = [0.0, 0.5, 1.0, 1.5, 2.0]
    [s / np.sqrt(252) for s in SR0_annual_list]
    RSEED = 2025
    # Mixture configs: (name, p_tail, mu_tail, sigma_tail, sigma_core)
    configs = [
        ("gaussian", 0.00, 0.00, 0.010, 0.010),
        ("mild", 0.04, -0.03, 0.015, 0.010),
        ("moderate", 0.03, -0.045, 0.020, 0.010),
        ("severe", 0.02, -0.060, 0.025, 0.010),
    ]

    RHOs = [0, 0.2]

    def mixture_variance(p_tail, mu_tail, sigma_tail, mu_core, sigma_core):
        w = 1.0 - p_tail
        mu = w * mu_core + p_tail * mu_tail
        m2 = w * (sigma_core**2 + mu_core**2) + p_tail * (sigma_tail**2 + mu_tail**2)
        return m2 - mu**2

    def gen_with_true_SR0(reps, T, cfg, SR0, seed):
        _name, p, mu_tail, sig_tail, sig_core = cfg
        # Zero-mean baseline mixture (choose mu_core so mean=0)
        mu_core0 = -p * mu_tail / (1.0 - p)
        std0 = np.sqrt(mixture_variance(p, mu_tail, sig_tail, mu_core0, sig_core))
        mu_shift = SR0 * std0  # sets population Sharpe to SR0, preserves skew/kurt
        rng = np.random.default_rng(seed)
        mask = rng.random((reps, T)) < p
        X = rng.normal(mu_core0 + mu_shift, sig_core, size=(reps, T))
        X[mask] = rng.normal(mu_tail + mu_shift, sig_tail, size=mask.sum())
        return X

    def psr_z_T(X, SR0):
        Tn = X.shape[1]
        s = X.std(axis=1, ddof=1)
        sr_hat = X.mean(axis=1) / s
        skew = stats.skew(X, axis=1, bias=False)
        kappa = stats.kurtosis(X, axis=1, fisher=True, bias=False) + 3.0
        den = np.sqrt((1.0 / Tn) * (1.0 - skew * SR0 + ((kappa - 1.0) / 4.0) * (SR0**2)))
        return (sr_hat - SR0) / den

    def t_stat(X, SR0):
        Tn = X.shape[1]
        s = X.std(axis=1, ddof=1)
        sr_hat = X.mean(axis=1) / s
        skew = 0
        kappa = 3.0
        den = np.sqrt((1.0 / Tn) * (1.0 - skew * SR0 + ((kappa - 1.0) / 4.0) * (SR0**2)))
        return (sr_hat - SR0) / den

    def my_psr_z_T(X, SR0, rho):
        Tn = X.shape[1]
        s = X.std(axis=1, ddof=1)
        sr_hat = X.mean(axis=1) / s
        skew = stats.skew(X, axis=1, bias=False)
        kappa = stats.kurtosis(X, axis=1, fisher=True, bias=False) + 3.0
        v = sharpe_ratio_variance(SR0, Tn, gamma3=skew, gamma4=kappa, rho=rho, K=1)
        den = np.sqrt(v)
        return (sr_hat - SR0) / den

    return REPS, RHOs, RSEED, configs, gen_with_true_SR0, my_psr_z_T, psr_z_T


@app.cell
def _(REPS, RSEED, configs, gen_with_true_SR0, np, pd, psr_z_T, stats):
    SR0_daily = 0.0
    SR1_annual_list = [0.5, 1.0, 1.5, 2.0]
    SR1_daily_list = [s / np.sqrt(252) for s in SR1_annual_list]
    SR1_daily_list = [0.15, 0.3, 0.45, 0.6]
    T_1 = 12 * 5

    def _confusion_metrics(y_true, pvals, alpha=0.05):
        yhat = pvals < alpha
        TP = int(((_y_true == 1) & yhat).sum())
        FP = int(((_y_true == 0) & yhat).sum())
        int(((_y_true == 0) & ~yhat).sum())
        FN = int(((_y_true == 1) & ~yhat).sum())
        _prec = TP / (TP + FP) if TP + FP > 0 else np.nan
        _rec = TP / (TP + FN) if TP + FN > 0 else np.nan
        _f1 = (
            2 * _prec * _rec / (_prec + _rec) if _prec > 0 and _rec > 0 else 0.0 if _prec == 0 or _rec == 0 else np.nan
        )
        return (_prec, _rec, _f1)

    rows = []
    for cfg in configs:
        for _SR1_daily, _SR1_annual in zip(SR1_daily_list, SR1_annual_list, strict=False):
            _X0 = gen_with_true_SR0(REPS, T_1, cfg, SR0=SR0_daily, seed=RSEED)
            _X1 = gen_with_true_SR0(REPS, T_1, cfg, SR0=_SR1_daily, seed=RSEED + 1)
            _y_true = np.r_[np.zeros(len(_X0), dtype=int), np.ones(len(_X1), dtype=int)]
            X = np.concatenate([_X0, _X1], axis=1)
            skew = stats.skew(X, axis=1, bias=False)
            kappa = stats.kurtosis(X, axis=1, fisher=True, bias=False) + 3.0
            _p_psr = np.r_[stats.norm.sf(psr_z_T(_X0, SR0_daily)), stats.norm.sf(psr_z_T(_X1, SR0_daily))]
            _prec, _rec, _f1 = _confusion_metrics(_y_true, _p_psr, alpha=0.05)
            rows.append(
                {
                    "config": cfg[0],
                    "SR1": _SR1_daily,
                    "gamma3": skew.mean(),
                    "gamma4": kappa.mean(),
                    "PSR_precision": _prec,
                    "PSR_recall": _rec,
                    "PSR_F1": _f1,
                }
            )
    _psr_table = pd.DataFrame(rows).sort_values(["config", "SR1"]).set_index(["config", "SR1"]).round(4)
    _psr_table.to_csv("appendix_2.csv")
    _psr_table
    return SR0_daily, SR1_annual_list, SR1_daily_list, T_1


@app.cell
def _(
    REPS,
    RHOs,
    SR0_daily,
    SR1_daily_list,
    T_1,
    generate_autocorrelated_non_gaussian_data,
    generate_non_gaussian_data,
    my_psr_z_T,
    np,
    pd,
    ray,
    stats,
    tqdm,
):
    @ray.remote
    def f2(rho, name, SR0_daily, SR1_daily):
        if rho == 0:
            _X0 = generate_non_gaussian_data(T_1, REPS, SR0=SR0_daily, name=name)
            _X1 = generate_non_gaussian_data(T_1, REPS, SR0=_SR1_daily, name=name)
        else:
            _X0 = generate_autocorrelated_non_gaussian_data(T_1, REPS, SR0=SR0_daily, name=name, rho=rho)
            _X1 = generate_autocorrelated_non_gaussian_data(T_1, REPS, SR0=_SR1_daily, name=name, rho=rho)
        _y_true = np.r_[np.zeros(REPS, dtype=int), np.ones(REPS, dtype=int)]
        X = np.concatenate([_X0, _X1], axis=1)
        skew = stats.skew(X, axis=1, bias=False)
        kappa = stats.kurtosis(X, axis=1, fisher=True, bias=False) + 3.0
        _p_psr = np.r_[
            stats.norm.sf(my_psr_z_T(_X0.T, SR0_daily, rho)), stats.norm.sf(my_psr_z_T(_X1.T, SR0_daily, rho))
        ]
        _prec, _rec, _f1 = _confusion_metrics(_y_true, _p_psr, alpha=0.05)
        return {
            "name": name,
            "rho": rho,
            "gamma3": skew.mean(),
            "gamma4": kappa.mean(),
            "SR1": _SR1_daily,
            "PSR_precision": _prec,
            "PSR_recall": _rec,
            "PSR_F1": _f1,
        }

    rows_1 = [
        f2.remote(rho, name, SR0_daily, _SR1_daily)
        for rho in RHOs
        for name in ["gaussian", "mild", "moderate", "severe"]
        for _SR1_daily in SR1_daily_list
    ]
    rows_1 = [ray.get(r) for r in tqdm(rows_1)]
    rows_1 = pd.DataFrame(rows_1)
    return (rows_1,)


@app.cell
def _(
    REPS,
    SR1_annual_list,
    SR1_daily_list,
    T_1,
    generate_non_gaussian_data,
    my_psr_z_T,
    np,
    stats,
):
    if False:
        rho = 0
        name = "gaussian"
        SR0_daily_1 = 0
        _SR1_daily = SR1_daily_list[1]
        SR1_annual_list[1]
        _X0 = generate_non_gaussian_data(T_1, REPS, SR0=SR0_daily_1, name=name)
        _X1 = generate_non_gaussian_data(T_1, REPS, SR0=_SR1_daily, name=name)
        _y_true = np.r_[np.zeros(REPS, dtype=int), np.ones(REPS, dtype=int)]
        _p_psr = np.r_[
            stats.norm.sf(my_psr_z_T(_X0.T, SR0_daily_1, rho)), stats.norm.sf(my_psr_z_T(_X1.T, SR0_daily_1, rho))
        ]
        _prec, _rec, _f1 = _confusion_metrics(_y_true, _p_psr, alpha=0.05)
        _rec
    return


@app.cell
def _(pd, rows_1):
    _psr_table = pd.DataFrame(rows_1).sort_values(["name", "rho", "SR1"])
    _psr_table.to_csv("exhibit_3.csv", index=False)
    _psr_table.set_index(["name", "rho", "SR1"]).round(2)
    return


if __name__ == "__main__":
    app.run()
