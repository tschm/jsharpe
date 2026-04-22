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
    # FWER control under different processes
    """)
    return


@app.cell
def _():
    import logging
    import math

    import numpy as np
    import pandas as pd
    import scipy
    from sklearn.metrics import f1_score, precision_score, recall_score

    from jsharpe import (
        expected_maximum_sharpe_ratio,
        generate_autocorrelated_non_gaussian_data,
        variance_of_the_maximum_of_k_Sharpe_ratios,
    )

    logging.basicConfig(
        format="%(asctime)-15s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    def LOG(*args) -> None:
        logging.info(*args)

    return (
        LOG,
        expected_maximum_sharpe_ratio,
        f1_score,
        generate_autocorrelated_non_gaussian_data,
        math,
        np,
        pd,
        precision_score,
        recall_score,
        scipy,
        variance_of_the_maximum_of_k_Sharpe_ratios,
    )


@app.cell
def _():
    SR0 = 0
    SR1_list = [0.5]
    T = 60
    REPS_H0 = 10_000  # null-calibration repetitions
    REPS_MIX = 10_000  # mixed H0/H1 repetitions
    TRIALS = 10
    P_H1 = 0.1
    ALPHA = 0.05  # Desired FPR
    return ALPHA, P_H1, REPS_H0, REPS_MIX, SR0, SR1_list, T, TRIALS


@app.cell
def _():
    if False:
        MODELS_1 = ["gaussian"]  # For debugging
        RHOs_1 = [0]
    return MODELS_1, RHOs_1


@app.cell
def _(
    ALPHA,
    LOG,
    MODELS_1,
    REPS_H0,
    RHOs_1,
    SR0,
    T,
    TRIALS,
    display,
    expected_maximum_sharpe_ratio,
    generate_autocorrelated_non_gaussian_data,
    math,
    np,
    pd,
    scipy,
    variance_of_the_maximum_of_k_Sharpe_ratios,
):
    LOG("Starting null calibration (global H0)")
    null_srs = {(_rho, _name): [] for _rho in RHOs_1 for _name in MODELS_1}
    for _rho in RHOs_1:
        for _name in MODELS_1:
            LOG(f"[H0 calibration] rho={_rho}, model={_name}")
            for _i in range(REPS_H0):
                _X = generate_autocorrelated_non_gaussian_data(T, TRIALS, rho=_rho, SR0=SR0, name=_name)
                _SR = _X.mean(axis=0) / _X.std(axis=0)
                null_srs[_rho, _name].extend(_SR)
    calib = {}
    z_alpha = scipy.stats.norm.ppf(1 - ALPHA)
    for (_rho, _name), srs in null_srs.items():
        srs = np.asarray(srs)
        var_SR0 = np.var(srs, ddof=1)
        _E_max_SR0 = expected_maximum_sharpe_ratio(number_of_trials=TRIALS, variance=var_SR0)
        _sigma_max = math.sqrt(variance_of_the_maximum_of_k_Sharpe_ratios(number_of_trials=TRIALS, variance=var_SR0))
        _SR0_adj = SR0 + _E_max_SR0
        _SR_c = _SR0_adj + _sigma_max * z_alpha
        calib[_rho, _name] = {
            "var_SR0": var_SR0,
            "E_max_SR0": _E_max_SR0,
            "sigma_max": _sigma_max,
            "SR0_adj": _SR0_adj,
            "SR_c": _SR_c,
        }
    calib_df = pd.DataFrame([{"rho": _rho, "name": _name, **vals} for (_rho, _name), vals in calib.items()])
    LOG("Null calibration completed")
    display(calib_df)
    return (calib,)


@app.cell
def _(
    LOG,
    MODELS_1,
    P_H1,
    REPS_MIX,
    RHOs_1,
    SR0,
    SR1_list,
    T,
    TRIALS,
    calib,
    generate_autocorrelated_non_gaussian_data,
    np,
    pd,
    scipy,
):
    LOG("Starting mixed H0/H1 experiment")
    rows = []
    for _rho in RHOs_1:
        for _name in MODELS_1:
            SR_c_info = calib[_rho, _name]
            _SR_c = SR_c_info["SR_c"]
            _E_max_SR0 = SR_c_info["E_max_SR0"]
            _sigma_max = SR_c_info["sigma_max"]
            _SR0_adj = SR_c_info["SR0_adj"]
            for _SR1 in SR1_list:
                LOG(f"[Mixed] rho={_rho}, model={_name}, SR1={_SR1}")
                for it in range(REPS_MIX):
                    H1 = np.random.uniform(size=TRIALS) < P_H1
                    H1.sort()
                    X0 = X1 = None
                    K1 = H1.sum()
                    K0 = TRIALS - K1
                    if K0 > 0:
                        X0 = generate_autocorrelated_non_gaussian_data(T, K0, rho=_rho, SR0=SR0, name=_name)
                    if K1 > 0:
                        X1 = generate_autocorrelated_non_gaussian_data(T, K1, rho=_rho, SR0=_SR1, name=_name)
                    if X0 is None:
                        _X = X1
                    elif X1 is None:
                        _X = X0
                    else:
                        _X = np.concatenate([X0, X1], axis=1)
                    gamma3 = scipy.stats.skew(_X.flatten())
                    gamma4 = scipy.stats.kurtosis(_X.flatten(), fisher=False)
                    _SR = _X.mean(axis=0) / _X.std(axis=0)
                    sr_max = np.max(_SR)
                    var_SR_emp = np.var(_SR, ddof=1)
                    reject = sr_max > _SR_c
                    _tmp = pd.DataFrame({"SR": _SR, "H1": H1})
                    _tmp["rho"] = _rho
                    _tmp["name"] = _name
                    _tmp["SR1"] = _SR1
                    _tmp["iteration"] = it
                    _tmp["Max(SR)"] = sr_max
                    _tmp["Var[SR]"] = var_SR_emp
                    _tmp["gamma3"] = gamma3
                    _tmp["gamma4"] = gamma4
                    _tmp["SR_c"] = _SR_c
                    _tmp["SR0_adj"] = _SR0_adj
                    _tmp["E[Max(SR)]"] = _E_max_SR0
                    _tmp["sigma_max"] = _sigma_max
                    _tmp["Reject"] = reject
                    rows.append(_tmp)
    d = pd.concat(rows, ignore_index=True)
    return (d,)


@app.cell
def _(
    ALPHA,
    MODELS_1,
    P_H1,
    RHOs_1,
    SR1_list,
    T,
    d,
    display,
    f1_score,
    np,
    pd,
    precision_score,
    recall_score,
):
    results = []
    for _rho in RHOs_1:
        for _name in MODELS_1:
            for _SR1 in SR1_list:
                _tmp = d[(d["rho"] == _rho) & (d["name"] == _name) & (d["SR1"] == _SR1)]
                y_true_strat = _tmp["H1"].values.astype(bool)
                y_pred_iter = _tmp["Reject"].values.astype(bool)
                mask_H0 = ~y_true_strat
                FPP = np.sum(y_pred_iter & mask_H0) / mask_H0.sum() if mask_H0.sum() > 0 else np.nan
                if y_pred_iter.any() and y_true_strat.any():
                    precision = precision_score(y_true_strat, y_pred_iter)
                    recall = recall_score(y_true_strat, y_pred_iter)
                    f1 = f1_score(y_true_strat, y_pred_iter)
                else:
                    precision = np.nan
                    recall = np.nan
                    f1 = np.nan
                it_group = _tmp.groupby("iteration")
                it_df = it_group["H1"].agg(any_H1=lambda x: x.any())
                it_rej = it_group["Reject"].first()
                mask_all_H0_iters = ~it_df["any_H1"].values
                FWER_emp = np.mean(it_rej.values[mask_all_H0_iters]) if mask_all_H0_iters.sum() > 0 else np.nan
                results.append(
                    {
                        "name": _name,
                        "rho": _rho,
                        "SR1": _SR1,
                        "T": T,
                        "P_H1": P_H1,
                        "gamma3": _tmp["gamma3"].mean(),
                        "gamma4": _tmp["gamma4"].mean(),
                        "SR_c": _tmp["SR_c"].mean(),
                        "H1_mean": _tmp["H1"].mean(),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "FPP": FPP,
                        "FWER_emp": FWER_emp,
                        "alpha": ALPHA,
                    }
                )
    results = pd.DataFrame(results).sort_values(["name", "rho", "SR1"]).reset_index(drop=True)
    results_rounded = results.round(3)
    results.to_csv("exhibit_5_corrected.csv", index=False)
    display(results_rounded)
    return


if __name__ == "__main__":
    app.run()
