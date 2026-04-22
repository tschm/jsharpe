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
    # FDR control under different processes
    """)
    return


@app.cell
def _():
    import logging

    import numpy as np
    import pandas as pd
    import scipy
    from sklearn.metrics import f1_score, precision_score, recall_score

    from jsharpe import (
        control_for_FDR,
        generate_autocorrelated_non_gaussian_data,
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
        control_for_FDR,
        f1_score,
        generate_autocorrelated_non_gaussian_data,
        np,
        pd,
        precision_score,
        recall_score,
        scipy,
    )


@app.cell
def _():
    SR0 = 0
    SR1_list = [0.15, 0.3, 0.45, 0.6]
    T = 60
    REPS = 1000
    TRIALS = 10  # Does not play much role here: we do not take the maximum of K trials, we keep all of them (so we actually have REPS*TRIALS samples)
    # However, the sample skewness and kurtosis are computed on TRIALS samples
    P_H1 = 0.10
    Q = 0.25  # Desired FDR
    return P_H1, Q, REPS, SR0, SR1_list, T, TRIALS


@app.cell
def _():
    if False:
        MODELS_1 = ["gaussian"]  # For debugging
        RHOs_1 = [0]
    return MODELS_1, RHOs_1


@app.cell
def _(
    LOG,
    MODELS_1,
    P_H1,
    Q,
    REPS,
    RHOs_1,
    SR0,
    SR1_list,
    T,
    TRIALS,
    control_for_FDR,
    generate_autocorrelated_non_gaussian_data,
    np,
    pd,
    scipy,
):
    d = []
    for _rho in RHOs_1:
        for _name in MODELS_1:
            for _SR1 in SR1_list:
                LOG(f"{_rho} {_name} {_SR1}")
                for i in range(REPS):
                    H1 = np.random.uniform(size=TRIALS) < P_H1
                    H1.sort()
                    X0 = X1 = None
                    if H1.sum() < TRIALS:
                        X0 = generate_autocorrelated_non_gaussian_data(
                            T, TRIALS - H1.sum(), rho=_rho, SR0=SR0, name=_name
                        )
                    if H1.sum() > 0:
                        X1 = generate_autocorrelated_non_gaussian_data(T, H1.sum(), rho=_rho, SR0=_SR1, name=_name)
                    if X0 is None:
                        X = X1
                    elif X1 is None:
                        X = X0
                    else:
                        X = np.concatenate([X0, X1], axis=1)
                    gamma3 = scipy.stats.skew(X.flatten())
                    gamma4 = scipy.stats.kurtosis(X.flatten(), fisher=False)
                    SR = X.mean(axis=0) / X.std(axis=0)
                    alpha, beta, SR_c, _q_hat = control_for_FDR(
                        Q, SR0=SR0, SR1=_SR1, p_H1=P_H1, T=T, gamma3=gamma3, gamma4=gamma4, rho=_rho, K=1
                    )
                    _tmp = pd.DataFrame({"SR": SR, "H1": H1, "SR>SR_c": SR_c < SR})
                    _tmp["rho"] = _rho
                    _tmp["name"] = _name
                    _tmp["SR1"] = _SR1
                    _tmp["gamma3"] = gamma3
                    _tmp["gamma4"] = gamma4
                    _tmp["iteration"] = i
                    _tmp["SR_c"] = SR_c
                    _tmp["alpha"] = alpha
                    _tmp["beta"] = beta
                    d.append(_tmp)
    d = pd.concat(d)
    d
    return (d,)


@app.cell
def _(
    MODELS_1,
    Q,
    RHOs_1,
    SR1_list,
    T,
    d,
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
                y_true = _tmp["H1"]
                y_pred = _tmp["SR"] > _tmp["SR_c"]
                FDP = np.sum(y_pred & ~y_true) / np.sum(y_pred)
                results.append(
                    {
                        "name": _name,
                        "rho": _rho,
                        "SR1": _SR1,
                        "T": T,
                        "gamma3": _tmp["gamma3"].mean(),
                        "gamma4": _tmp["gamma4"].mean(),
                        "precision": precision_score(y_true, y_pred),
                        "recall": recall_score(y_true, y_pred),
                        "f1": f1_score(y_true, y_pred),
                        "FDP": FDP,
                        "q": Q,
                        "FDP-q": FDP - Q,
                        "SR_c": _tmp["SR_c"].mean(),
                        "H1": _tmp["H1"].mean(),
                    }
                )
    results = pd.DataFrame(results)
    results.sort_values(["name", "rho", "SR1"], inplace=True)
    results.reset_index(drop=True, inplace=True)
    results.to_csv("exhibit_7.csv", index=False)
    results.round(2)
    return


if __name__ == "__main__":
    app.run()
