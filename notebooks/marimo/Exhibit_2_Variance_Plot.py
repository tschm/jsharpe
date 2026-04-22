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
    # Plot the variance as a function of the autocorrelation
    Compare different estimators of the variance (Gaussian, Gaussian + autocorrelation, Non-Gaussian iid, Non-Gaussian + autocorrelation) with a simulation
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats
    from tqdm.auto import tqdm

    from jsharpe import generate_autocorrelated_non_gaussian_data, sharpe_ratio_variance

    return (
        generate_autocorrelated_non_gaussian_data,
        np,
        pd,
        plt,
        scipy,
        sharpe_ratio_variance,
        tqdm,
    )


@app.cell
def _():
    import ray

    ray.init()
    return (ray,)


@app.cell
def _(np):
    def legend_thick(ax, *args, **kwargs):
        leg = _ax.legend(*args, **kwargs)
        for _i in leg.legend_handles:
            _i.set_linewidth(7)
            _i.set_solid_capstyle("butt")

    def remove_empty_axes(axs: np.ndarray) -> None:
        for _ax in _axs.flatten():
            if not _ax.lines and (not _ax.collections) and (not _ax.has_data()):
                _ax.axis("off")

    return legend_thick, remove_empty_axes


@app.cell
def _():
    SR0_list = [0, 0.15, 0.30, 0.45, 0.60]
    return (SR0_list,)


@app.cell
def _(SR0_list, np, plt, remove_empty_axes, sharpe_ratio_variance):
    T = 24
    _gamma3 = -2.448
    _gamma4 = 10.164
    rhos = np.linspace(0, 0.5, 100)
    _fig, _axs = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for _SR, _ax in zip(SR0_list, _axs.flatten(), strict=False):
        _SR = _SR / np.sqrt(12)
        variances1 = [sharpe_ratio_variance(_SR, T, K=1) for rho in rhos]
        variances2 = [sharpe_ratio_variance(_SR, T, gamma3=_gamma3, gamma4=_gamma4, K=1) for rho in rhos]
        variances3 = [sharpe_ratio_variance(_SR, T, rho=rho, K=1) for rho in rhos]
        variances4 = [sharpe_ratio_variance(_SR, T, gamma3=_gamma3, gamma4=_gamma4, rho=rho, K=1) for rho in rhos]
        _ax.plot(rhos, variances1, label="Gaussian")
        _ax.plot(rhos, variances3, label="Gaussian + autocorrelation")
        _ax.plot(rhos, variances2, label="Non-Gaussian iid")
        _ax.plot(rhos, variances4, label="Non-Gaussian + autocorrelation")
        _ax.axhline(0, color="black", linewidth=1)
        _handles, _labels = _ax.get_legend_handles_labels()
        _ax.legend(_handles[::-1], _labels[::-1])
        _ax.set_xlabel("Autocorrelation")
        _ax.set_ylabel("Variance")
        _ax.set_title(f"Variance of the Sharpe Ratio (SR={_SR:.2f})")
    remove_empty_axes(_axs)
    plt.show()
    return T, rhos


@app.cell
def _(
    SR0_list,
    T,
    generate_autocorrelated_non_gaussian_data,
    np,
    pd,
    rhos,
    scipy,
    sharpe_ratio_variance,
    tqdm,
):
    variances = []
    for _SR in SR0_list:
        _SR = _SR / np.sqrt(12)
        for rho in tqdm(rhos):
            X = generate_autocorrelated_non_gaussian_data(T, 10000, SR0=_SR, name="severe", rho=rho)
            _gamma3 = scipy.stats.skew(X.flatten())
            _gamma4 = scipy.stats.kurtosis(X.flatten(), fisher=False)
            T_1 = X.shape[0]
            variances.append(
                {
                    "SR": _SR,
                    "rho": rho,
                    "simulation": np.var(X.mean(axis=0) / X.std(axis=0)),
                    "Gaussian": sharpe_ratio_variance(_SR, T_1),
                    "Gaussian + autocorrelation": sharpe_ratio_variance(_SR, T_1, rho=rho),
                    "Non-Gaussian iid": sharpe_ratio_variance(_SR, T_1, gamma3=_gamma3, gamma4=_gamma4),
                    "Non-Gaussian + autocorrelation": sharpe_ratio_variance(
                        _SR, T_1, gamma3=_gamma3, gamma4=_gamma4, rho=rho
                    ),
                }
            )
    variances = pd.DataFrame(variances)
    return T_1, variances


@app.cell
def _(T_1, legend_thick, plt, remove_empty_axes, variances):
    _fig, _axs = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for _i, _SR in enumerate(variances["SR"].unique()):
        _ax = _axs.flatten()[_i]
        _tmp = variances[variances["SR"] == _SR]
        for _column in _tmp.columns[2:]:
            _ax.plot(_tmp["rho"], _tmp[_column], label=_column)
        _ax.axhline(0, linewidth=1, color="black", linestyle=":")
        _ax.set_ylim(0, variances.iloc[:, 2:].max().max() * 1.05)
        if _i == 0:
            legend_thick(_ax)
        _ax.set_xlabel("Autocorrelation")
        _ax.set_ylabel("Variance")
        _ax.set_title(f"SR={_SR:.2f}")
    remove_empty_axes(_axs)
    _fig.suptitle(f"Variance of the Sharpe ratio (T={T_1})")
    plt.show()
    return


@app.cell
def _(
    SR0_list,
    generate_autocorrelated_non_gaussian_data,
    np,
    pd,
    ray,
    scipy,
    sharpe_ratio_variance,
    tqdm,
):
    @ray.remote
    def f3(SR, rho, T, name):
        X = generate_autocorrelated_non_gaussian_data(T, 10000, SR0=_SR, name=name, rho=rho)
        _gamma3 = scipy.stats.skew(X.flatten())
        _gamma4 = scipy.stats.kurtosis(X.flatten(), fisher=False)
        return {
            "name": name,
            "SR": _SR,
            "rho": rho,
            "simulation": np.var(X.mean(axis=0) / X.std(axis=0)),
            "Gaussian iid": sharpe_ratio_variance(_SR, T),
            "Gaussian + autocorrelation": sharpe_ratio_variance(_SR, T, rho=rho),
            "Non-Gaussian iid": sharpe_ratio_variance(_SR, T, gamma3=_gamma3, gamma4=_gamma4),
            "Non-Gaussian + autocorrelation": sharpe_ratio_variance(_SR, T, gamma3=_gamma3, gamma4=_gamma4, rho=rho),
        }

    YEARS = 5
    PERIODS_PER_YEAR = 12
    T_2 = YEARS * PERIODS_PER_YEAR
    rhos_1 = np.linspace(0, 0.5, 100)
    variances_1 = [
        f3.remote(_SR, rho, T_2, name)
        for _SR in SR0_list
        for rho in rhos_1
        for name in ["gaussian", "mild", "moderate", "severe"]
    ]
    variances_1 = [ray.get(v) for v in tqdm(variances_1)]
    variances_1 = pd.DataFrame(variances_1)
    return T_2, variances_1


@app.cell
def _(T_2, legend_thick, np, plt, remove_empty_axes, variances_1):
    for which in ["Variance", "Standard deviation"]:
        for name in variances_1["name"].unique():
            s = 0.7
            _fig, _axs = plt.subplots(2, 3, figsize=(s * 12, s * 6), layout="constrained", dpi=300)
            for _i, _SR in enumerate(variances_1["SR"].unique()):
                _ax = _axs.flatten()[_i]
                i1 = variances_1["SR"] == _SR
                i2 = variances_1["name"] == name
                _tmp = variances_1[i1 & i2].copy()
                m = variances_1[i2].iloc[:, 3:].max().max()
                if which == "Standard deviation":
                    m = np.sqrt(m)
                    _tmp.iloc[:, 3:] = np.sqrt(_tmp.iloc[:, 3:])
                for _column in _tmp.columns[3:]:
                    _ax.plot(
                        _tmp["rho"],
                        _tmp[_column],
                        label=_column,
                        color="black" if _column == "simulation" else None,
                        linewidth=3 if _column == "simulation" else None,
                    )
                _ax.axhline(0, linewidth=1, color="black", linestyle=":")
                _ax.set_ylim(0, m * 1.05)
                if _i == 0:
                    _handles, _labels = _ax.get_legend_handles_labels()
                _ax.set_xlabel("Autocorrelation")
                _ax.set_ylabel(which)
                _ax.set_title(f"SR={_SR:.2f}")
            remove_empty_axes(_axs)
            _ax = _axs.flatten()[-1]
            legend_thick(_ax, _handles, _labels, loc="center")
            _fig.suptitle(f"{which} of the Sharpe ratio (distribution={name}, T={T_2})")
            _fig.savefig(f"Exhibit_2_{which}_{name}_T={T_2}.png", dpi=300)
            plt.show()
    return


if __name__ == "__main__":
    app.run()
