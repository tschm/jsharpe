import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from jsharpe import moments_Mk

    return moments_Mk, np, pd, plt


@app.cell
def _(moments_Mk, np, plt):
    ks = np.arange(1, 101)
    variances = [moments_Mk(k)[2] for k in ks]
    _fig, _ax = plt.subplots(figsize=(5, 3.5), dpi=300, layout="constrained")
    _ax.scatter(ks, variances)
    _ax.set_ylim(0, 1.04)
    _ax.set_xlim(0, 100)
    _ax.set_ylabel("Variance of the maximum\nof k iid standard Gaussians")
    _ax.set_xlabel("k")
    plt.show()
    return ks, variances


@app.cell
def _(ks, np, plt, variances):
    _fig, _ax = plt.subplots(figsize=(5, 3.5), dpi=100, layout="constrained")
    _ax.scatter(ks, np.sqrt(variances))
    # ax.set_yscale('log')
    _ax.set_xscale("log")
    _ax.set_ylim(0, 1.04)
    # ax.set_xlim(0,100)
    _ax.set_ylabel("Standard deviation of the maximum\nof k iid standard Gaussians")
    _ax.set_xlabel("k")
    _ax.set_xticks([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100], [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100])
    _ax.set_xticks([1, 10, 100], [1, 10, 100])
    # ax.set_xticks(
    #  [2,3,4,5,6,7,8,9,     20,30,40,50,60,70,80,90],
    #  [2,3,4,5,'','','','', 20,30,40,50,'','','',''],
    #  minor = True, fontsize = 8,
    #   )
    plt.show()
    return


@app.cell
def _(ks, np, pd, variances):
    import os
    import tempfile

    _out = os.path.join(tempfile.gettempdir(), "variances.csv")
    pd.DataFrame(
        {
            "k": ks,
            "variance": variances,
            "std": np.sqrt(variances),
        }
    ).to_csv(_out, index=False)
    return


if __name__ == "__main__":
    app.run()
