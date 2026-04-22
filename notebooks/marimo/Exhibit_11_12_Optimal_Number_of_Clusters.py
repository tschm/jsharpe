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
    # Effective number of trials
    """)
    return


app._unparsable_cell(
    r"""
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import ray
    from functions import *
    from tqdm.auto import tqdm
    """,
    name="_",
)


@app.cell
def _():
    FAST = True  # Set to False to have smoother / more stable plots
    return (FAST,)


@app.cell
def _(ray):
    ray.init()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Silhouette quality
    """)
    return


@app.cell
def _(get_random_correlation_matrix, np, number_of_clusters, plt):
    # This slightly under-estimates the number of clusters

    effective_number_of_trials = 10

    np.random.seed(1)
    for iteration in range(5):
        C, _, _ = get_random_correlation_matrix(noise=1, effective_number_of_trials=effective_number_of_trials)
        _k, qualities, _clusters = number_of_clusters(C)
        _fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout="constrained", dpi=300 if iteration == 0 else 100)
        ax = axs[0]
        ax.imshow(C, vmin=-1, vmax=+1, cmap="RdBu", aspect=1, interpolation="nearest")
        ax.axis("off")
        ax = axs[1]
        ax.plot(qualities)
        i = np.argmax(qualities)
        x, y = qualities.index[i], qualities.iloc[i]
        ax.scatter(x, y)
        ax.text(x, y, f"  {qualities.index[i]}", va="center", ha="left")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Quality")
        plt.show()
    return (effective_number_of_trials,)


@app.cell
def _(
    FAST,
    effective_number_of_trials,
    get_random_correlation_matrix,
    np,
    number_of_clusters,
    plt,
    ray,
    tqdm,
):
    @ray.remote
    def f():
        C = get_random_correlation_matrix(noise=1, effective_number_of_trials=10)[0]
        k, _qualities, _clusters = number_of_clusters(C)
        return k

    results = [f.remote() for _ in range(100 if FAST else 1000)]
    results = [ray.get(u) for u in tqdm(results)]
    results_1 = results  # 1 minute for 100
    _fig_1, ax_1 = plt.subplots(figsize=(3, 2), layout="constrained")
    ax_1.hist(
        results,
        bins=np.linspace(min(results) - 0.5, max(results) + 0.5, max(results) - min(results) + 2),
        facecolor="lightblue",
        edgecolor="tab:blue",
    )
    ax_1.axvline(effective_number_of_trials, color="black", linestyle="--")
    for side in ["left", "right", "top"]:
        ax_1.spines[side].set_visible(False)
    ax_1.set_yticks([])
    ax_1.set_xlabel("Estimated number of clusters")
    plt.show()
    return (results_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Effective rank
    """)
    return


@app.cell
def _(effective_rank, get_random_correlation_matrix, plt, tqdm):
    # The effective rank strongly over-estimates the number of clusters
    # This is sometimes what we want: if we only have 10 investment ideas, and tried a lot of noisy variants of them,
    # we have more than 10 independent trials -- the more we try, the more we are likely to find a good Sharpe ratio.
    N_iterations = 10000
    effective_number_of_trials_1 = 10
    results_4 = []
    for _ in tqdm(range(N_iterations)):
        C_1, X, _ = get_random_correlation_matrix(noise=1, effective_number_of_trials=effective_number_of_trials_1)
        results_4.append(effective_rank(C_1))
    results_2 = results_4
    _fig_2, ax_2 = plt.subplots(figsize=(3, 2), layout="constrained")
    ax_2.hist(results_4, facecolor="lightblue", edgecolor="tab:blue")
    ax_2.axvline(effective_number_of_trials_1, color="black", linestyle="--")
    for side_1 in ["left", "right", "top"]:
        ax_2.spines[side_1].set_visible(False)
    ax_2.set_yticks([])
    ax_2.set_xlabel("Effective Number of Trials")
    plt.show()
    return X, effective_number_of_trials_1, results_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random Matrix Theory (RMT)
    """)
    return


@app.cell
def _(X, math, np, plt):
    # Check that the distribution is correct
    e = []
    for _ in range(1000):
        X_1 = np.random.normal(size=X.shape)
        C_2 = np.corrcoef(X_1.T)
        C_2 = np.clip(C_2, -1, 1)
        np.fill_diagonal(C_2, 1)
        e.append(np.linalg.eigvalsh(C_2))
    e = np.hstack(e)
    lambda_ = X_1.shape[1] / X_1.shape[0]
    sigma = 1
    lambda_plus = sigma**2 * (1 + math.sqrt(lambda_)) ** 2
    lambda_minus = sigma**2 * (1 - math.sqrt(lambda_)) ** 2
    _fig_3, ax_3 = plt.subplots(figsize=(12, 2), layout="constrained")
    ax_3.hist(e, bins=200, density=True)
    ax_3.axvline(lambda_plus, linestyle="--", linewidth=1, color="black")
    xs = np.linspace(0, lambda_plus, 1000)
    ys = np.where(
        (xs > lambda_minus) & (xs < lambda_plus),
        1 / (2 * math.pi * sigma**2) * np.sqrt((lambda_plus - xs) * (xs - lambda_minus)) / (lambda_ * xs),
        0,
    )
    ax_3.plot(xs, ys, color="tab:orange", linewidth=3)
    for side_2 in ["left", "right", "top"]:
        ax_3.spines[side_2].set_visible(False)
    ax_3.set_yticks([])  # Marchenko-Pastur
    ax_3.set_xlabel("Distribution of the eigenvalues of the correlation matrix of N(0,I) data")
    plt.show()
    return


@app.cell
def _(
    effective_number_of_trials_1,
    get_random_correlation_matrix,
    math,
    np,
    plt,
):
    np.random.seed(0)
    C_3, X_2, _ = get_random_correlation_matrix(noise=1, effective_number_of_trials=effective_number_of_trials_1)
    lambda__1 = X_2.shape[1] / X_2.shape[0]
    sigma_1 = 1
    lambda_plus_1 = sigma_1**2 * (1 + math.sqrt(lambda__1)) ** 2
    lambda_minus_1 = sigma_1**2 * (1 - math.sqrt(lambda__1)) ** 2
    _fig_4, ax_4 = plt.subplots(figsize=(12, 2), layout="constrained")
    e_1 = np.linalg.eigvalsh(C_3)
    ax_4.hist(e_1, bins=100, density=True)
    ax_4.axvline(lambda_plus_1, linestyle="--", linewidth=1, color="black")
    xs_1 = np.linspace(0, lambda_plus_1, 100)
    ys_1 = np.where(
        (xs_1 > lambda_minus_1) & (xs_1 < lambda_plus_1),
        1 / (2 * math.pi * sigma_1**2) * np.sqrt((lambda_plus_1 - xs_1) * (xs_1 - lambda_minus_1)) / (lambda__1 * xs_1),
        0,
    )
    ax_4.plot(xs_1, ys_1, color="tab:orange", linewidth=3)
    for side_3 in ["left", "right", "top"]:  # Marchenko-Pastur
        ax_4.spines[side_3].set_visible(False)
    ax_4.set_yticks([])
    ax_4.set_ylabel("Eigenvalues")
    ax_4.set_title(
        f"Distribution of the eigenvalues of the correlation matrix\nEigenvalues beyond the Marchenko-Pastur limit: {np.sum(e_1 > lambda_plus_1)}"
    )
    plt.show()
    return


@app.cell
def _(math, np, scipy):
    def detone_numpy(V, keep_trace=True):
        """Remove the largest eigenvalue of a variance matrix."""
        trace = np.diag(V).sum()
        e, u = scipy.sparse.linalg.eigsh(V, 1)
        e = e[0]
        V = V - e * u @ u.T
        if keep_trace:
            missing = trace - np.diag(V).sum()
            V = V + missing * np.eye(V.shape[0]) / V.shape[0]
        return V

    def count_non_trivial_eigenvalues(X, naive=False):
        """Count the eigenvalues beyond the the Marchenko-Pastur limit,
        remove them, re-compute the eigenvalues, and iterate
        until there are no more eigenvalues beyond the limit.

        The "naive" version does this just once.

        This gives a lower bound on the number of effective trials.

        Inputs: X: numpy array, one column per trial
                naive: boolean
        Output: integer, number of non-trivial eigenvalues
        """
        C = np.corrcoef(X.T)
        C = np.clip(C, -1, 1)
        np.fill_diagonal(C, 1)
        lambda_ = X.shape[1] / X.shape[0]
        sigma = 1
        lambda_plus = sigma**2 * (1 + math.sqrt(lambda_)) ** 2
        sigma**2 * (1 - math.sqrt(lambda_)) ** 2
        count = 0
        while True:
            e = np.linalg.eigvalsh(C)
            k = np.sum(e > lambda_plus)
            if naive or k == 0:
                break
            count = count + k
            for _ in range(k):
                C = detone_numpy(C)
        return count

    return (count_non_trivial_eigenvalues,)


@app.cell
def _(
    count_non_trivial_eigenvalues,
    get_random_correlation_matrix,
    np,
    plt,
    tqdm,
):
    N_iterations_1 = 10000
    effective_number_of_trials_2 = 10
    results_5 = []
    for _ in tqdm(range(N_iterations_1)):
        _C_4, X_3, _ = get_random_correlation_matrix(noise=1, effective_number_of_trials=effective_number_of_trials_2)
        results_5.append(count_non_trivial_eigenvalues(X_3))
    results_5 = np.array(results_5).astype(int)
    results_3 = results_5
    _fig_5, ax_5 = plt.subplots(figsize=(3, 2), layout="constrained")
    ax_5.hist(
        results_5,
        facecolor="lightblue",
        edgecolor="tab:blue",
        bins=np.linspace(-0.5, max(results_5) + 0.5, max(results_5) + 2),
    )
    ax_5.axvline(effective_number_of_trials_2, color="black", linestyle="--")
    for side_4 in ["left", "right", "top"]:
        ax_5.spines[side_4].set_visible(False)
    ax_5.set_yticks([])
    ax_5.set_xticks(np.arange(0, 12, 2))
    ax_5.set_xlabel("Non-trivial eigenvalues\nof the correlation matrix")
    plt.show()
    return effective_number_of_trials_2, results_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plots
    """)
    return


@app.cell
def _(effective_number_of_trials_2, np, plt, results_1, results_2, results_3):
    fig_6, axs_1 = plt.subplots(1, 3, figsize=(9, 2), dpi=300)
    ax_6 = axs_1[0]
    ax_6.hist(
        results_1,
        bins=np.linspace(min(results_1) - 0.5, max(results_1) + 0.5, max(results_1) - min(results_1) + 2),
        facecolor="lightblue",
        edgecolor="tab:blue",
    )
    ax_6.axvline(effective_number_of_trials_2, color="black", linestyle="--")
    for side_5 in ["left", "right", "top"]:
        ax_6.spines[side_5].set_visible(False)
    ax_6.set_yticks([])
    ax_6.set_xticks(np.arange(0, 14, 2))
    ax_6.set_xlabel("Estimated number of clusters\nfrom the silhouette quality")
    ax_6 = axs_1[2]
    ax_6.hist(results_2, facecolor="lightblue", edgecolor="tab:blue")
    ax_6.axvline(effective_number_of_trials_2, color="black", linestyle="--")
    for side_5 in ["left", "right", "top"]:
        ax_6.spines[side_5].set_visible(False)
    ax_6.set_yticks([])
    ax_6.set_xticks(np.arange(0, 50, 10))
    ax_6.set_xlabel("Effective rank\n of the correlation matrix")
    ax_6 = axs_1[1]
    ax_6.hist(
        results_3,
        facecolor="lightblue",
        edgecolor="tab:blue",
        bins=np.linspace(-0.5, max(results_3) + 0.5, max(results_3) + 2),
    )
    ax_6.axvline(effective_number_of_trials_2, color="black", linestyle="--")
    for side_5 in ["left", "right", "top"]:
        ax_6.spines[side_5].set_visible(False)
    ax_6.set_yticks([])
    ax_6.set_xticks(np.arange(0, 14, 2))
    ax_6.set_xlabel("Non-trivial eigenvalues\nof the correlation matrix")
    fig_6.subplots_adjust(wspace=0.2)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
