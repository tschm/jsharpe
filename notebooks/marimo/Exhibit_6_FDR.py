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
    # FDR
    """)
    return


app._unparsable_cell(
    r"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from functions import *
    from tqdm.auto import tqdm
    """,
    name="_",
)


@app.cell
def _():
    import ray

    ray.init()
    return (ray,)


@app.cell
def _(control_for_FDR, np, pd, plt, ray, tqdm):
    results = {}

    for rho in [0, 0.2, 0.5, 0.8]:
        T = 24
        gamma3 = -2.448
        gamma4 = 10.164

        SR0 = 0
        SR1 = 0.2
        SR2 = 0.5

        q = 0.25

        xs = np.unique(
            sorted(
                np.hstack(
                    [
                        # More points close to zero, to show the vertical asymptote
                        np.logspace(-12, -2, 11),
                        np.linspace(0, 0.01, 11),
                        np.linspace(0, 1, 101),
                    ]
                )
            )
        )[1:-1]

        @ray.remote
        def control_for_FDR_ray(q, SR0, SR1, p_H1, T, gamma3, gamma4, rho, K):
            return control_for_FDR(q=q, SR0=SR0, SR1=SR1, p_H1=p_H1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)

        ys = [
            control_for_FDR_ray.remote(
                q=q, SR0=SR0, SR1=SR1, p_H1=p_H1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=1
            )
            for p_H1 in xs
        ]
        ys = [ray.get(y) for y in tqdm(ys)]
        ys = pd.DataFrame(np.array(ys), index=xs, columns=["alpha", "beta", "SR_c", "q_hat"])

        ys2 = [
            control_for_FDR_ray.remote(
                q=q, SR0=SR0, SR1=SR2, p_H1=p_H1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=1
            )
            for p_H1 in xs
        ]
        ys2 = [ray.get(y) for y in tqdm(ys2)]
        ys2 = pd.DataFrame(np.array(ys2), index=xs, columns=["alpha", "beta", "SR_c", "q_hat"])

        results[rho] = {
            "xs": xs,
            "ys": ys,
            "ys2": ys2,
        }

        _fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained", dpi=300)
        ax.plot(ys.index, ys["SR_c"], linewidth=3, label=f"SR0 = {SR0}, SR1 = {SR1}")
        ax.plot(ys2.index, ys2["SR_c"], linewidth=3, label=f"SR0 = {SR0}, SR1 = {SR2}")
        ax.set_ylabel("SR_c")
        ax.set_xlabel("p[H1]")
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_title(f"SR_c vs P[H1] (q={q})")
        if rho != 0:
            ax.set_title(f"SR_c vs P[H1] (q={q}, ρ={rho})")
        ax.legend()
        ax.axvline(1 - q, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(-0.5, 1.6)
        plt.show()
    return SR0, SR1, SR2, q, results


@app.cell
def _(SR0, SR1, SR2, plt, q, results):
    _fig_1, ax_1 = plt.subplots(figsize=(5, 3.5), layout="constrained", dpi=300)
    rho_1 = 0
    xs_1 = results[rho_1]["xs"]
    ys_1 = results[rho_1]["ys"]
    ys2_1 = results[rho_1]["ys2"]
    ax_1.plot(xs_1, ys_1["SR_c"], linewidth=3, label=f"SR0 = {SR0}, SR1 = {SR1}, ρ={rho_1}", color="tab:blue")
    ax_1.plot(xs_1, ys2_1["SR_c"], linewidth=3, label=f"SR0 = {SR0}, SR1 = {SR2}, ρ={rho_1}", color="tab:orange")
    rho_1 = 0.2
    xs_1 = results[rho_1]["xs"]
    ys_1 = results[rho_1]["ys"]
    ys2_1 = results[rho_1]["ys2"]
    ax_1.plot(
        xs_1, ys_1["SR_c"], linewidth=3, label=f"SR0 = {SR0}, SR1 = {SR1}, ρ={rho_1}", color="tab:blue", linestyle=":"
    )
    ax_1.plot(
        xs_1,
        ys2_1["SR_c"],
        linewidth=3,
        label=f"SR0 = {SR0}, SR1 = {SR2}, ρ={rho_1}",
        color="tab:orange",
        linestyle=":",
    )
    ax_1.set_ylabel("SR_c")
    ax_1.set_xlabel("p[H1]")
    ax_1.axhline(0, color="black", linestyle=":", linewidth=1)
    ax_1.set_xlim(0, 1)
    ax_1.set_title(f"SR_c vs P[H1] (q={q})")
    if rho_1 != 0:
        ax_1.set_title(f"SR_c vs P[H1] (q={q})")
    ax_1.legend()
    ax_1.axvline(1 - q, color="black", linestyle="--", linewidth=1)
    ax_1.set_ylim(-0.5, 1.6)
    plt.show()
    return


@app.cell
def _(control_for_FDR, expected_maximum_sharpe_ratio, np, pd, ray, tqdm):
    T_1 = 24
    gamma3_1 = -2.448
    gamma4_1 = 10.164
    SR0_1 = 0
    SR1_1 = 0.5
    q_1 = 0.25
    number_of_trials = 10
    variance = 0.1
    SR0_adj = expected_maximum_sharpe_ratio(number_of_trials, variance)
    xs_2 = np.unique(sorted(np.hstack([np.logspace(-12, -2, 11), np.linspace(0, 0.01, 11), np.linspace(0, 1, 101)])))[
        1:-1
    ]

    @ray.remote
    def control_for_FDR_ray_1(q, SR0, SR1, p_H1, T, gamma3, gamma4, K):
        return control_for_FDR(q=q, SR0=SR0, SR1=SR1, p_H1=p_H1, T=T, gamma3=gamma3, gamma4=gamma4, K=K)

    ys_2 = [
        control_for_FDR_ray_1.remote(
            q=q_1, SR0=SR0_1, SR1=SR1_1, p_H1=p_H1, T=T_1, gamma3=gamma3_1, gamma4=gamma4_1, K=1
        )
        for p_H1 in xs_2
    ]
    ys_2 = [ray.get(y) for y in tqdm(ys_2)]  # More points close to zero, to show the vertical asymptote
    ys_2 = pd.DataFrame(np.array(ys_2), index=xs_2, columns=["alpha", "beta", "SR_c", "q_hat"])
    ys2_2 = [
        control_for_FDR_ray_1.remote(
            q=q_1,
            SR0=SR0_adj,
            SR1=SR1_1 + SR0_adj,
            p_H1=p_H1,
            T=T_1,
            gamma3=gamma3_1,
            gamma4=gamma4_1,
            K=number_of_trials,
        )
        for p_H1 in xs_2
    ]
    ys2_2 = [ray.get(y) for y in tqdm(ys2_2)]
    ys2_2 = pd.DataFrame(np.array(ys2_2), index=xs_2, columns=["alpha", "beta", "SR_c", "q_hat"])
    return SR0_1, SR0_adj, SR1_1, number_of_trials, q_1, ys2_2, ys_2


@app.cell
def _(SR0_1, SR0_adj, SR1_1, plt, q_1, ys2_2, ys_2):
    fig_2, axs = plt.subplots(1, 3, figsize=(9, 2.5), layout="constrained")
    axs[0].plot(ys_2.index, ys_2["alpha"], label="alpha")
    axs[1].plot(ys_2.index, ys_2["beta"], label="beta")
    axs[2].plot(ys_2.index, ys_2["SR_c"], label="SR_c")
    axs[2].plot(ys2_2.index, ys2_2["SR_c"], label="SR_c_adj")
    axs[0].set_ylabel("alpha")
    axs[1].set_ylabel("beta")
    axs[2].set_ylabel("SR_c")
    axs[0].set_ylim(-0.04, 1.04)
    axs[1].set_ylim(-0.04, 1.04)
    axs[2].set_ylim(-0.5, 1.6)
    for ax_2 in axs:
        ax_2.set_xlabel("p[H1]")
        ax_2.axhline(0, color="black", linestyle=":", linewidth=1)
        ax_2.set_xlim(0, 1)
        ax_2.axvline(1 - q_1, color="black", linestyle="--", linewidth=1)
    fig_2.suptitle(f"SR0 = {SR0_1}, SR1 = {SR1_1}, q = {q_1}\nSR0_adj = {SR0_adj:.3f}, SR1_adj = {SR1_1 + SR0_adj:.3f}")
    plt.show()
    return


@app.cell
def _(SR0_1, SR0_adj, SR1_1, number_of_trials, plt, q_1, ys2_2, ys_2):
    _fig_3, ax_3 = plt.subplots(figsize=(5, 3.5), layout="constrained", dpi=300)
    ax_3.plot(ys_2.index, ys_2["SR_c"], linewidth=3, label=f"SR0 = {SR0_1}, SR1 = {SR1_1}, K=1")
    ax_3.plot(
        ys2_2.index,
        ys2_2["SR_c"],
        linewidth=3,
        label=f"SR0 = {SR0_adj:.3f}, SR1 = {SR1_1 + SR0_adj:.3f}, K={number_of_trials}",
    )
    ax_3.set_ylabel("SR_c")
    ax_3.set_xlabel("p[H1]")
    ax_3.axhline(0, color="black", linestyle=":", linewidth=1)
    ax_3.set_xlim(0, 1)
    ax_3.set_title(f"SR_c vs P[H1] (q={q_1})")
    ax_3.legend()
    ax_3.axvline(1 - q_1, color="black", linestyle="--", linewidth=1)
    ax_3.set_ylim(-0.5, 1.6)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
