import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import math

    import functions  # <- required local module
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    EULER_GAMMA = 0.5772156649015328606

    # ------------------------------------------------------------
    # Exact (Numerical Integration) — uses functions.moments_Mk
    # ------------------------------------------------------------
    def sd_numerical_integration(K: int) -> float:
        """Exact Std[M_K] via Gauss–Hermite integration implemented in functions.py."""
        _, _, var = functions.moments_Mk(K, rho=0)
        return math.sqrt(max(0.0, var))

    # ------------------------------------------------------------
    # EVT Quantile Normalization
    # ------------------------------------------------------------
    def u1_u2_delta(K: int):
        p1 = 1.0 - 1.0 / K
        p2 = 1.0 - 1.0 / (K * math.e)
        u1 = norm.ppf(p1)
        u2 = norm.ppf(p2)
        return u1, u2, (u2 - u1)

    # ------------------------------------------------------------
    # Classical EVT approximation
    # Var ≈ (π² / 6) Δ²
    # ------------------------------------------------------------
    def sd_evt_standard(K: int) -> float:
        _, _, delta = u1_u2_delta(K)
        return abs(delta) * math.sqrt((math.pi**2) / 6.0)

    # ------------------------------------------------------------
    # Corrected (FST-style) approximation — Eq. (75)
    # Var ≈ Δ² (π²/6 − γ²/(1+γ))
    # ------------------------------------------------------------
    def sd_fst_style(K: int) -> float:
        _, _, delta = u1_u2_delta(K)
        c = (math.pi**2) / 6.0 - (EULER_GAMMA**2) / (1.0 + EULER_GAMMA)
        return abs(delta) * math.sqrt(max(0.0, c))

    # ------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------
    def sd_monte_carlo(K: int, n_mc: int = 120_000, seed: int = 123) -> float:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(size=(n_mc, K))
        m = np.max(x, axis=1)
        return float(np.std(m, ddof=0))

    # ------------------------------------------------------------
    # Plot Exhibit 13
    # ------------------------------------------------------------
    def plot_exhibit_13(K_min: int = 2, K_max: int = 100, n_mc: int = 120_000, seed: int = 123):
        Ks = np.arange(K_min, K_max + 1)

        sd_exact = np.array([sd_numerical_integration(int(K)) for K in Ks])
        sd_fst = np.array([sd_fst_style(int(K)) for K in Ks])
        sd_evt = np.array([sd_evt_standard(int(K)) for K in Ks])
        sd_mc = np.array([sd_monte_carlo(int(K), n_mc=n_mc, seed=seed + int(K)) for K in Ks])

        # Match paper styling
        plt.figure(figsize=(9, 5.5))
        plt.plot(Ks, sd_exact, color="black", linewidth=4.0, label="Numerical integration (exact)")
        plt.plot(Ks, sd_fst, color="green", linewidth=2.0, label="FST-style approximation")
        plt.plot(Ks, sd_evt, color="orange", linewidth=2.0, label="Standard EVT approximation")
        plt.plot(Ks, sd_mc, color="red", linestyle="--", linewidth=2.0, label="Monte Carlo")

        plt.xlabel("K")
        plt.ylabel("Std. dev. of max{X1,...,XK}")
        plt.title("Exhibit 13 – Standard Deviation of the Maximum")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_exhibit_13()
    return


if __name__ == "__main__":
    app.run()
