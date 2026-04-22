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
    # Moments, autocorrelation and non-normality of hedge fund returns
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    from statsmodels.stats.diagnostic import acorr_ljungbox

    # CSV file with hedge fund daily returns; columns: Date,HFRIFWI Index,HFRIEHI Index,HFRIEDI Index,HFRIRVA Index,HFRIMI Index
    # (We cannot redistribute this file: it is not included in the repository.)
    CSV_PATH = "HFR.csv"

    # Exhibit 1 columns / order (paper uses these Bloomberg-style names)
    COL_ORDER = [
        "HFRIFWI Index",  # Composite
        "HFRIEHI Index",  # Equity Hedge
        "HFRIEDI Index",  # Event-Driven
        "HFRIRVA Index",  # Relative Value
        "HFRIMI Index",  # Macro
    ]
    COL_LABELS = {
        "HFRIFWI Index": "Composite",
        "HFRIEHI Index": "Equity Hedge",
        "HFRIEDI Index": "Event-Driven",
        "HFRIRVA Index": "Relative Value",
        "HFRIMI Index": "Macro",
    }

    def summarize_exhibit_1(x: pd.Series) -> dict:
        x = x.dropna().astype(float).values
        T = len(x)

        mean = float(np.mean(x))
        stdev = float(np.std(x, ddof=1))  # sample stdev

        # skewness and Pearson kurtosis (kurtosis=3 under Normality)
        skew = float(st.skew(x, bias=False))
        kurt = float(st.kurtosis(x, fisher=False, bias=False))

        # AR(1): sample autocorrelation at lag 1
        ar1 = float(pd.Series(x).autocorr(lag=1))

        # Jarque–Bera
        jb = st.jarque_bera(x)
        jb_stat = float(jb.statistic)
        jb_p = float(jb.pvalue)

        # Ljung–Box at 10 lags
        lb = acorr_ljungbox(x, lags=[10], return_df=True)
        lb10_stat = float(lb["lb_stat"].iloc[0])
        lb10_p = float(lb["lb_pvalue"].iloc[0])

        return {
            "Mean": mean,
            "StDev": stdev,
            "Skew": skew,
            "Kurt": kurt,
            "AR(1)": ar1,
            "T": T,
            "JB (stat)": jb_stat,
            "JB (p)": jb_p,
            "LB-10 (stat)": lb10_stat,
            "LB-10 (p)": lb10_p,
        }

    def format_like_exhibit(df_stats: pd.DataFrame) -> pd.DataFrame:
        out = df_stats.copy()
        # round to 3 decimals like the exhibit; show p-values to 3 decimals (so they print as 0.000)
        for c in ["Mean", "StDev", "Skew", "Kurt", "AR(1)", "JB (stat)", "LB-10 (stat)"]:
            out[c] = out[c].round(3)
        for c in ["JB (p)", "LB-10 (p)"]:
            out[c] = out[c].apply(lambda v: 0.0 if v < 0.0005 else round(v, 3))
        out["T"] = out["T"].astype(int)
        return out

    # --- Load data ---
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # If you want to enforce the paper’s sample window explicitly:
    # df = df[(df["Date"] >= "1990-01-01") & (df["Date"] <= "2025-11-30")]

    # --- Compute Exhibit 1 table ---
    stats_by_col = {col: summarize_exhibit_1(df[col]) for col in COL_ORDER}
    table = pd.DataFrame(stats_by_col).rename(columns=COL_LABELS)

    # Match exhibit layout: rows are statistics, columns are indices
    table = table.loc[
        ["Mean", "StDev", "Skew", "Kurt", "AR(1)", "T", "JB (stat)", "JB (p)", "LB-10 (stat)", "LB-10 (p)"],
        ["Composite", "Equity Hedge", "Event-Driven", "Relative Value", "Macro"],
    ]

    # Print a rounded version (matches Exhibit 1 rounding)
    print(format_like_exhibit(table.T).T)

    # Optional: save to CSV for inclusion in a report
    format_like_exhibit(table.T).T.to_csv("exhibit_1_reproduced.csv", index=True)
    return


if __name__ == "__main__":
    app.run()
