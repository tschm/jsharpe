# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.13.15",
#     "numpy==2.3.0",
#     "matplotlib==3.10.0",
#     "altair==5.5.0",
#     "vega-datasets==0.9.0",
#     "pyarrow",
# ]
# ///

"""Demo marimo notebook for slide presentations.

This module demonstrates how to create interactive slide presentations with marimo.
"""

import marimo

__generated_with = "0.13.11-dev14"
app = marimo.App()

with app.setup:
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from vega_datasets import data


@app.cell(hide_code=True)
def _(leaves):
    mo.md(
        f"""
        # marimo slides! {"🍃" * leaves.value}

        **A presentation by the marimo team.**

        {leaves}
        """
    ).left().center()
    return


@app.cell
def _():
    leaves = mo.ui.slider(1, 12)
    return (leaves,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **marimo** is an open-source Python notebook that's reproducible, shareable as an app, and executable as a script.

    Install it with `pip install marimo`, and learn more at [GitHub](https://github.com/marimo-team/marimo).

    Starting with **v0.7.2**, every marimo notebook can be shared as an interactive slide deck.

    **In fact, this slide deck was made with a marimo notebook!** To present your notebook as slides,
    just toggle the app view and choose the slide layout in the top right.
    Every cell with an output becomes its own slide.

    Next are a couple of examples of dynamic slides.
    """
    )
    return


@app.cell
def _():
    mo.md(r"## Dynamic plots").center()
    return


@app.cell
def _():
    exponent = mo.ui.slider(1, 3, label="exponent")
    return (exponent,)


@app.cell
def _(exponent):
    x = np.linspace(-3, 3, 100)
    y = x**exponent.value

    mo.md(
        rf"""
        {exponent}

        \[
        y = x^{exponent.value}
        \]

        {mo.as_html(make_plot(x, y))}
        """
    ).center()
    return


@app.function
def make_plot(x, y):
    """Create a matplotlib plot of x vs y.

    Args:
        x: Array of x values
        y: Array of y values

    Returns:
        Matplotlib axes object with the plot
    """
    plt.plot(x, y)
    plt.ylim(-27, 27)
    plt.xlim(-3, 3)
    return plt.gca()


@app.cell
def _():
    mo.md("""# Select data in plots""")
    return


@app.cell
def _():
    chart = mo.ui.altair_chart(
        alt.Chart(data.cars())
        .mark_circle(size=60)
        .encode(
            x="Horsepower",
            y="Miles_per_Gallon",
            color="Origin",
            tooltip=["Name", "Origin", "Horsepower", "Miles_per_Gallon"],
        )
    )
    return (chart,)


@app.cell
def _(chart):
    mo.vstack([mo.md("**Click and drag to select data.**"), chart, chart.value])
    return


@app.cell
def _():
    mo.md(r"""## Thanks!""")
    return


if __name__ == "__main__":
    app.run()
