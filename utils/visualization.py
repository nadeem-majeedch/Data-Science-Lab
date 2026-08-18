"""Reusable Plotly visualization builders and interpretation helpers for EDA.

Every chart builder is a pure function that takes a pandas DataFrame (plus
column names) and returns a ``plotly.graph_objects.Figure``, with no Streamlit
dependency, which keeps them easy to unit-test.

The ``*_hints`` functions compute simple, factual observations about the data
behind a chart. They are meant to be shown as *educational hints*, never as
authoritative conclusions.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_analysis import categorical_columns, numeric_columns, top_correlations

MAX_PAIRWISE_COLUMNS = 4


def select_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric column names available for analysis."""
    return numeric_columns(df)


def select_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return the non-numeric column names available for analysis."""
    return categorical_columns(df)


def _num(value: float, ndigits: int = 2) -> str:
    try:
        return f"{value:.{ndigits}f}"
    except (TypeError, ValueError):
        return str(value)


# Numerical charts ------------------------------------------------------------


def create_histogram(df: pd.DataFrame, x: str, bins: int = 30) -> go.Figure:
    """Create an interactive histogram of a numeric column."""
    fig = px.histogram(
        df,
        x=x,
        nbins=bins,
        title=f"Histogram of {x}",
        labels={"count": "Count"},
    )
    fig.update_layout(xaxis_title=x, yaxis_title="Count")
    return fig


def create_box_plot(df: pd.DataFrame, y: str, x: str | None = None) -> go.Figure:
    """Create a box plot of a numeric column, optionally grouped by a category."""
    if x and x in df.columns:
        fig = px.box(df, x=x, y=y, color=x, title=f"Box plot of {y} by {x}")
    else:
        fig = px.box(df, y=y, title=f"Box plot of {y}")
    fig.update_layout(xaxis_title=x or "", yaxis_title=y)
    return fig


def create_density_plot(df: pd.DataFrame, x: str) -> go.Figure:
    """Create a smoothed density (KDE) plot of a numeric column."""
    series = df[x].dropna()
    if len(series) < 2:
        raise ValueError(f"'{x}' needs at least two non-missing values for a density plot.")

    from scipy.stats import gaussian_kde

    kde = gaussian_kde(series.to_numpy())
    xs = np.linspace(series.min(), series.max(), 300)
    ys = kde(xs)

    fig = go.Figure(
        go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy", name=x)
    )
    fig.update_layout(
        title=f"Density plot of {x}",
        xaxis_title=x,
        yaxis_title="Density",
        showlegend=False,
    )
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
) -> go.Figure:
    """Create an interactive scatter plot between two numeric columns."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        title=f"Scatter plot of {y} vs {x}",
    )
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig


def create_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Create a correlation heatmap of the numeric columns.

    Raises:
        ValueError: If fewer than two numeric columns are available.
    """
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        raise ValueError("A correlation matrix needs at least two numeric columns.")

    corr = numeric.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation matrix",
    )
    return fig


# Categorical charts ----------------------------------------------------------


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    aggregation: str = "mean",
) -> go.Figure:
    """Create a bar chart of a numeric column aggregated per category."""
    grouped = (
        df.groupby(x, observed=True)[y].agg(aggregation).dropna().reset_index()
    )
    grouped.columns = [x, y]
    fig = px.bar(
        grouped,
        x=x,
        y=y,
        title=f"{aggregation.title()} of {y} by {x}",
    )
    fig.update_layout(xaxis_title=x, yaxis_title=f"{aggregation.title()} of {y}")
    return fig


def create_frequency_distribution(df: pd.DataFrame, x: str) -> go.Figure:
    """Create a bar chart of category frequencies, with percentages."""
    counts = df[x].value_counts(dropna=False)
    percentages = counts / counts.sum() * 100

    fig = go.Figure(
        go.Bar(
            x=counts.index.astype(str),
            y=counts.to_numpy(),
            text=[f"{_num(p, 1)}%" for p in percentages],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Frequency distribution of {x}",
        xaxis_title=x,
        yaxis_title="Frequency",
    )
    return fig


def create_count_plot(df: pd.DataFrame, x: str) -> go.Figure:
    """Create a horizontal bar chart of category counts (count plot)."""
    counts = df[x].value_counts(dropna=True).sort_values()
    fig = px.bar(
        x=counts.to_numpy(),
        y=counts.index.astype(str),
        orientation="h",
        title=f"Count plot of {x}",
        labels={"x": "Count", "y": x},
    )
    fig.update_layout(xaxis_title="Count", yaxis_title=x)
    return fig


# Additional charts -----------------------------------------------------------


def create_missing_values_plot(df: pd.DataFrame) -> go.Figure | None:
    """Create a bar chart of missing values per column.

    Returns ``None`` when the dataset has no missing values.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None

    fig = px.bar(
        x=missing.index.astype(str),
        y=missing.to_numpy(),
        title="Missing values per column",
        labels={"x": "Column", "y": "Missing values"},
    )
    fig.update_layout(xaxis_title="Column", yaxis_title="Missing values")
    return fig


def create_pairwise_plot(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    """Create a pairwise scatter matrix with histograms on the diagonal.

    At most ``MAX_PAIRWISE_COLUMNS`` columns are shown to stay readable.
    """
    selected = list(columns)[:MAX_PAIRWISE_COLUMNS]
    fig = px.scatter_matrix(
        df,
        dimensions=selected,
        title="Pairwise relationships",
    )
    return fig


# Educational interpretation helpers ------------------------------------------


def histogram_hints(df: pd.DataFrame, x: str) -> list[str]:
    """Return educational observations about a histogram."""
    series = df[x].dropna()
    if len(series) < 2:
        return [
            f"Educational hint: '{x}' has too few non-missing values to describe its distribution."
        ]

    mean, median, std = series.mean(), series.median(), series.std()
    missing = int(df[x].isna().sum())

    hints: list[str] = []
    if std == 0:
        hints.append(
            f"Educational hint: '{x}' is constant — the standard deviation is 0, "
            "so every value is identical."
        )
    elif abs(mean - median) / std < 0.1:
        hints.append(
            f"Educational hint: The distribution of '{x}' looks roughly symmetric: "
            f"the mean ({_num(mean)}) is close to the median ({_num(median)})."
        )
    elif mean > median:
        hints.append(
            f"Educational hint: The distribution of '{x}' appears right-skewed: "
            f"the mean ({_num(mean)}) is above the median ({_num(median)}). A few "
            "unusually large values may be pulling the mean upward."
        )
    else:
        hints.append(
            f"Educational hint: The distribution of '{x}' appears left-skewed: "
            f"the mean ({_num(mean)}) is below the median ({_num(median)}). A few "
            "unusually small values may be pulling the mean downward."
        )
    hints.append(
        f"Educational hint: Values of '{x}' spread roughly within {_num(2 * std)} "
        "units of the mean. For a bell-shaped distribution about 95% of values "
        "fall within two standard deviations."
    )
    if missing:
        hints.append(
            f"Educational hint: '{x}' has {missing} missing values that are "
            "excluded from this chart."
        )
    return hints


def box_plot_hints(df: pd.DataFrame, y: str, x: str | None = None) -> list[str]:
    """Return educational observations about a box plot."""
    series = df[y].dropna()
    if len(series) == 0:
        return [f"Educational hint: '{y}' has no non-missing values to plot."]

    q1, median, q3 = series.quantile([0.25, 0.50, 0.75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower_fence) | (series > upper_fence)]

    hints = [
        f"Educational hint: The box spans the middle 50% of '{y}' (from "
        f"{_num(q1)} to {_num(q3)}); the line inside is the median ({_num(median)})."
    ]
    if iqr == 0:
        hints.append(
            f"Educational hint: The interquartile range is 0, so at least half of "
            "the values are identical."
        )
    elif len(outliers) == 0:
        hints.append(
            f"Educational hint: No values fall beyond 1.5 x IQR, so no points are "
            "flagged as outliers."
        )
    else:
        hints.append(
            f"Educational hint: {len(outliers)} values fall beyond 1.5 x IQR and are "
            "shown as dots. Decide whether they are genuine extremes or data errors."
        )

    if x and x in df.columns:
        medians = df.groupby(x, observed=True)[y].median().dropna()
        if len(medians) > 1:
            highest, lowest = medians.idxmax(), medians.idxmin()
            hints.append(
                f"Educational hint: The median of '{y}' differs across '{x}': "
                f"highest for '{highest}' ({_num(medians[highest])}) and lowest for "
                f"'{lowest}' ({_num(medians[lowest])})."
            )
    return hints


def scatter_hints(df: pd.DataFrame, x: str, y: str) -> list[str]:
    """Return educational observations about a scatter plot."""
    paired = df[[x, y]].dropna()
    if len(paired) < 3:
        return [
            f"Educational hint: Too few paired non-missing values between '{x}' "
            "and '{y}' to estimate a relationship."
        ]

    correlation = paired[x].corr(paired[y])
    if pd.isna(correlation):
        return [
            f"Educational hint: The correlation between '{x}' and '{y}' could not "
            "be computed, usually because one column is constant."
        ]

    direction = "positive" if correlation > 0 else "negative"
    magnitude = abs(correlation)
    if magnitude < 0.3:
        strength = "weak"
    elif magnitude < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    hints = [
        f"Educational hint: The sample Pearson correlation between '{x}' and "
        f"'{y}' is {_num(correlation)}, indicating a {direction}, {strength} "
        "linear relationship in this dataset.",
        "Educational hint: Correlation does not imply causation — the association "
        "could be coincidental or driven by another variable.",
    ]
    excluded = len(df) - len(paired)
    if excluded:
        hints.append(
            f"Educational hint: {excluded} rows with missing values were excluded "
            "from this chart."
        )
    return hints


def correlation_hints(df: pd.DataFrame, k: int = 5) -> list[str]:
    """Return educational observations about the correlation analysis."""
    top = top_correlations(df, k)
    if top.empty:
        return [
            "Educational hint: Not enough numeric columns with variability to "
            "rank correlations."
        ]

    pairs = [
        f"'{row['Column A']}' and '{row['Column B']}' ({row['Correlation']:+.2f})"
        for _, row in top.iterrows()
    ]
    return [
        "Educational hint: The strongest correlations in this dataset are "
        + "; ".join(pairs)
        + ".",
        "Educational hint: Correlations capture only linear associations; "
        "non-linear relationships may be missed, and correlation does not imply "
        "causation.",
    ]


def bar_chart_hints(df: pd.DataFrame, x: str, y: str, aggregation: str) -> list[str]:
    """Return educational observations about an aggregated bar chart."""
    grouped = df.groupby(x, observed=True)[y].agg(aggregation).dropna()
    if grouped.empty:
        return [
            f"Educational hint: No values could be aggregated for '{y}' grouped by "
            f"'{x}'."
        ]

    highest, lowest = grouped.idxmax(), grouped.idxmin()
    return [
        f"Educational hint: The {aggregation} of '{y}' is highest for "
        f"'{highest}' ({_num(grouped[highest])}) and lowest for '{lowest}' "
        f"({_num(grouped[lowest])}).",
        "Educational hint: Aggregated bars summarize each group with one number "
        "and hide the spread inside the group — check a box plot for the full "
        "picture.",
    ]


def categorical_hints(df: pd.DataFrame, x: str) -> list[str]:
    """Return educational observations about a frequency or count chart."""
    counts = df[x].value_counts(dropna=False)
    if counts.empty:
        return [f"Educational hint: '{x}' has no values to count."]

    total = int(counts.sum())
    top = counts.index[0]
    top_pct = counts.iloc[0] / total * 100
    hints = [
        f"Educational hint: '{x}' has {int(counts.nunique())} distinct values; "
        f"the most frequent is '{top}' ({_num(top_pct, 1)}% of values)."
    ]
    missing = int(df[x].isna().sum())
    if missing:
        hints.append(
            f"Educational hint: '{x}' has {missing} missing values, which are "
            "shown separately if present in the chart."
        )
    return hints


def missing_hints(df: pd.DataFrame) -> list[str]:
    """Return educational observations about the missing-value chart."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return ["Educational hint: This dataset has no missing values."]

    total = int(missing.sum())
    worst = missing.sort_values(ascending=False).head(3)
    columns = ", ".join(f"'{c}' ({v})" for c, v in worst.items())
    return [
        f"Educational hint: {total} cells are missing in total. The columns with "
        f"the most missing values are {columns}.",
        "Educational hint: High-missing columns may need imputation or removal "
        "before modeling — the Data Preprocessing module covers both.",
    ]


def pairwise_hints(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return educational observations about a pairwise scatter matrix."""
    return [
        f"Educational hint: Each panel compares two of the selected numeric "
        f"columns ({', '.join(columns)}); the diagonal shows each column's "
        "distribution.",
        "Educational hint: Look for clear linear trends, clusters, or outliers "
        "that stand out across several panels — those are promising patterns to "
        "investigate further.",
    ]


# Static educational explainers ------------------------------------------------

CHART_EXPLAINERS: dict[str, str] = {
    "Histogram": (
        "A **histogram** groups a numeric column into bins and counts how many "
        "values fall into each bin. The shape tells you about the distribution: "
        "is it centered, spread out, symmetric, or skewed? It is the first thing "
        "to check for any numeric variable."
    ),
    "Box plot": (
        "A **box plot** summarizes a numeric column with five numbers: minimum, "
        "first quartile, median, third quartile, and maximum. The box holds the "
        "middle 50% of the data and the whiskers extend to the most extreme "
        "non-outlier points. Dots beyond the whiskers are potential outliers."
    ),
    "Density plot": (
        "A **density plot** is a smoothed version of a histogram. The area under "
        "the curve always sums to 1, so it shows the shape of the distribution "
        "independent of bin width. Use it alongside a histogram to compare "
        "distributions."
    ),
    "Scatter plot": (
        "A **scatter plot** places each row at the intersection of two numeric "
        "columns. It reveals how two variables move together: upward trends "
        "(positive), downward trends (negative), no pattern (independent), or "
        "curved / clustered patterns that linear correlation would miss."
    ),
    "Correlation matrix": (
        "The **correlation matrix** shows the Pearson correlation between every "
        "pair of numeric columns. Values range from -1 (perfect negative linear "
        "relationship) to +1 (perfect positive), with 0 meaning no linear "
        "relationship. Strongly correlated pairs may be redundant for modeling."
    ),
    "Bar chart": (
        "A **bar chart** aggregates a numeric column by category (for example, "
        "the mean score per subject) and draws one bar per category. It is the "
        "best way to compare a numeric measure across groups. The chosen "
        "aggregation (mean, sum, median, ...) changes what each bar represents."
    ),
    "Frequency distribution": (
        "A **frequency distribution** counts how often each category occurs and "
        "shows the percentage above each bar. It reveals the balance of your "
        "categories: is one category dominant, or are they evenly distributed?"
    ),
    "Count plot": (
        "A **count plot** is a horizontal bar chart of how many rows belong to "
        "each category. It is the fastest way to see the size of each group and "
        "to spot rare or dominant categories."
    ),
    "Missing values": (
        "This chart shows how many values are missing per column. Missing data "
        "is a fact of real-world datasets; the key decisions are whether a "
        "column is too incomplete to use and how to fill the gaps. You will "
        "practice both in the Data Preprocessing module."
    ),
    "Pairwise relationships": (
        "A **pairwise scatter matrix** plots every combination of the selected "
        "numeric columns, with each column's distribution on the diagonal. It "
        "gives a fast overview of which variables are related before you dig "
        "into individual pairs."
    ),
}
