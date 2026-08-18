"""Exploratory Data Analysis (EDA) module.

This page analyzes whatever dataset was selected in the Dataset Explorer. It
provides an automatic EDA summary, interactive Plotly visualizations for
numeric and categorical columns, missing-value and correlation analysis, and
an interpretation section for every chart.

Automatically generated observations are always labeled as *educational
hints*, never as conclusions about the student's research question.
"""

import streamlit as st

from utils import (
    get_module,
    render_education,
    render_page_header,
    render_page_link,
    render_page_sidebar,
    render_sidebar_footer,
)
from utils.data_analysis import eda_summary, missing_values, top_correlations
from utils.session import get_current_dataset, get_current_dataset_name
from utils.visualization import (
    CHART_EXPLAINERS,
    categorical_hints,
    correlation_hints,
    create_bar_chart,
    create_box_plot,
    create_correlation_matrix,
    create_count_plot,
    create_density_plot,
    create_frequency_distribution,
    create_histogram,
    create_missing_values_plot,
    create_pairwise_plot,
    create_scatter_plot,
    histogram_hints,
    box_plot_hints,
    missing_hints,
    pairwise_hints,
    scatter_hints,
    bar_chart_hints,
    select_categorical_columns,
    select_numeric_columns,
)

_MODULE = get_module("EDA")

NUMERICAL_CHARTS = ["Histogram", "Box plot", "Density plot", "Scatter plot"]
CATEGORICAL_CHARTS = ["Bar chart", "Frequency distribution", "Count plot"]
AGGREGATIONS = ["mean", "sum", "median", "min", "max", "count"]
CHART_AREAS = [
    "Numerical",
    "Categorical",
    "Correlation matrix",
    "Missing values",
    "Pairwise relationships",
]


def render_dataset_banner(df, name: str | None) -> None:
    """Show a compact banner describing the dataset under analysis."""
    st.caption(
        f"Analyzing: **{name or 'current dataset'}** "
        f"({df.shape[0]:,} rows x {df.shape[1]} columns)"
    )


def render_auto_summary(df) -> None:
    """Render the automatic EDA summary with clear hint labeling."""
    st.subheader("Automatic EDA summary")

    summary = eda_summary(df)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", summary["rows"])
    col_b.metric("Columns", summary["columns"])
    col_c.metric("Missing cells", summary["missing_cells"])
    col_d.metric("Duplicate rows", summary["duplicate_rows"])

    st.markdown(
        f"- **{summary['numeric_columns']}** numeric and "
        f"**{summary['categorical_columns']}** non-numeric columns"
    )
    if summary["constant_columns"]:
        st.markdown(
            "- Constant columns (single value): "
            + ", ".join(f"`{c}`" for c in summary["constant_columns"])
        )
    if summary["high_cardinality_columns"]:
        st.markdown(
            "- High-cardinality text columns (many unique values): "
            + ", ".join(f"`{c}`" for c in summary["high_cardinality_columns"])
        )

    st.warning(
        "This summary is computed automatically from the data. Use it as a "
        "starting point for your own exploration — the numbers are factual "
        "observations, not conclusions about what the data means."
    )

    render_education(
        "What is EDA?",
        "Exploratory Data Analysis is the process of inspecting and summarizing "
        "a dataset before formal modeling. It answers questions like: What "
        "variables exist? How are they distributed? Are there missing values, "
        "outliers, or strong relationships? The goal is to build intuition, not "
        "to prove a hypothesis.",
    )


def render_plot(fig, key: str) -> None:
    """Render a Plotly figure with a stable key and stretch width."""
    st.plotly_chart(fig, width="stretch", key=key)


def render_interpretation(hints: list[str], explainer_key: str) -> None:
    """Render the interpretation section: hints plus an educational explainer."""
    st.markdown("#### Interpretation")
    st.info(
        "The observations below are **automatically generated educational "
        "hints**. They describe what the chart shows, but they are not "
        "conclusions about your research question."
    )
    for hint in hints:
        st.markdown(f"- {hint}")
    render_education("How to interpret this chart", CHART_EXPLAINERS[explainer_key])


def render_numerical(df, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    """Render the numerical visualization builder."""
    chart_type = st.selectbox("Chart type", NUMERICAL_CHARTS, key="eda_num_chart")

    x_col = st.selectbox("X column (numeric)", numeric_cols, key="eda_num_x")
    color_col = st.selectbox(
        "Color by (optional)",
        ["None"] + categorical_cols,
        key="eda_num_color",
    )
    color = None if color_col == "None" else color_col

    y_col = None
    if chart_type in ("Box plot", "Scatter plot"):
        remaining = [c for c in numeric_cols if c != x_col] or numeric_cols
        y_col = st.selectbox(
            "Y column (numeric)",
            remaining,
            key="eda_num_y",
            index=0,
        )

    try:
        if chart_type == "Histogram":
            fig = create_histogram(df, x_col)
            hints = histogram_hints(df, x_col)
        elif chart_type == "Box plot":
            if y_col is None:
                st.info("Add a numeric column to plot a box plot.")
                return
            fig = create_box_plot(df, y=y_col, x=color)
            hints = box_plot_hints(df, y_col, x=color)
        elif chart_type == "Density plot":
            fig = create_density_plot(df, x_col)
            hints = histogram_hints(df, x_col)
        else:  # Scatter plot
            if y_col is None or y_col == x_col:
                st.info("Pick two different numeric columns for a scatter plot.")
                return
            fig = create_scatter_plot(df, x_col, y_col, color=color)
            hints = scatter_hints(df, x_col, y_col)
    except (ValueError, KeyError) as exc:
        st.error(f"Could not create the {chart_type.lower()}: {exc}")
        return

    render_plot(fig, key=f"eda_plot_{chart_type.lower().replace(' ', '_')}")
    render_interpretation(hints, chart_type)


def render_categorical(df, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    """Render the categorical visualization builder."""
    chart_type = st.selectbox("Chart type", CATEGORICAL_CHARTS, key="eda_cat_chart")
    x_col = st.selectbox("X column (categorical)", categorical_cols, key="eda_cat_x")

    aggregation = None
    y_col = None
    if chart_type == "Bar chart":
        if not numeric_cols:
            st.info("This dataset has no numeric column to aggregate.")
            return
        y_col = st.selectbox("Y column (numeric)", numeric_cols, key="eda_cat_y")
        aggregation = st.selectbox(
            "Aggregation",
            AGGREGATIONS,
            key="eda_cat_agg",
        )

    try:
        if chart_type == "Bar chart":
            fig = create_bar_chart(df, x_col, y_col, aggregation)
            hints = bar_chart_hints(df, x_col, y_col, aggregation)
        elif chart_type == "Frequency distribution":
            fig = create_frequency_distribution(df, x_col)
            hints = categorical_hints(df, x_col)
        else:  # Count plot
            fig = create_count_plot(df, x_col)
            hints = categorical_hints(df, x_col)
    except (ValueError, KeyError) as exc:
        st.error(f"Could not create the {chart_type.lower()}: {exc}")
        return

    render_plot(fig, key=f"eda_plot_{chart_type.lower().replace(' ', '_')}")
    render_interpretation(hints, chart_type)


def render_correlation(df, numeric_cols: list[str]) -> None:
    """Render the correlation matrix and strongest-pairs analysis."""
    if len(numeric_cols) < 2:
        st.info("A correlation analysis needs at least two numeric columns.")
        return

    try:
        fig = create_correlation_matrix(df)
    except ValueError as exc:
        st.error(str(exc))
        return
    render_plot(fig, key="eda_plot_correlation")

    st.markdown("**Strongest correlations**")
    top = top_correlations(df)
    if top.empty:
        st.info("No computable correlations between numeric columns.")
    else:
        st.dataframe(top, width="stretch")

    render_interpretation(correlation_hints(df), "Correlation matrix")


def render_missing(df) -> None:
    """Render the missing-values visualization."""
    fig = create_missing_values_plot(df)
    if fig is None:
        st.success("No missing values found in this dataset.")
        hints = missing_hints(df)
    else:
        render_plot(fig, key="eda_plot_missing")
        st.dataframe(missing_values(df), width="stretch")
        hints = missing_hints(df)

    render_interpretation(hints, "Missing values")


def render_pairwise(df, numeric_cols: list[str]) -> None:
    """Render the pairwise relationship matrix."""
    if len(numeric_cols) < 2:
        st.info("A pairwise analysis needs at least two numeric columns.")
        return

    selected = numeric_cols[:4]
    if len(numeric_cols) > 4:
        st.caption(
            "Showing the first 4 numeric columns to keep the plot readable: "
            + ", ".join(f"`{c}`" for c in selected)
        )

    try:
        fig = create_pairwise_plot(df, selected)
    except (ValueError, KeyError) as exc:
        st.error(f"Could not create the pairwise plot: {exc}")
        return

    render_plot(fig, key="eda_plot_pairwise")
    render_interpretation(pairwise_hints(df, selected), "Pairwise relationships")


def render_visualization_builder(df) -> None:
    """Render the main interactive visualization builder."""
    st.subheader("Interactive visualization")

    numeric_cols = select_numeric_columns(df)
    categorical_cols = select_categorical_columns(df)

    if not numeric_cols and not categorical_cols:
        st.info("No plottable columns found in this dataset.")
        return

    area = st.selectbox("What do you want to explore?", CHART_AREAS, key="eda_area")

    if area == "Numerical":
        if not numeric_cols:
            st.info("This dataset has no numeric columns to plot.")
            return
        render_numerical(df, numeric_cols, categorical_cols)
    elif area == "Categorical":
        if not categorical_cols:
            st.info("This dataset has no non-numeric columns to plot.")
            return
        render_categorical(df, numeric_cols, categorical_cols)
    elif area == "Correlation matrix":
        render_correlation(df, numeric_cols)
    elif area == "Missing values":
        render_missing(df)
    else:
        render_pairwise(df, numeric_cols)


def main() -> None:
    """Assemble the EDA page."""
    render_page_sidebar(_MODULE)
    render_page_header(_MODULE.title, _MODULE.subtitle, help_text=_MODULE.help_text)

    df = get_current_dataset()
    name = get_current_dataset_name()

    if df is None:
        st.info(
            "No dataset loaded yet. Open the **Dataset Explorer** in the sidebar "
            "to upload a dataset or load one of the sample datasets first."
        )
        render_page_link("pages/1_Dataset_Explorer.py", "Go to Dataset Explorer")
        render_sidebar_footer()
        return

    render_dataset_banner(df, name)
    render_auto_summary(df)
    st.markdown("---")
    render_visualization_builder(df)

    render_sidebar_footer()


main()
