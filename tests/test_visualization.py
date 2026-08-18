"""Tests for the EDA visualization and interpretation utilities."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from utils.data_analysis import eda_summary, top_correlations
from utils.visualization import (
    CHART_EXPLAINERS,
    bar_chart_hints,
    box_plot_hints,
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
    missing_hints,
    pairwise_hints,
    scatter_hints,
    select_categorical_columns,
    select_numeric_columns,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "samples"


def load_student_grades() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "student_grades.csv")


def build_mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 22, 21, 23, None, 24],
            "score": [50.0, 80.0, 90.0, 70.0, 60.0, None],
            "city": ["A", "B", "A", "C", "B", "A"],
            "notes": ["ok", None, "good", "ok", "ok", "good"],
        }
    )


def test_column_selectors():
    df = build_mixed_frame()
    assert select_numeric_columns(df) == ["age", "score"]
    assert set(select_categorical_columns(df)) == {"city", "notes"}


def test_create_histogram_returns_figure():
    df = build_mixed_frame()
    fig = create_histogram(df, "score")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert "score" in fig.layout.xaxis.title.text


def test_create_box_plot_with_and_without_grouping():
    df = build_mixed_frame()
    plain = create_box_plot(df, y="score")
    assert isinstance(plain, go.Figure)
    grouped = create_box_plot(df, y="score", x="city")
    assert isinstance(grouped, go.Figure)


def test_create_density_plot():
    df = build_mixed_frame()
    fig = create_density_plot(df, "age")
    assert isinstance(fig, go.Figure)


def test_create_density_plot_needs_two_values():
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError):
        create_density_plot(df, "x")


def test_create_scatter_plot():
    df = build_mixed_frame()
    fig = create_scatter_plot(df, "age", "score")
    assert isinstance(fig, go.Figure)


def test_create_correlation_matrix():
    df = build_mixed_frame()
    fig = create_correlation_matrix(df)
    assert isinstance(fig, go.Figure)


def test_create_correlation_matrix_needs_two_numeric_columns():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(ValueError):
        create_correlation_matrix(df)


def test_create_bar_chart_aggregation_values():
    df = build_mixed_frame()
    fig = create_bar_chart(df, "city", "age", aggregation="mean")
    assert isinstance(fig, go.Figure)
    # City A has ages 20, 21, 24 -> mean 21.666...
    x_values = list(fig.data[0].x)
    y_values = list(fig.data[0].y)
    index = x_values.index("A")
    assert y_values[index] == pytest.approx((20 + 21 + 24) / 3)


def test_create_frequency_distribution():
    df = build_mixed_frame()
    fig = create_frequency_distribution(df, "city")
    assert isinstance(fig, go.Figure)
    # City A appears 3 times (including NaN handling: no NaN here).
    x_values = list(fig.data[0].x)
    y_values = list(fig.data[0].y)
    assert y_values[x_values.index("A")] == 3


def test_create_count_plot():
    df = build_mixed_frame()
    fig = create_count_plot(df, "city")
    assert isinstance(fig, go.Figure)


def test_create_missing_values_plot_none_when_complete():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert create_missing_values_plot(df) is None


def test_create_missing_values_plot_detects_missing():
    df = build_mixed_frame()
    fig = create_missing_values_plot(df)
    assert isinstance(fig, go.Figure)


def test_create_pairwise_plot_limits_columns():
    df = load_student_grades()
    fig = create_pairwise_plot(df, ["attendance_pct", "midterm", "final"])
    assert isinstance(fig, go.Figure)


def test_all_hint_functions_return_prefixed_observations():
    df = load_student_grades()
    hint_sets = [
        histogram_hints(df, "midterm"),
        box_plot_hints(df, "final", x="subject"),
        scatter_hints(df, "midterm", "final"),
        correlation_hints(df),
        bar_chart_hints(df, "subject", "final", "mean"),
        categorical_hints(df, "subject"),
        missing_hints(df),
        pairwise_hints(df, ["midterm", "final"]),
    ]
    for hints in hint_sets:
        assert hints, "expected at least one hint"
        for hint in hints:
            assert hint.startswith("Educational hint:")


def test_chart_explainers_cover_all_chart_types():
    for kind in [
        "Histogram",
        "Box plot",
        "Density plot",
        "Scatter plot",
        "Correlation matrix",
        "Bar chart",
        "Frequency distribution",
        "Count plot",
        "Missing values",
        "Pairwise relationships",
    ]:
        assert kind in CHART_EXPLAINERS


def test_top_correlations_shape_and_order():
    df = build_mixed_frame()
    top = top_correlations(df)
    assert set(top.columns) == {"Column A", "Column B", "Correlation"}
    # age/score correlation: computed over 5 paired values.
    paired = df[["age", "score"]].dropna()
    expected = paired["age"].corr(paired["score"])
    assert top.iloc[0]["Correlation"] == pytest.approx(expected)


def test_eda_summary_returns_expected_keys():
    df = build_mixed_frame()
    summary = eda_summary(df)
    assert summary["rows"] == 6
    assert summary["columns"] == 4
    assert summary["missing_cells"] == 3
    assert summary["numeric_columns"] == 2
    assert summary["categorical_columns"] == 2
    assert set(summary) >= {
        "rows",
        "columns",
        "missing_cells",
        "duplicate_rows",
        "numeric_columns",
        "categorical_columns",
        "constant_columns",
        "high_cardinality_columns",
        "dtypes",
    }
