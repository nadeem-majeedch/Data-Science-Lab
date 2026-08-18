"""Tests for the dataset analysis utilities."""

import pandas as pd
import pytest

from utils.data_analysis import (
    categorical_columns,
    categorical_stats,
    constant_columns,
    data_quality_score,
    dataframe_info,
    duplicate_rows,
    format_bytes,
    head,
    memory_usage_bytes,
    missing_values,
    numeric_columns,
    numeric_stats,
    quality_report,
    random_sample,
    tail,
    unique_counts,
)


def build_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 25, 30, None],
            "score": [10.0, 20.0, 10.0, 20.0, 50.0],
            "name": ["Ana", "Bob", "Ana", "Cid", "Eve"],
            "city": ["X", "Y", "X", "Y", None],
            "status": ["active"] * 5,
        }
    )


def test_dataframe_info_keys():
    info = dataframe_info(build_sample_frame())
    assert set(info) == {"rows", "columns", "memory_bytes", "dtypes"}
    assert info["rows"] == 5
    assert info["columns"] == 5


def test_memory_usage_bytes_is_positive():
    assert memory_usage_bytes(build_sample_frame()) > 0


def test_column_classification():
    df = build_sample_frame()
    assert numeric_columns(df) == ["age", "score"]
    assert set(categorical_columns(df)) == {"name", "city", "status"}


def test_head_tail_and_random_sample():
    df = build_sample_frame()
    assert len(head(df)) == 5
    assert len(head(df, 2)) == 2
    assert len(tail(df, 2)) == 2
    sample = random_sample(df, 3)
    assert len(sample) == 3
    assert random_sample(df, 3).equals(random_sample(df, 3))


def test_numeric_stats_shape_and_columns():
    stats = numeric_stats(build_sample_frame())
    assert {"Column", "Missing"}.issubset(stats.columns)
    assert set(stats["Column"]) == {"age", "score"}
    assert "Missing" in stats.columns


def test_numeric_stats_empty_without_numeric():
    df = pd.DataFrame({"name": ["a", "b"]})
    assert numeric_stats(df).empty


def test_categorical_stats_contains_top_and_frequency():
    stats = categorical_stats(build_sample_frame())
    row = stats[stats["Column"] == "name"].iloc[0]
    assert row["Top"] == "Ana"
    assert row["Frequency"] == 2
    assert row["Unique"] == 4


def test_missing_values_counts_and_percentages():
    report = missing_values(build_sample_frame())
    age_row = report[report["Column"] == "age"].iloc[0]
    assert age_row["Missing count"] == 1
    assert age_row["Missing %"] == 20.0
    score_row = report[report["Column"] == "score"].iloc[0]
    assert score_row["Missing count"] == 0


def test_duplicate_rows_count():
    df = build_sample_frame()
    # rows 0 and 2 (25 / 10.0 / Ana / X) are exact duplicates.
    assert duplicate_rows(df) == 1


def test_unique_counts():
    df = build_sample_frame()
    assert unique_counts(df)["status"] == 1
    assert unique_counts(df)["age"] == 2
    assert unique_counts(df)["name"] == 4


def test_constant_columns_detection():
    df = build_sample_frame()
    assert constant_columns(df) == ["status"]


def test_quality_report_marks_constant_columns():
    report = quality_report(build_sample_frame())
    status_row = report[report["Column"] == "status"].iloc[0]
    assert status_row["Constant"]
    age_row = report[report["Column"] == "age"].iloc[0]
    assert not age_row["Constant"]
    assert {"Column", "Missing count", "Missing %", "Unique values"}.issubset(
        report.columns
    )


def test_data_quality_score():
    df = build_sample_frame()
    # 2 missing cells (one in age, one in city) out of 25.
    assert data_quality_score(df) == pytest.approx(1 - 2 / 25)


def test_data_quality_score_empty():
    assert data_quality_score(pd.DataFrame()) == 0.0


def test_format_bytes():
    assert format_bytes(0) == "0.0 bytes"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(1024 * 1024) == "1.0 MB"
