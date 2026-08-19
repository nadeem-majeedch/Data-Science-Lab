"""Tests for the feature engineering utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils.feature_engineering import (
    BINNING_METHODS,
    DATETIME_PARTS,
    MATH_TRANSFORMS,
    NUMERIC_OPERATIONS,
    TEXT_PARTS,
    apply_feature_op,
    apply_math_transform,
    bin_numeric,
    binning_code,
    correlation_code,
    create_interaction,
    create_numeric_feature,
    create_polynomial_features,
    datetime_code,
    extract_datetime_features,
    extract_text_features,
    feature_importance,
    interaction_code,
    math_transform_code,
    numeric_feature_code,
    operation_effect,
    polynomial_code,
    select_by_correlation,
    select_by_variance,
    text_code,
    variance_code,
)


def build_numeric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [30000, 50000, 40000, 60000, 80000],
            "score": [80, 90, 70, 60, 95],
        }
    )


def build_mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "score": [80, 90, 70, 60, 95],
            "name": ["Ana Maria", "Bob", "Cid", "Dana", "Eve"],
            "joined": ["2020-01-15", "2021-03-22", "2019-11-02", "2022-07-08", "2023-12-25"],
        }
    )


# Numeric features -----------------------------------------------------------


def test_create_numeric_feature_sum():
    df = build_numeric_frame()
    result, info = create_numeric_feature(df, "age", "score", "sum")
    assert info["column"] == "age_plus_score"
    assert result["age_plus_score"].tolist() == [105, 120, 105, 100, 140]


def test_create_numeric_feature_difference_and_product():
    df = build_numeric_frame()
    result, _ = create_numeric_feature(df, "income", "age", "difference")
    assert result["income_minus_age"].tolist() == [29975, 49970, 39965, 59960, 79955]
    result, _ = create_numeric_feature(df, "age", "score", "product")
    assert result["age_times_score"].tolist() == [2000, 2700, 2450, 2400, 4275]


def test_create_numeric_feature_ratio():
    df = build_numeric_frame()
    result, _ = create_numeric_feature(df, "score", "age", "ratio")
    expected = [80 / 25, 90 / 30, 70 / 35, 60 / 40, 95 / 45]
    assert result["score_over_age"].tolist() == pytest.approx(expected)


def test_create_numeric_feature_custom_name():
    df = build_numeric_frame()
    result, info = create_numeric_feature(df, "age", "score", "sum", name="total")
    assert info["column"] == "total"
    assert "total" in result.columns


def test_create_numeric_feature_ratio_by_zero_raises():
    df = build_numeric_frame()
    df.loc[0, "age"] = 0
    with pytest.raises(ValueError):
        create_numeric_feature(df, "score", "age", "ratio")


# Math transforms ------------------------------------------------------------


def test_math_log_transform():
    df = build_numeric_frame()
    result, info = apply_math_transform(df, ["age", "income"], "log")
    assert info["columns"] == ["age_log", "income_log"]
    assert result["age_log"].tolist() == pytest.approx(np.log([25, 30, 35, 40, 45]))


def test_math_sqrt_transform():
    df = build_numeric_frame()
    result, _ = apply_math_transform(df, ["score"], "sqrt")
    assert result["score_sqrt"].tolist() == pytest.approx(np.sqrt([80, 90, 70, 60, 95]))


def test_math_square_transform():
    df = build_numeric_frame()
    result, _ = apply_math_transform(df, ["score"], "square")
    assert result["score_squared"].tolist() == [6400, 8100, 4900, 3600, 9025]


def test_math_log_negative_raises():
    df = build_numeric_frame()
    df.loc[0, "age"] = -5
    with pytest.raises(ValueError):
        apply_math_transform(df, ["age"], "log")


def test_math_sqrt_negative_raises():
    df = build_numeric_frame()
    df.loc[0, "score"] = -10
    with pytest.raises(ValueError):
        apply_math_transform(df, ["score"], "sqrt")


def test_math_originals_untouched():
    df = build_numeric_frame()
    result, _ = apply_math_transform(df, ["age"], "square")
    assert result["age"].equals(df["age"])


# Binning --------------------------------------------------------------------


def test_bin_equal_width():
    df = build_numeric_frame()
    result, info = bin_numeric(df, "age", n_bins=2, method="equal width")
    assert info["column"] == "age_binned"
    assert result["age_binned"].nunique() == 2


def test_bin_quantile():
    df = pd.DataFrame({"x": list(range(1, 21))})
    result, _ = bin_numeric(df, "x", n_bins=4, method="quantile")
    counts = result["x_binned"].value_counts()
    assert len(counts) == 4


def test_bin_missing_column_raises():
    with pytest.raises(ValueError):
        bin_numeric(build_numeric_frame(), "nope", n_bins=2)


# Date/time extraction -------------------------------------------------------


def test_datetime_extraction():
    df = build_mixed_frame()
    result, info = extract_datetime_features(
        df, "joined", ["year", "month", "day", "weekday"]
    )
    assert info["columns"] == [
        "joined_year",
        "joined_month",
        "joined_day",
        "joined_weekday",
    ]
    assert result["joined_year"].tolist() == [2020, 2021, 2019, 2022, 2023]
    # 2020-01-15 was a Wednesday -> weekday 2 (Monday=0)
    assert result["joined_weekday"].tolist() == [2, 0, 5, 4, 0]


def test_datetime_unparseable_raises():
    df = pd.DataFrame({"d": ["not-a-date", "also-not"]})
    with pytest.raises(ValueError):
        extract_datetime_features(df, "d", ["year"])


# Text features --------------------------------------------------------------


def test_text_length_and_word_count():
    df = build_mixed_frame()
    result, info = extract_text_features(df, "name", ["length", "word count"])
    assert info["columns"] == ["name_length", "name_word_count"]
    assert result["name_length"].tolist() == [9, 3, 3, 4, 3]
    assert result["name_word_count"].tolist() == [2, 1, 1, 1, 1]


# Interaction ----------------------------------------------------------------


def test_interaction_product():
    df = build_numeric_frame()
    result, info = create_interaction(df, "age", "score")
    assert info["column"] == "age_x_score"
    assert result["age_x_score"].tolist() == [2000, 2700, 2450, 2400, 4275]


# Polynomial features --------------------------------------------------------


def test_polynomial_features_degree_two():
    df = build_numeric_frame()
    result, info = create_polynomial_features(df, ["age", "score"], degree=2)
    assert info["columns"] == ["age_pow2", "age_x_score", "score_pow2"]
    assert result["age_pow2"].tolist() == [625, 900, 1225, 1600, 2025]
    assert "age" in result.columns  # original kept, not duplicated


def test_polynomial_features_degree_three():
    df = build_numeric_frame()
    result, info = create_polynomial_features(df, ["age"], degree=3)
    assert set(info["columns"]) == {"age_pow2", "age_pow3"}


def test_polynomial_features_with_missing_raises():
    df = build_numeric_frame()
    df.loc[0, "age"] = np.nan
    with pytest.raises(ValueError):
        create_polynomial_features(df, ["age", "score"], degree=2)


# Feature selection ----------------------------------------------------------


def test_select_by_variance_drops_constant():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [7, 7, 7, 7], "c": [2, 4, 6, 8]})
    result, info = select_by_variance(df, ["a", "b", "c"], threshold=0.0)
    assert info["dropped_columns"] == ["b"]
    assert "b" not in result.columns
    assert "a" in result.columns


def test_select_by_variance_with_missing_raises():
    df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
    with pytest.raises(ValueError):
        select_by_variance(df, ["a"], threshold=0.0)


def test_select_by_correlation_keeps_strong_features():
    df = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5],
            "strong": [10, 20, 30, 40, 50],
            "weak": [1, 0, 1, 0, 1],
        }
    )
    result, info = select_by_correlation(df, "target", threshold=0.8)
    assert info["kept_columns"] == ["strong"]
    assert info["dropped_columns"] == ["weak"]


def test_select_by_correlation_non_numeric_target_raises():
    df = pd.DataFrame({"target": ["a", "b", "c"], "x": [1, 2, 3]})
    with pytest.raises(ValueError):
        select_by_correlation(df, "target", threshold=0.2)


def test_select_by_correlation_missing_target_raises():
    with pytest.raises(ValueError):
        select_by_correlation(build_numeric_frame(), "nope", threshold=0.2)


# Feature importance ---------------------------------------------------------


def test_feature_importance_tree_model():
    df = build_numeric_frame()
    X = df[["age", "income"]]
    y = np.array([0, 1, 0, 1, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    importance = feature_importance(model, ["age", "income"])
    assert list(importance["Feature"]) == ["age", "income"] or list(
        importance["Feature"]
    ) == ["income", "age"]
    assert set(importance.columns) == {"Feature", "Importance"}


def test_feature_importance_linear_model():
    df = build_numeric_frame()
    X = df[["age", "income"]]
    y = np.array([0, 1, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    importance = feature_importance(model, ["age", "income"])
    assert len(importance) == 2
    assert (importance["Importance"] >= 0).all()


def test_feature_importance_unsupported_model_returns_none():
    class Dummy:
        pass

    assert feature_importance(Dummy(), ["a"]) is None


def test_feature_importance_mismatched_sizes_returns_none():
    df = build_numeric_frame()
    X = df[["age", "income"]]
    y = np.array([0, 1, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    assert feature_importance(model, ["only_one"]) is None


# Operation dispatcher -------------------------------------------------------


def test_apply_feature_op_dispatches():
    df = build_numeric_frame()
    op = {
        "key": "math",
        "label": "square",
        "code": "",
        "params": {"columns": ["age"], "method": "square"},
    }
    result = apply_feature_op(df, op)
    assert "age_squared" in result.columns


def test_apply_feature_op_unknown_key_raises():
    with pytest.raises(ValueError):
        apply_feature_op(build_numeric_frame(), {"key": "bogus", "params": {}})


def test_operation_effect_describes_added_columns():
    df = build_numeric_frame()
    after, _ = create_interaction(df, "age", "score")
    effect = operation_effect(df, after, {"key": "interaction"})
    assert "age_x_score" in effect


def test_operation_effect_describes_removed_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 1]})
    after, _ = select_by_variance(df, ["a", "b"], 0.0)
    effect = operation_effect(df, after, {"key": "variance"})
    assert "removed" in effect


# Code generation ------------------------------------------------------------


def test_code_generation_returns_strings():
    assert "df['age'] + df['score']" in numeric_feature_code("age", "score", "sum")
    assert "np.log" in math_transform_code(["age"], "log")
    assert "pd.qcut" in binning_code("age", 4, "quantile")
    assert "dt.year" in datetime_code("joined", ["year", "weekday"])
    assert "str.len" in text_code("name", ["length"])
    assert "*" in interaction_code("age", "score")
    assert "PolynomialFeatures" in polynomial_code(["age"], 2)
    assert "VarianceThreshold" in variance_code(["age"], 0.0)
    assert "corrwith" in correlation_code("target", 0.2)


def test_option_enum_contents():
    assert NUMERIC_OPERATIONS == ["sum", "difference", "product", "ratio"]
    assert MATH_TRANSFORMS == ["log", "sqrt", "square"]
    assert BINNING_METHODS == ["equal width", "quantile"]
    assert set(DATETIME_PARTS) == {"year", "month", "day", "weekday"}
    assert set(TEXT_PARTS) == {"length", "word count"}
