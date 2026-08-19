"""Tests for the data preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import (
    ENCODING_METHODS,
    MISSING_STRATEGIES,
    SCALING_METHODS,
    build_preprocessor,
    compare_before_after,
    duplicates_code,
    encode_categorical,
    encode_code,
    handle_missing,
    missing_values_code,
    outlier_counts,
    outlier_mask,
    outliers_code,
    preprocessor_code,
    remove_duplicates,
    remove_outliers,
    scale_code,
    scale_numeric,
    split_code,
    split_train_test,
)


def build_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 25, 30, None, 40, 28, 33],
            "score": [10.0, 20.0, 10.0, 20.0, 50.0, 10.0, 12.0, 18.0],
            "name": ["Ana", "Bob", "Ana", "Cid", "Eve", "Fay", "Gus", "Haz"],
            "city": ["X", "Y", "X", "Y", None, "Z", "Z", "X"],
            "status": ["active"] * 8,
        }
    )


def build_outlier_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "other": [10, 20, 30, 40, 50, 60],
        }
    )


# Missing values ------------------------------------------------------------


def test_missing_drop_rows():
    df = build_sample_frame()
    result, info = handle_missing(df, ["age"], "drop rows")
    assert info["removed_rows"] == 1
    assert len(result) == len(df) - 1
    assert result["age"].isna().sum() == 0


def test_missing_drop_columns():
    df = build_sample_frame()
    result, info = handle_missing(df, ["age", "city"], "drop columns")
    assert info["removed_columns"] == 2
    assert "age" not in result.columns
    assert "city" not in result.columns


def test_missing_median_imputes_numeric():
    df = build_sample_frame()
    result, info = handle_missing(df, ["age"], "median")
    assert result["age"].isna().sum() == 0
    assert result["age"].iloc[4] == pytest.approx(30.0)


def test_missing_mean_imputes_numeric():
    df = build_sample_frame()
    result, info = handle_missing(df, ["age"], "mean")
    assert result["age"].iloc[4] == pytest.approx(df["age"].mean())


def test_missing_mode_imputes_categorical():
    df = build_sample_frame()
    result, info = handle_missing(df, ["city"], "mode")
    assert result["city"].iloc[4] == "X"
    assert result["city"].isna().sum() == 0


def test_missing_constant_fill():
    df = build_sample_frame()
    result, info = handle_missing(df, ["age"], "constant", fill_value=0)
    assert result["age"].isna().sum() == 0
    assert result["age"].iloc[4] == 0


def test_missing_skips_nonnumeric_for_mean_median():
    df = build_sample_frame()
    result, info = handle_missing(df, ["city"], "median")
    assert "warnings" in info
    assert result["city"].isna().sum() == 1  # untouched


def test_missing_all_nan_column_warns():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]})
    result, info = handle_missing(df, ["a"], "mean")
    assert "warnings" in info
    assert result["a"].isna().sum() == 2


def test_missing_ignores_unknown_columns():
    df = build_sample_frame()
    result, info = handle_missing(df, ["does_not_exist"], "median")
    assert result.equals(df)


# Duplicates ----------------------------------------------------------------


def test_remove_duplicates():
    df = build_sample_frame()
    result, info = remove_duplicates(df)
    assert info["removed_rows"] == 1  # rows 0 and 2 are identical
    assert result.duplicated().sum() == 0


# Outliers ------------------------------------------------------------------


def test_outlier_mask_marks_extreme_values():
    df = build_outlier_frame()
    mask = outlier_mask(df, ["value"])
    assert mask["value"].tolist() == [False] * 5 + [True]


def test_outlier_counts_per_column():
    df = build_outlier_frame()
    counts = outlier_counts(df, ["value"])
    assert counts[counts["Column"] == "value"]["Outliers"].iloc[0] == 1


def test_remove_outliers_drops_flagged_rows():
    df = build_outlier_frame()
    result, info = remove_outliers(df, ["value"])
    assert info["removed_rows"] == 1
    assert len(result) == len(df) - 1


def test_outlier_threshold_tighter_catches_more():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 15]})
    assert outlier_mask(df, ["value"], threshold=1.0)["value"].sum() == 1
    assert outlier_mask(df, ["value"], threshold=3.0)["value"].sum() == 0


# Encoding ------------------------------------------------------------------


def test_one_hot_encoding_adds_columns():
    df = build_sample_frame()
    result, info = encode_categorical(df, ["city"], "one-hot")
    assert {"city_X", "city_Y", "city_Z"}.issubset(result.columns)
    assert "city" not in result.columns
    assert (result["city_X"] + result["city_Y"] + result["city_Z"]).max() == 1


def test_label_encoding_replaces_with_codes():
    df = build_sample_frame()
    result, info = encode_categorical(df, ["city"], "label")
    assert "city" in result.columns
    assert result["city"].dtype.kind == "i"


# Scaling -------------------------------------------------------------------


def test_standard_scaler_centers_columns():
    df = build_sample_frame().dropna()
    result, info = scale_numeric(df, ["age", "score"], "StandardScaler")
    assert result["age"].mean() == pytest.approx(0.0, abs=1e-9)
    # StandardScaler uses the population standard deviation (ddof=0).
    assert result["age"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_minmax_scaler_bounds():
    df = build_sample_frame().dropna()
    result, info = scale_numeric(df, ["age"], "MinMaxScaler")
    assert result["age"].min() == pytest.approx(0.0)
    assert result["age"].max() == pytest.approx(1.0)


def test_robust_scaler_returns_scaler_in_info():
    df = build_sample_frame().dropna()
    result, info = scale_numeric(df, ["age"], "RobustScaler")
    assert info["scaler"] is not None
    assert result["age"].notna().all()


def test_scaling_with_missing_values_raises():
    df = build_sample_frame()
    with pytest.raises(ValueError):
        scale_numeric(df, ["age"], "StandardScaler")


# Train/test split ----------------------------------------------------------


def test_split_train_test_shapes():
    df = build_sample_frame().dropna()  # 7 rows after dropping the NaN row
    split = split_train_test(df, "status", test_size=0.25, random_state=42)
    assert len(split["X_train"]) == len(split["y_train"]) == 5
    assert len(split["X_test"]) == len(split["y_test"]) == 2
    assert split["X_train"].shape[1] == 4  # all columns except target


def test_split_reproducible_with_seed():
    df = build_sample_frame().dropna()
    a = split_train_test(df, "status", random_state=7)
    b = split_train_test(df, "status", random_state=7)
    assert a["X_train"].equals(b["X_train"])
    assert a["X_test"].equals(b["X_test"])


def test_split_missing_target_raises():
    df = build_sample_frame()
    with pytest.raises(ValueError):
        split_train_test(df, "nope")


# Reusable preprocessor -----------------------------------------------------


def test_build_preprocessor_none_without_config():
    assert build_preprocessor() is None


def test_build_preprocessor_contains_configured_steps():
    pp = build_preprocessor(
        impute_columns=["age"],
        impute_strategy="median",
        encode_columns=["city"],
        encode_method="one-hot",
        scale_columns=["score"],
        scale_method="MinMaxScaler",
    )
    names = [name for name, _, _ in pp.transformers]
    assert names == ["imputer", "encoder", "scaler"]


def test_preprocessor_fit_on_train_transform_test():
    df = build_sample_frame()
    pp = build_preprocessor(
        impute_columns=["age"],
        encode_columns=["city"],
        scale_columns=["score"],
    )
    X = df.drop(columns=["status"])
    pp.fit(X)
    transformed = pp.transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


def test_preprocessor_handles_unknown_category():
    df = build_sample_frame()
    pp = build_preprocessor(encode_columns=["city"], encode_method="one-hot")
    pp.fit(df.drop(columns=["status"]))
    # New data must include every passthrough column; the encoder is the only
    # learned transformer and must tolerate a category it has never seen.
    new = pd.DataFrame({"city": ["NEW"], "age": [1], "score": [1], "name": ["N"]})
    out = pp.transform(new)
    assert out.shape[0] == 1


# Comparison ----------------------------------------------------------------


def test_compare_before_after_rows():
    df = build_sample_frame()
    cleaned, _ = handle_missing(df, ["age", "city"], "median")
    report = compare_before_after(df, cleaned)
    assert set(report["Metric"]) == {
        "Rows",
        "Columns",
        "Missing cells",
        "Duplicate rows",
        "Numeric columns",
        "Non-numeric columns",
    }
    # age (numeric) is imputed; city (text) is skipped by median, so one NaN remains.
    assert report[report["Metric"] == "Missing cells"]["After"].iloc[0] == 1


# Code generation -----------------------------------------------------------


def test_code_generation_returns_strings():
    assert isinstance(missing_values_code(["age"], "median"), str)
    assert "drop_duplicates" in duplicates_code()
    assert "Q1" in outliers_code(["age"], 1.5)
    assert "get_dummies" in encode_code(["city"], "one-hot")
    assert "StandardScaler" in scale_code(["age"], "StandardScaler")
    assert "train_test_split" in split_code("status", 0.2, 42, True)
    assert "ColumnTransformer" in preprocessor_code(
        impute_columns=["age"], encode_columns=["city"], scale_columns=["score"]
    )


def test_constant_enum_options():
    assert "drop rows" in MISSING_STRATEGIES
    assert set(ENCODING_METHODS) == {"one-hot", "label"}
    assert set(SCALING_METHODS) == {"StandardScaler", "MinMaxScaler", "RobustScaler"}
