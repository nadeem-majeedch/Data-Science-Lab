"""Tests for the regression training helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from utils.regression_training import (
    predict_sample,
    prediction_frame,
    regression_metrics,
    train_regressor,
    training_code,
    validate_regression_target,
)


def build_frame(n=80) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    hours = rng.uniform(0, 40, n)
    return pd.DataFrame(
        {
            "hours": hours,
            "difficulty": rng.uniform(1, 5, n),
            "subject": rng.choice(["Math", "Science", "Arts"], n),
            "score": 40 + hours * 0.8 + rng.normal(0, 5, n),
        }
    )


def build_linear_frame(n=80) -> pd.DataFrame:
    rng = np.random.RandomState(1)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    y = 3 * x1 + 2 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


# Target validation ----------------------------------------------------------


def test_validate_target_numeric():
    df = build_frame()
    info = validate_regression_target(df, "score")
    assert info["n_values"] == len(df)
    assert info["dtype"] in ("float64", "float32")
    assert info["n_unique"] >= 2
    assert info["min"] <= info["mean"] <= info["max"]


def test_validate_target_non_numeric_raises():
    df = build_frame()
    with pytest.raises(ValueError):
        validate_regression_target(df, "subject")


def test_validate_target_missing_column_raises():
    with pytest.raises(ValueError):
        validate_regression_target(build_frame(), "nope")


def test_validate_target_constant_raises():
    df = build_frame()
    df["score"] = 50.0
    with pytest.raises(ValueError):
        validate_regression_target(df, "score")


def test_validate_target_all_missing_raises():
    df = build_frame()
    df["score"] = np.nan
    with pytest.raises(ValueError):
        validate_regression_target(df, "score")


def test_validate_target_counts_missing_values():
    df = build_frame()
    df.loc[:4, "score"] = np.nan
    info = validate_regression_target(df, "score")
    assert info["missing"] == 5
    assert info["n_values"] == len(df) - 5


# Metrics and prediction table -----------------------------------------------


def test_regression_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 4.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(0.2)
    assert metrics["mse"] == pytest.approx(0.2)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.2))
    assert metrics["r2"] == pytest.approx(0.9)


def test_prediction_frame_columns_and_residual():
    y_true = np.array([3.0, 5.0, 7.0])
    y_pred = np.array([3.5, 4.0, 7.0])
    frame = prediction_frame(y_true, y_pred)
    assert list(frame.columns) == ["Actual", "Predicted", "Residual"]
    assert frame["Residual"].tolist() == pytest.approx([-0.5, 1.0, 0.0])


# Training -------------------------------------------------------------------


def test_train_regressor_linear_recovers_relationship():
    df = build_linear_frame()
    results = train_regressor(
        df[["x1", "x2"]],
        df["y"],
        LinearRegression(),
        random_state=42,
    )
    assert results["metrics"]["r2"] > 0.9
    assert set(results["metrics"]) == {"mae", "mse", "rmse", "r2"}
    assert list(results["predictions"].columns) == ["Actual", "Predicted", "Residual"]
    assert len(results["y_pred"]) == len(results["y_test"])
    assert results["feature_names"] is not None


def test_train_regressor_multicollinear_with_preprocessor():
    df = build_frame()
    results = train_regressor(
        df[["hours", "difficulty", "subject"]],
        df["score"],
        RandomForestRegressor(n_estimators=10, random_state=0),
        test_size=0.25,
        random_state=7,
    )
    assert 0.0 <= results["metrics"]["r2"] <= 1.0
    assert results["metrics"]["r2"] > 0.1
    assert results["train_score"] >= results["test_score"] - 0.6


def test_train_regressor_split_is_reproducible():
    df = build_frame()
    X = df[["hours", "difficulty", "subject"]]
    y = df["score"]
    a = train_regressor(X, y, LinearRegression(), random_state=42)
    b = train_regressor(X, y, LinearRegression(), random_state=42)
    assert a["X_train"].equals(b["X_train"])
    assert a["X_test"].equals(b["X_test"])


def test_train_regressor_split_does_not_overlap():
    df = build_frame()
    results = train_regressor(
        df[["hours", "difficulty", "subject"]],
        df["score"],
        LinearRegression(),
        random_state=1,
    )
    overlap = results["X_train"].index.intersection(results["X_test"].index)
    assert len(overlap) == 0


def test_train_regressor_drops_rows_with_missing_target():
    df = build_frame()
    df.loc[[0, 1, 2], "score"] = np.nan
    results = train_regressor(
        df[["hours", "difficulty", "subject"]],
        df["score"],
        LinearRegression(),
        random_state=42,
    )
    assert len(results["X_test"]) + len(results["X_train"]) == len(df) - 3


def test_train_regressor_no_features_raises():
    df = build_frame()
    with pytest.raises(ValueError):
        train_regressor(df[[]], df["score"], LinearRegression())


def test_train_regressor_preprocessor_fitted_on_train_only():
    df = build_frame()
    X = df[["hours", "difficulty", "subject"]]
    y = df["score"]
    from utils.model_training import build_default_preprocessor

    pp = build_default_preprocessor(X)
    results = train_regressor(
        X, y, LinearRegression(), preprocessor=pp, random_state=3
    )
    assert results["preprocessor"] is pp
    unseen = pd.DataFrame({"hours": [10], "difficulty": [4], "subject": ["UNSEEN"]})
    out = results["preprocessor"].transform(unseen)
    assert out.shape[0] == 1
    assert len(results["feature_names"]) == out.shape[1]


# Prediction -----------------------------------------------------------------


def test_predict_sample_returns_float():
    df = build_frame()
    results = train_regressor(
        df[["hours", "difficulty", "subject"]],
        df["score"],
        LinearRegression(),
        random_state=42,
    )
    row = {"hours": 20, "difficulty": 3, "subject": "Math"}
    value = predict_sample(results["pipeline"], row)
    assert isinstance(value, float)
    assert df["score"].min() - 50 <= value <= df["score"].max() + 50


# Code generation ------------------------------------------------------------


def test_training_code_contains_core_steps():
    code = training_code(
        "Random Forest Regressor",
        "RandomForestRegressor(n_estimators=50)",
        "score",
        ["hours", "difficulty", "subject"],
        0.2,
        42,
    )
    assert "train_test_split" in code
    assert "ColumnTransformer" in code
    assert "RandomForestRegressor(n_estimators=50)" in code
    assert "r2_score" in code
    assert "test_size=0.2" in code
    assert "random_state=42" in code


def test_training_code_excludes_stratify():
    code = training_code("Linear Regression", "LinearRegression()", "y", ["x1"], 0.25, 7)
    assert "stratify" not in code
