"""Tests for the model comparison helpers."""

import numpy as np
import pandas as pd
import pytest

from utils.model_comparison import (
    CLASSIFICATION_COLUMNS,
    REGRESSION_COLUMNS,
    _with_random_state,
    best_model,
    compare_classifiers,
    compare_regressors,
    comparison_code,
)
from utils.models import build_classifier


def build_frame(n=120) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "group": rng.choice(["x", "y", "z"], n),
            "target": rng.choice(["A", "B", "C"], n),
        }
    )


def build_regression_frame(n=120) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    a = rng.uniform(0, 10, n)
    b = rng.uniform(0, 10, n)
    return pd.DataFrame(
        {
            "a": a,
            "b": b,
            "group": rng.choice(["x", "y"], n),
            "target": 3 * a + 2 * b + rng.normal(0, 1, n),
        }
    )


# Classification comparison -------------------------------------------------


def test_compare_classifiers_columns_and_row_count():
    df = build_frame()
    results = compare_classifiers(df[["a", "b", "group"]], df["target"], random_state=42)
    assert list(results["table"].columns) == CLASSIFICATION_COLUMNS
    assert len(results["table"]) == len(results["config"]["model_keys"]) == 7
    for metric in ("Accuracy", "Precision", "Recall", "F1"):
        assert results["table"][metric].between(0.0, 1.0).all()


def test_compare_classifiers_is_reproducible():
    df = build_frame()
    X, y = df[["a", "b", "group"]], df["target"]
    first = compare_classifiers(X, y, random_state=42)["table"]
    second = compare_classifiers(X, y, random_state=42)["table"]
    assert first.equals(second)


def test_compare_classifiers_auc_present_for_binary_and_multiclass():
    df = build_frame()
    results = compare_classifiers(df[["a", "b", "group"]], df["target"], random_state=42)
    assert results["table"]["AUC"].notna().all()


def test_compare_classifiers_drops_missing_targets():
    df = build_frame(60)
    df.loc[:9, "target"] = np.nan
    results = compare_classifiers(df[["a", "b", "group"]], df["target"], random_state=42)
    n_train = len(results["X_train"])
    n_test = len(results["X_test"])
    assert n_train + n_test == 50


def test_compare_classifiers_no_models_raises():
    df = build_frame()
    with pytest.raises(ValueError, match="at least one model"):
        compare_classifiers(df[["a", "b"]], df["target"], model_keys=[])


def test_compare_classifiers_share_the_same_split():
    df = build_frame()
    results = compare_classifiers(df[["a", "b", "group"]], df["target"], random_state=42)
    train_index = set(results["X_train"].index)
    test_index = set(results["X_test"].index)
    assert len(train_index & test_index) == 0
    assert len(train_index) + len(test_index) == len(df)


# Regression comparison -----------------------------------------------------


def test_compare_regressors_columns_and_row_count():
    df = build_regression_frame()
    results = compare_regressors(df[["a", "b", "group"]], df["target"], random_state=42)
    assert list(results["table"].columns) == REGRESSION_COLUMNS
    assert len(results["table"]) == 7
    assert results["table"]["MAE"].min() >= 0.0
    assert results["table"]["RMSE"].min() >= 0.0


def test_compare_regressors_is_reproducible():
    df = build_regression_frame()
    X, y = df[["a", "b", "group"]], df["target"]
    first = compare_regressors(X, y, random_state=42)["table"]
    second = compare_regressors(X, y, random_state=42)["table"]
    assert first.equals(second)


def test_compare_regressors_drops_missing_targets():
    df = build_regression_frame(60)
    df.loc[:4, "target"] = np.nan
    results = compare_regressors(df[["a", "b"]], df["target"], random_state=42)
    assert len(results["X_train"]) + len(results["X_test"]) == 55


# Ranking -------------------------------------------------------------------


def test_best_model_classification_ranks_by_accuracy():
    table = pd.DataFrame(
        {
            "Model": ["A", "B"],
            "Accuracy": [0.8, 0.9],
            "Precision": [0.7, 0.6],
            "Recall": [0.7, 0.8],
            "F1": [0.7, 0.7],
        }
    )
    assert best_model(table) == "B"


def test_best_model_regression_ranks_by_r2():
    table = pd.DataFrame({"Model": ["A", "B"], "MAE": [2.0, 1.0], "RMSE": [3.0, 1.5], "R2": [0.5, 0.8]})
    assert best_model(table) == "B"


# Random-state pinning ------------------------------------------------------


def test_with_random_state_pins_supported_estimators():
    estimator = build_classifier("Random Forest")
    _with_random_state(estimator, 42)
    assert estimator.get_params()["random_state"] == 42


def test_with_random_state_ignores_unsupported_estimators():
    estimator = build_classifier("K-Nearest Neighbors")
    before = dict(estimator.get_params(deep=False))
    _with_random_state(estimator, 42)
    assert estimator.get_params(deep=False) == before


# Code generation -----------------------------------------------------------


def test_comparison_code_classification_contains_shared_setup():
    code = comparison_code(
        "classification",
        {
            "target": "grade",
            "features": ["a", "b"],
            "model_keys": ["Logistic Regression", "Random Forest"],
            "test_size": 0.2,
            "random_state": 42,
        },
    )
    assert "SAME split" in code
    assert "results.append(row)" in code
    assert "roc_auc_score" in code
    assert "random_state=42" in code


def test_comparison_code_regression_contains_shared_setup():
    code = comparison_code(
        "regression",
        {
            "target": "score",
            "features": ["a", "b"],
            "model_keys": ["Linear Regression", "Ridge Regression"],
            "test_size": 0.2,
            "random_state": 42,
        },
    )
    assert "SAME split" in code
    assert "results.append(row)" in code
    assert "mean_absolute_error" in code
