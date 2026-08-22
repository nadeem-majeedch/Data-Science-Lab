"""Tests for the model evaluation helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from utils.evaluation import (
    HIGHEST_NOT_BEST,
    METRIC_GUIDANCE,
    cross_validate,
    evaluation_code,
    residual_statistics,
    roc_auc_brief,
    roc_curves,
)
from utils.model_training import train_classifier


def build_frame(n=120) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "group": rng.choice(["x", "y", "z"], n),
        }
    )


def build_binary(n=120) -> pd.DataFrame:
    df = build_frame(n)
    df["target"] = np.where(df["a"] + df["b"] > 0, "yes", "no")
    return df


def build_multiclass(n=120) -> pd.DataFrame:
    df = build_frame(n)
    rng = np.random.RandomState(1)
    df["target"] = rng.choice(["A", "B", "C"], n)
    return df


# AUC ------------------------------------------------------------------------


def test_roc_auc_brief_binary():
    df = build_binary()
    results = train_classifier(
        df[["a", "b", "group"]],
        df["target"],
        LogisticRegression(max_iter=1000),
        random_state=42,
        stratify=True,
    )
    value = roc_auc_brief(results["y_test"], results["y_proba"])
    assert value is not None
    assert 0.0 <= value <= 1.0


def test_roc_auc_brief_multiclass():
    df = build_multiclass()
    results = train_classifier(
        df[["a", "b", "group"]],
        df["target"],
        RandomForestClassifier(n_estimators=20, random_state=0),
        random_state=1,
        stratify=True,
    )
    value = roc_auc_brief(results["y_test"], results["y_proba"])
    assert value is not None
    assert 0.0 <= value <= 1.0


def test_roc_auc_brief_none_when_no_proba():
    assert roc_auc_brief(np.array([0, 1, 0, 1]), None) is None


def test_roc_auc_brief_rejects_single_class():
    y = np.array(["a", "a", "a"])
    proba = np.array([[0.4, 0.6], [0.3, 0.7], [0.5, 0.5]])
    assert roc_auc_brief(y, proba) is None


# ROC curves ----------------------------------------------------------------


def test_roc_curves_binary_returns_two_curves():
    df = build_binary()
    results = train_classifier(
        df[["a", "b", "group"]],
        df["target"],
        LogisticRegression(max_iter=1000),
        random_state=42,
        stratify=True,
    )
    curves = roc_curves(results["y_test"], results["y_proba"], results["classes"])
    assert len(curves) == 2
    for curve in curves:
        assert set(curve) == {"class", "fpr", "tpr", "auc"}
        assert len(curve["fpr"]) == len(curve["tpr"])
        assert 0.0 <= curve["auc"] <= 1.0


def test_roc_curves_multiclass_returns_one_per_class():
    df = build_multiclass()
    results = train_classifier(
        df[["a", "b", "group"]],
        df["target"],
        RandomForestClassifier(n_estimators=20, random_state=0),
        random_state=1,
        stratify=True,
    )
    curves = roc_curves(results["y_test"], results["y_proba"], results["classes"])
    assert len(curves) == len(results["classes"]) == 3


def test_roc_curves_empty_without_proba():
    assert roc_curves(np.array([0, 1]), None, ["0", "1"]) == []


# Cross-validation ----------------------------------------------------------


def test_cross_validate_classification_scores_in_range():
    df = build_binary()
    summary = cross_validate(
        lambda: LogisticRegression(max_iter=1000),
        df[["a", "b", "group"]],
        df["target"],
        task="classification",
        n_folds=5,
        random_state=42,
    )
    assert len(summary["scores"]) == 5
    assert all(0.0 <= score <= 1.0 for score in summary["scores"])
    assert 0.0 <= summary["mean"] <= 1.0
    assert summary["std"] >= 0.0
    assert summary["min"] <= summary["mean"] <= summary["max"]


def test_cross_validate_regression():
    df = build_frame()
    y = df["a"] * 2 + df["b"]
    summary = cross_validate(
        lambda: RandomForestRegressor(n_estimators=20, random_state=0),
        df[["a", "b"]],
        y,
        task="regression",
        n_folds=4,
        random_state=7,
    )
    assert len(summary["scores"]) == 4


def test_cross_validate_raises_on_rare_class():
    df = build_frame(40)
    df["target"] = np.random.RandomState(0).choice(["a", "b"], 40, p=[0.95, 0.05])
    with pytest.raises(ValueError, match="Cross-validation failed"):
        cross_validate(
            lambda: LogisticRegression(max_iter=1000),
            df[["a", "b"]],
            df["target"],
            task="classification",
            n_folds=5,
        )


# Residual diagnostics ------------------------------------------------------


def test_residual_statistics():
    frame = pd.DataFrame(
        {"Actual": [3.0, 5.0, 7.0], "Predicted": [3.5, 4.0, 7.0], "Residual": [-0.5, 1.0, 0.0]}
    )
    stats = residual_statistics(frame)
    assert stats["mean"] == pytest.approx(1 / 6)
    assert stats["median"] == pytest.approx(0.0)
    assert stats["min"] == pytest.approx(-0.5)
    assert stats["max"] == pytest.approx(1.0)
    assert stats["std"] >= 0.0


# Code generation -----------------------------------------------------------


def test_evaluation_code_classification_contains_core_steps():
    code = evaluation_code(
        "classification",
        {"model_key": "Random Forest", "target": "grade", "features": ["a", "b"], "random_state": 42},
    )
    assert "roc_curve" in code
    assert "cross_val_score" in code
    assert "StratifiedKFold" in code
    assert "confusion_matrix" in code


def test_evaluation_code_regression_contains_core_steps():
    code = evaluation_code(
        "regression",
        {"model_key": "Random Forest Regressor", "target": "score", "features": ["a", "b"], "random_state": 42},
    )
    assert "mean_absolute_error" in code
    assert "residuals" in code
    assert "cross_val_score" in code
    assert "KFold" in code


# Educational content -------------------------------------------------------


def test_metric_guidance_covers_all_tracked_metrics():
    for name in ("accuracy", "precision", "recall", "f1", "auc", "mae", "mse", "rmse", "r2"):
        assert name in METRIC_GUIDANCE
        assert METRIC_GUIDANCE[name]


def test_highest_not_best_message_exists():
    assert "not automatically the best" in HIGHEST_NOT_BEST
