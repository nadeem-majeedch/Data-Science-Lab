"""Tests for the classification training helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from utils.model_training import (
    MAX_CLASSES,
    build_default_preprocessor,
    categorical_feature_columns,
    classification_metrics,
    classification_report_frame,
    confusion_matrix_frame,
    predict_sample,
    train_classifier,
    training_code,
    transformed_feature_names,
    validate_classification_target,
)


def build_multiclass_frame(n=60) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "age": rng.randint(18, 65, n),
            "score": rng.normal(70, 15, n),
            "city": rng.choice(["X", "Y", "Z"], n),
            "grade": rng.choice(["A", "B", "C"], n, p=[0.3, 0.4, 0.3]),
        }
    )


def build_binary_frame(n=60) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    return pd.DataFrame(
        {
            "hours": rng.randint(0, 40, n),
            "gpa": rng.uniform(1.0, 4.0, n),
            "target": rng.choice([0, 1], n),
        }
    )


# Target validation ----------------------------------------------------------


def test_validate_target_categorical():
    df = build_multiclass_frame()
    info = validate_classification_target(df, "grade")
    assert info["n_classes"] == 3
    assert set(info["classes"]) == {"A", "B", "C"}
    assert info["dtype"] in ("object", "str")
    assert info["missing"] == 0


def test_validate_target_binary_numeric():
    df = build_binary_frame()
    info = validate_classification_target(df, "target")
    assert info["n_classes"] == 2
    assert set(info["classes"]) == {"0", "1"}


def test_validate_target_missing_column_raises():
    with pytest.raises(ValueError):
        validate_classification_target(build_multiclass_frame(), "nope")


def test_validate_target_single_class_raises():
    df = build_multiclass_frame()
    df["grade"] = "A"
    with pytest.raises(ValueError):
        validate_classification_target(df, "grade")


def test_validate_target_all_missing_raises():
    df = build_multiclass_frame()
    df["grade"] = np.nan
    with pytest.raises(ValueError):
        validate_classification_target(df, "grade")


def test_validate_target_high_cardinality_numeric_raises():
    df = pd.DataFrame({"continuous": list(range(1, MAX_CLASSES + 2)), "x": [1] * (MAX_CLASSES + 1)})
    with pytest.raises(ValueError):
        validate_classification_target(df, "continuous")


def test_validate_target_counts_ignore_missing():
    df = build_multiclass_frame()
    df.loc[:2, "grade"] = np.nan
    info = validate_classification_target(df, "grade")
    assert info["missing"] == 3
    assert info["counts"].sum() == len(df) - 3


# Feature helpers ------------------------------------------------------------


def test_categorical_feature_columns_excludes_numbers():
    df = build_multiclass_frame()
    assert categorical_feature_columns(df) == ["city", "grade"]
    assert categorical_feature_columns(build_binary_frame()) == []


def test_build_default_preprocessor_has_both_transformer_groups():
    df = build_multiclass_frame()
    pp = build_default_preprocessor(df[["age", "score", "city"]])
    names = [name for name, _, _ in pp.transformers]
    assert names == ["numeric", "categorical"]


def test_build_default_preprocessor_returns_none_for_empty_features():
    df = pd.DataFrame()
    assert build_default_preprocessor(df) is None


def test_transformed_feature_names_after_fit():
    df = build_multiclass_frame()
    X = df[["age", "city"]]
    pp = build_default_preprocessor(X)
    pp.fit(X)
    names = transformed_feature_names(pp)
    assert any(name.endswith("age") for name in names)
    assert any(name.startswith("categorical__city_") for name in names)


def test_transformed_feature_names_none_for_no_preprocessor():
    assert transformed_feature_names(None) is None


# Training -------------------------------------------------------------------


def test_train_classifier_binary():
    df = build_binary_frame()
    results = train_classifier(
        df[["hours", "gpa"]],
        df["target"],
        LogisticRegression(max_iter=500),
        random_state=42,
        stratify=True,
    )
    assert set(results["classes"]) == {"0", "1"}
    assert results["metrics"]["average"] == "binary"
    assert 0.0 <= results["metrics"]["accuracy"] <= 1.0
    assert set(results["metrics"]) >= {"accuracy", "precision", "recall", "f1"}
    assert len(results["y_pred"]) == len(results["y_test"])


def test_train_classifier_multiclass():
    df = build_multiclass_frame()
    results = train_classifier(
        df[["age", "score", "city"]],
        df["grade"],
        RandomForestClassifier(n_estimators=10, random_state=0),
        test_size=0.25,
        random_state=42,
        stratify=True,
    )
    assert len(results["classes"]) == 3
    assert results["metrics"]["average"] == "macro"
    assert results["report"].shape[0] == 3 + 2  # classes + macro + weighted


def test_train_classifier_split_is_reproducible():
    df = build_multiclass_frame()
    X = df[["age", "score", "city"]]
    y = df["grade"]
    a = train_classifier(X, y, DecisionTreeClassifier(max_depth=4), random_state=42)
    b = train_classifier(X, y, DecisionTreeClassifier(max_depth=4), random_state=42)
    assert a["X_train"].equals(b["X_train"])
    assert a["X_test"].equals(b["X_test"])


def test_train_classifier_split_does_not_overlap():
    df = build_multiclass_frame()
    results = train_classifier(
        df[["age", "score", "city"]],
        df["grade"],
        DecisionTreeClassifier(max_depth=4),
        random_state=1,
    )
    overlap = results["X_train"].index.intersection(results["X_test"].index)
    assert len(overlap) == 0


def test_train_classifier_drops_rows_with_missing_target():
    df = build_multiclass_frame()
    df.loc[[0, 1, 2], "grade"] = np.nan
    results = train_classifier(
        df[["age", "score", "city"]],
        df["grade"],
        DecisionTreeClassifier(max_depth=4),
        random_state=42,
    )
    assert len(results["X_test"]) + len(results["X_train"]) == len(df) - 3


def test_train_classifier_no_features_raises():
    df = build_multiclass_frame()
    with pytest.raises(ValueError):
        train_classifier(df[[]], df["grade"], DecisionTreeClassifier())


def test_train_classifier_stratify_failure_raises_helpful_message():
    df = build_multiclass_frame()
    df = df[df["grade"] != "C"]  # two classes
    with pytest.raises(ValueError):
        train_classifier(
            df[["age", "score", "city"]],
            df["grade"],
            DecisionTreeClassifier(),
            test_size=0.5,
            stratify=True,
        )


def test_train_classifier_preprocessor_fitted_on_train_only():
    df = build_multiclass_frame()
    X = df[["age", "score", "city"]]
    y = df["grade"]
    pp = build_default_preprocessor(X)
    results = train_classifier(
        X, y, DecisionTreeClassifier(max_depth=4), preprocessor=pp, random_state=3
    )
    assert results["preprocessor"] is pp
    # The fitted preprocessor must be able to transform unseen rows, including
    # a category it never saw in training (handle_unknown="ignore").
    unseen = pd.DataFrame(
        {"age": [20], "score": [90], "city": ["UNKNOWN_CITY"]}
    )
    out = results["preprocessor"].transform(unseen)
    assert out.shape[0] == 1
    # Feature names come from the fitted preprocessor and label the model input.
    assert results["feature_names"] is not None
    assert len(results["feature_names"]) == out.shape[1]


# Metrics and reports --------------------------------------------------------


def test_classification_metrics_binary_average():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["average"] == "binary"
    assert metrics["accuracy"] == pytest.approx(3 / 5)


def test_classification_metrics_multiclass_macro_average():
    y_true = np.array(["a", "a", "b", "b", "c", "c"])
    y_pred = np.array(["a", "b", "b", "b", "c", "c"])
    metrics = classification_metrics(y_true, y_pred, n_classes=3)
    assert metrics["average"] == "macro"
    assert metrics["accuracy"] == pytest.approx(5 / 6)


def test_report_frame_contains_class_and_average_rows():
    y_true = np.array(["a", "a", "b", "b", "c", "c"])
    y_pred = np.array(["a", "a", "b", "b", "c", "c"])
    frame = classification_report_frame(y_true, y_pred, ["a", "b", "c"])
    assert set(frame["Class"]) == {"a", "b", "c", "Macro average", "Weighted average"}
    assert list(frame.columns) == ["Class", "Precision", "Recall", "F1-score", "Support"]


def test_confusion_matrix_frame_layout():
    y_true = np.array(["a", "a", "b", "b", "c", "c"])
    y_pred = np.array(["a", "a", "b", "b", "c", "c"])
    frame = confusion_matrix_frame(y_true, y_pred, ["a", "b", "c"])
    assert frame.shape == (3, 3)
    assert frame.loc["Actual a", "Predicted a"] == 2
    assert frame.to_numpy().sum() == 6


# Prediction -----------------------------------------------------------------


def test_predict_sample_returns_prediction_and_probabilities():
    df = build_multiclass_frame()
    results = train_classifier(
        df[["age", "score", "city"]],
        df["grade"],
        RandomForestClassifier(n_estimators=10, random_state=0),
        random_state=42,
    )
    row = {"age": 30, "score": 75, "city": "X"}
    outcome = predict_sample(results["pipeline"], row)
    assert outcome["prediction"] in {"A", "B", "C"}
    assert outcome["probabilities"] is not None
    assert len(outcome["probabilities"]) == 3
    assert abs(outcome["probabilities"].sum() - 1.0) < 1e-6


def test_predict_sample_accepts_series():
    df = build_multiclass_frame()
    results = train_classifier(
        df[["age", "score", "city"]],
        df["grade"],
        DecisionTreeClassifier(max_depth=4),
        random_state=42,
    )
    sample = df.iloc[0][["age", "score", "city"]]
    outcome = predict_sample(results["pipeline"], sample)
    assert outcome["prediction"] in {"A", "B", "C"}


# Code generation ------------------------------------------------------------


def test_training_code_contains_core_steps():
    code = training_code(
        "Random Forest", "RandomForestClassifier(n_estimators=50)", "grade",
        ["age", "score", "city"], 0.2, 42, True,
    )
    assert "train_test_split" in code
    assert "ColumnTransformer" in code
    assert "RandomForestClassifier(n_estimators=50)" in code
    assert "stratify=y" in code
    assert "classification_report" in code


def test_training_code_without_stratify():
    code = training_code(
        "Logistic Regression", "LogisticRegression(max_iter=500)", "target",
        ["hours", "gpa"], 0.25, 7, False,
    )
    assert "test_size=0.25" in code
    assert "random_state=7" in code
    assert "stratify" not in code
