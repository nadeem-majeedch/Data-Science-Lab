"""Tests for the classification model registry."""

import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.models import (
    MODEL_KEYS,
    MODELS,
    build_classifier,
    classifier_constructor_code,
    default_classifier_params,
    get_model,
)

EXPECTED_CLASSES = {
    "Logistic Regression": LogisticRegression,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Naive Bayes": GaussianNB,
    "Support Vector Machine": SVC,
    "Gradient Boosting": GradientBoostingClassifier,
}

REQUIRED_FIELDS = (
    "model_class",
    "requires_scaling",
    "supports_importance",
    "why",
    "key_parameters",
    "advantages",
    "limitations",
    "when_to_use",
    "params",
)


def test_all_seven_models_are_registered():
    assert set(MODEL_KEYS) == set(EXPECTED_CLASSES)
    assert len(MODELS) == 7


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_every_model_has_full_metadata(model_key):
    spec = get_model(model_key)
    for field in REQUIRED_FIELDS:
        assert field in spec, f"{model_key} is missing field {field}"
    assert spec["model_class"] is EXPECTED_CLASSES[model_key]


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_every_model_builds_the_expected_estimator(model_key):
    estimator = build_classifier(model_key)
    assert isinstance(estimator, EXPECTED_CLASSES[model_key])


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_build_classifier_accepts_custom_values(model_key):
    defaults = default_classifier_params(model_key)
    estimator = build_classifier(model_key, defaults)
    assert isinstance(estimator, EXPECTED_CLASSES[model_key])


def test_none_allowed_param_maps_zero_to_none():
    estimator = build_classifier("Decision Tree", {"max_depth": 0, "criterion": "gini", "min_samples_split": 2})
    assert estimator.max_depth is None
    estimator = build_classifier("Decision Tree", {"max_depth": 7, "criterion": "gini", "min_samples_split": 2})
    assert estimator.max_depth == 7


def test_naive_bayes_has_no_parameter_widgets():
    assert default_classifier_params("Naive Bayes") == {}


def test_gradient_boosting_defaults():
    estimator = build_classifier("Gradient Boosting")
    assert estimator.n_estimators == 100
    assert estimator.learning_rate == 0.1


def test_svc_is_built_with_probability():
    estimator = build_classifier("Support Vector Machine")
    assert isinstance(estimator, SVC)
    assert estimator.probability is True


def test_constructor_code_contains_class_and_params():
    code = classifier_constructor_code("Random Forest", {"n_estimators": 50, "max_depth": 0})
    assert "RandomForestClassifier" in code
    assert "n_estimators=50" in code
    # none_allowed zero values are omitted so sklearn's native None default applies
    assert "max_depth" not in code


def test_constructor_code_with_custom_value():
    code = classifier_constructor_code("Logistic Regression", {"C": 0.5, "max_iter": 500})
    assert "C=0.5" in code
    assert "max_iter=500" in code
