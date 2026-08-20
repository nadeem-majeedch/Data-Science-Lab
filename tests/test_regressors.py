"""Tests for the regression model registry."""

import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from utils.regressors import (
    REGRESSOR_KEYS,
    REGRESSORS,
    build_regressor,
    default_regressor_params,
    get_regressor,
    regressor_constructor_code,
)

EXPECTED_CLASSES = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "Decision Tree Regressor": DecisionTreeRegressor,
    "Random Forest Regressor": RandomForestRegressor,
    "Gradient Boosting Regressor": GradientBoostingRegressor,
    "KNN Regressor": KNeighborsRegressor,
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


def test_all_seven_regressors_are_registered():
    assert set(REGRESSOR_KEYS) == set(EXPECTED_CLASSES)
    assert len(REGRESSORS) == 7


@pytest.mark.parametrize("model_key", REGRESSOR_KEYS)
def test_every_regressor_has_full_metadata(model_key):
    spec = get_regressor(model_key)
    for field in REQUIRED_FIELDS:
        assert field in spec, f"{model_key} is missing field {field}"
    assert spec["model_class"] is EXPECTED_CLASSES[model_key]


@pytest.mark.parametrize("model_key", REGRESSOR_KEYS)
def test_every_regressor_builds_the_expected_estimator(model_key):
    assert isinstance(build_regressor(model_key), EXPECTED_CLASSES[model_key])


@pytest.mark.parametrize("model_key", REGRESSOR_KEYS)
def test_build_regressor_accepts_custom_values(model_key):
    estimator = build_regressor(model_key, default_regressor_params(model_key))
    assert isinstance(estimator, EXPECTED_CLASSES[model_key])


def test_none_allowed_param_maps_zero_to_none():
    estimator = build_regressor("Random Forest Regressor", {"n_estimators": 10, "max_depth": 0})
    assert estimator.max_depth is None
    estimator = build_regressor("Random Forest Regressor", {"n_estimators": 10, "max_depth": 8})
    assert estimator.max_depth == 8


def test_linear_regression_defaults():
    estimator = build_regressor("Linear Regression")
    assert isinstance(estimator, LinearRegression)
    assert estimator.fit_intercept is True


def test_constructor_code_contains_class_and_params():
    code = regressor_constructor_code("Ridge Regression", {"alpha": 0.5})
    assert "Ridge" in code
    assert "alpha=0.5" in code


def test_constructor_code_omits_none_allowed_zero():
    code = regressor_constructor_code("Decision Tree Regressor", {"max_depth": 0, "min_samples_split": 2})
    assert "DecisionTreeRegressor" in code
    assert "max_depth" not in code
