"""Model registry and educational content for the Regression module.

Every regressor is described by the same metadata shape so the page can
render parameter controls, explainer text, and capability flags from a single
source of truth - mirroring the ``utils.models`` registry used by the
Classification module.
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

REGRESSORS: dict[str, dict] = {
    "Linear Regression": {
        "model_class": LinearRegression,
        "requires_scaling": True,
        "supports_importance": True,
        "why": (
            "Linear regression fits the line (or hyperplane) that minimizes "
            "the sum of squared errors between the observed and predicted "
            "values. It is the foundation of almost all regression modeling "
            "and estimates a *coefficient* for every feature."
        ),
        "key_parameters": (
            "- **fit_intercept**: whether the model estimates an intercept "
            "term. Leaving it on is almost always the right choice."
        ),
        "advantages": (
            "- Simple, fast, and widely understood.\n"
            "- Coefficients give the size and direction of each feature's "
            "effect, so it is very interpretable.\n"
            "- Closed-form solution needs no iterative tuning."
        ),
        "limitations": (
            "- Assumes a (roughly) linear relationship.\n"
            "- Sensitive to outliers and to strongly correlated features "
            "(multicollinearity).\n"
            "- Can overfit when there are many features relative to rows."
        ),
        "when_to_use": (
            "The first model to try on a regression problem, and the right "
            "choice when you need interpretable effect sizes - for example "
            "estimating how much each factor changes a house price."
        ),
        "params": [
            {
                "name": "fit_intercept",
                "label": "Fit intercept",
                "type": "choice",
                "options": [("Yes", True), ("No", False)],
                "value": True,
                "help": "Whether to estimate an intercept term for the model.",
            },
        ],
    },
    "Ridge Regression": {
        "model_class": Ridge,
        "requires_scaling": True,
        "supports_importance": True,
        "why": (
            "Ridge regression is linear regression with an **L2 penalty** on "
            "the size of the coefficients. The penalty shrinks large "
            "coefficients toward zero, which reduces variance and handles "
            "correlated features far better than plain linear regression."
        ),
        "key_parameters": (
            "- **alpha**: the strength of the penalty. Larger alpha shrinks "
            "coefficients more and produces a simpler model; smaller alpha "
            "moves toward plain linear regression."
        ),
        "advantages": (
            "- More stable than linear regression on correlated features.\n"
            "- Reduces overfitting while keeping every feature in the model.\n"
            "- Coefficients remain interpretable."
        ),
        "limitations": (
            "- Shrinks coefficients but never sets them to zero, so it does "
            "not perform feature selection.\n"
            "- Requires features to be scaled for the penalty to be fair."
        ),
        "when_to_use": (
            "When you have many, possibly correlated features and want a "
            "stable, interpretable model - a strong default before trying "
            "non-linear methods."
        ),
        "params": [
            {
                "name": "alpha",
                "label": "Regularization strength (alpha)",
                "type": "float",
                "min": 0.01,
                "max": 10.0,
                "value": 1.0,
                "step": 0.1,
                "help": "Strength of the L2 penalty. Larger values shrink "
                "coefficients more.",
            },
        ],
    },
    "Lasso Regression": {
        "model_class": Lasso,
        "requires_scaling": True,
        "supports_importance": True,
        "why": (
            "Lasso is linear regression with an **L1 penalty** that can drive "
            "coefficients *exactly to zero*. This makes it perform automatic "
            "feature selection while fitting the model."
        ),
        "key_parameters": (
            "- **alpha**: the strength of the penalty. Larger alpha drops "
            "more features.\n"
            "- **max_iter**: iterations allowed for the solver to converge."
        ),
        "advantages": (
            "- Performs feature selection: unimportant features get zero "
            "coefficients.\n"
            "- Produces sparse, interpretable models.\n"
            "- Reduces overfitting."
        ),
        "limitations": (
            "- Unstable when features are strongly correlated - it may pick "
            "one arbitrarily.\n"
            "- Requires scaled features for a fair penalty.\n"
            "- Solver can fail to converge without enough iterations."
        ),
        "when_to_use": (
            "When you suspect many features are irrelevant and want a sparse "
            "model that only keeps the important ones."
        ),
        "params": [
            {
                "name": "alpha",
                "label": "Regularization strength (alpha)",
                "type": "float",
                "min": 0.01,
                "max": 10.0,
                "value": 1.0,
                "step": 0.1,
                "help": "Strength of the L1 penalty. Larger values drop more "
                "features.",
            },
            {
                "name": "max_iter",
                "label": "Maximum iterations",
                "type": "int",
                "min": 100,
                "max": 5000,
                "value": 1000,
                "step": 100,
                "help": "Maximum number of solver iterations. Raise it if the "
                "model warns that it did not converge.",
            },
        ],
    },
    "Decision Tree Regressor": {
        "model_class": DecisionTreeRegressor,
        "requires_scaling": False,
        "supports_importance": True,
        "why": (
            "A decision tree regressor repeatedly splits the feature space "
            "into regions and predicts the **mean of the training targets** "
            "inside each region. The result is a step function you can read "
            "as if-else rules."
        ),
        "key_parameters": (
            "- **max_depth**: how deep the tree may grow. Deeper trees fit "
            "the training data better but overfit more easily.\n"
            "- **min_samples_split**: minimum samples required to split a node."
        ),
        "advantages": (
            "- No feature scaling needed.\n"
            "- Captures non-linear relationships and interactions naturally.\n"
            "- Interpretable rule structure."
        ),
        "limitations": (
            "- Cannot extrapolate beyond the range seen in training - "
            "predictions are always a mean of observed targets.\n"
            "- Very prone to overfitting without depth limits.\n"
            "- Unstable: small data changes can change the whole tree."
        ),
        "when_to_use": (
            "When you need interpretable non-linear rules, or as a building "
            "block for random forests and gradient boosting."
        ),
        "params": [
            {
                "name": "max_depth",
                "label": "Max depth (0 = unlimited)",
                "type": "int",
                "min": 0,
                "max": 30,
                "value": 0,
                "step": 1,
                "none_allowed": True,
                "help": "Maximum tree depth. 0 lets the tree grow until every "
                "leaf is pure (risks overfitting).",
            },
            {
                "name": "min_samples_split",
                "label": "Min samples per split",
                "type": "int",
                "min": 2,
                "max": 20,
                "value": 2,
                "step": 1,
                "help": "Minimum number of samples required to split a node. "
                "Higher values produce simpler trees.",
            },
        ],
    },
    "Random Forest Regressor": {
        "model_class": RandomForestRegressor,
        "requires_scaling": False,
        "supports_importance": True,
        "why": (
            "A random forest averages the predictions of **many decision "
            "trees**, each trained on a random bootstrap sample of the rows "
            "and a random subset of the features. Averaging smooths the "
            "step function of individual trees into a far more accurate "
            "predictor."
        ),
        "key_parameters": (
            "- **n_estimators**: the number of trees. More trees usually "
            "mean better, more stable predictions at the cost of speed.\n"
            "- **max_depth**: per-tree depth limit (0 = unlimited)."
        ),
        "advantages": (
            "- Very robust and often a strong default.\n"
            "- Reduces the overfitting of a single tree.\n"
            "- Handles mixed features and non-linear relationships.\n"
            "- Provides feature importances."
        ),
        "limitations": (
            "- Cannot extrapolate beyond the observed target range.\n"
            "- Slower to train and predict than linear models.\n"
            "- Less interpretable than a single tree."
        ),
        "when_to_use": (
            "A reliable general-purpose default for tabular regression when "
            "relationships may be non-linear."
        ),
        "params": [
            {
                "name": "n_estimators",
                "label": "Number of trees",
                "type": "int",
                "min": 10,
                "max": 500,
                "value": 100,
                "step": 10,
                "help": "Number of decision trees in the forest.",
            },
            {
                "name": "max_depth",
                "label": "Max depth (0 = unlimited)",
                "type": "int",
                "min": 0,
                "max": 30,
                "value": 0,
                "step": 1,
                "none_allowed": True,
                "help": "Maximum depth of each tree. 0 lets trees grow fully.",
            },
        ],
    },
    "Gradient Boosting Regressor": {
        "model_class": GradientBoostingRegressor,
        "requires_scaling": False,
        "supports_importance": True,
        "why": (
            "Gradient boosting trains **weak trees one after another**, each "
            "one learning to correct the residuals left by the previous "
            "trees. Because it optimizes against the errors directly, it is "
            "often the most accurate method on tabular data."
        ),
        "key_parameters": (
            "- **n_estimators**: number of boosting stages.\n"
            "- **learning_rate**: how strongly each tree updates the model. "
            "Smaller values are slower but usually more accurate.\n"
            "- **max_depth**: trees are typically kept shallow (2-4)."
        ),
        "advantages": (
            "- Usually the strongest out-of-the-box accuracy on tabular data.\n"
            "- Handles mixed features and provides feature importances.\n"
            "- Can model complex, non-linear relationships."
        ),
        "limitations": (
            "- Many hyperparameters to tune.\n"
            "- Sequential training is slower to fit.\n"
            "- Overfits easily when the learning rate is too high or trees "
            "too deep."
        ),
        "when_to_use": (
            "Structured/tabular problems where predictive accuracy matters "
            "most - the classic winner of tabular regression competitions."
        ),
        "params": [
            {
                "name": "n_estimators",
                "label": "Number of boosting stages",
                "type": "int",
                "min": 10,
                "max": 300,
                "value": 100,
                "step": 10,
                "help": "Number of boosting rounds.",
            },
            {
                "name": "learning_rate",
                "label": "Learning rate",
                "type": "float",
                "min": 0.01,
                "max": 1.0,
                "value": 0.1,
                "step": 0.05,
                "help": "How strongly each tree contributes. Smaller is slower "
                "but often more accurate.",
            },
            {
                "name": "max_depth",
                "label": "Max depth (0 = unlimited)",
                "type": "int",
                "min": 0,
                "max": 10,
                "value": 3,
                "step": 1,
                "none_allowed": True,
                "help": "Maximum depth of each weak tree. Shallow trees are "
                "typically best.",
            },
        ],
    },
    "KNN Regressor": {
        "model_class": KNeighborsRegressor,
        "requires_scaling": True,
        "supports_importance": False,
        "why": (
            "K-nearest neighbors regression predicts a new point by "
            "**averaging the target values of its k nearest training points** "
            "in feature space. Like its classification sibling, it is "
            "instance-based - the training data itself is the model."
        ),
        "key_parameters": (
            "- **n_neighbors (k)**: how many neighbors to average. Smaller k "
            "follows the data more closely; larger k smooths more.\n"
            "- **weights**: ``uniform`` averages neighbors equally; "
            "``distance`` weights each neighbor by how close it is."
        ),
        "advantages": (
            "- Simple and easy to understand.\n"
            "- No training phase.\n"
            "- Can fit very irregular, local patterns."
        ),
        "limitations": (
            "- Prediction is slow on large datasets.\n"
            "- Highly sensitive to feature scale - features must be scaled.\n"
            "- Cannot extrapolate beyond the observed target range.\n"
            "- No feature importances."
        ),
        "when_to_use": (
            "Small to medium datasets with low dimensionality where the "
            "relationship is highly local or irregular."
        ),
        "params": [
            {
                "name": "n_neighbors",
                "label": "Number of neighbors (k)",
                "type": "int",
                "min": 1,
                "max": 20,
                "value": 5,
                "step": 1,
                "help": "Number of nearest neighbors averaged for each "
                "prediction.",
            },
            {
                "name": "weights",
                "label": "Averaging weights",
                "type": "choice",
                "options": [("uniform", "uniform"), ("distance", "distance")],
                "value": "uniform",
                "help": "How to weight the neighbors' contributions.",
            },
        ],
    },
}

REGRESSOR_KEYS: list[str] = list(REGRESSORS.keys())


def get_regressor(model_key: str) -> dict:
    """Return the metadata dict for a regressor by its display name."""
    return REGRESSORS[model_key]


def default_regressor_params(model_key: str) -> dict:
    """Return ``{param_name: default_value}`` for a regressor."""
    return {param["name"]: param["value"] for param in REGRESSORS[model_key]["params"]}


def build_regressor(model_key: str, values: dict | None = None) -> object:
    """Construct an unfitted sklearn regressor from user-provided values.

    Args:
        model_key: Display name of the model in ``REGRESSORS``.
        values: Optional mapping of parameter name -> value. Missing
            parameters fall back to their spec defaults. Parameters flagged
            ``none_allowed`` translate a value of ``0`` to ``None``.
    """
    spec = REGRESSORS[model_key]
    values = values or {}
    kwargs = dict(spec.get("fixed_kwargs", {}))
    for param in spec["params"]:
        value = values.get(param["name"], param["value"])
        if param.get("none_allowed") and value == 0:
            value = None
        kwargs[param["name"]] = value
    return spec["model_class"](**kwargs)


def regressor_constructor_code(model_key: str, values: dict | None = None) -> str:
    """Return the Python snippet that builds the chosen regressor."""
    spec = REGRESSORS[model_key]
    values = values or {}
    kwargs = dict(spec.get("fixed_kwargs", {}))
    for param in spec["params"]:
        value = values.get(param["name"], param["value"])
        if param.get("none_allowed") and value == 0:
            continue
        kwargs[param["name"]] = value
    arguments = ", ".join(f"{name}={value!r}" for name, value in kwargs.items())
    return f"{spec['model_class'].__name__}({arguments})"
