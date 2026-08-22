"""Model comparison helpers used by the Model Comparison module.

Every function here is a pure function with no Streamlit dependency so it can
be unit-tested directly. The key guarantee of a fair comparison is that every
model sees the **same** data: a single train/test split and a single
preprocessor fitted on the training set only. The models differ only in the
algorithm, so differences in the metrics reflect the algorithms - not
different data handling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.evaluation import roc_auc_brief
from utils.model_training import (
    build_default_preprocessor,
    classification_metrics,
)
from utils.models import (
    MODEL_KEYS,
    build_classifier,
    classifier_constructor_code,
)
from utils.regressors import (
    REGRESSOR_KEYS,
    build_regressor,
    regressor_constructor_code,
)
from utils.regression_training import regression_metrics

CLASSIFICATION_COLUMNS = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]

REGRESSION_COLUMNS = ["Model", "MAE", "RMSE", "R2"]


def _shared_split(X, y, test_size: float, random_state: int, stratify):
    """Create the one train/test split shared by every model."""
    if isinstance(y, pd.Series):
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
    if len(X) == 0:
        raise ValueError("No valid rows remain after dropping missing targets.")
    if X.shape[1] == 0:
        raise ValueError("Select at least one feature column to compare models.")
    stratify_y = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )


def _with_random_state(estimator, random_state: int):
    """Pin ``random_state`` on estimators that accept it.

    Tree ensembles and other randomized estimators default to an unpinned
    global RNG state, which would make every comparison run slightly
    different. Seeding them keeps the comparison reproducible.
    """
    if "random_state" in estimator.get_params(deep=False):
        estimator.set_params(random_state=random_state)
    return estimator


def compare_classifiers(
    X,
    y,
    model_keys=None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> dict:
    """Train every classifier on the same split and preprocessor.

    Args:
        X: Feature DataFrame (raw, untransformed).
        y: Target Series.
        model_keys: Display names of the classifiers to compare; defaults to
            all models in ``MODELS``.
        test_size: Fraction of rows held out for evaluation.
        random_state: Seed for the reproducible split.
        stratify: Whether to preserve class proportions in both splits.

    Returns:
        A dict with the comparison ``table`` (one row per model), ``details``
        (per-model metrics, fitted pipeline, predictions and probabilities),
        the shared split arrays, and the ``config`` describing the run.

    Raises:
        ValueError: When the split cannot be created or no models are chosen.
    """
    model_keys = list(model_keys) if model_keys is not None else list(MODEL_KEYS)
    if not model_keys:
        raise ValueError("Choose at least one model to compare.")

    X_train, X_test, y_train, y_test = _shared_split(
        X, y, test_size, random_state, stratify
    )
    preprocessor = build_default_preprocessor(X_train)

    rows = []
    details = {}
    for key in model_keys:
        estimator = _with_random_state(build_classifier(key), random_state)
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = classification_metrics(
            y_test, y_pred, n_classes=len(pipeline.classes_)
        )
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
            except Exception:
                y_proba = None
        auc_value = roc_auc_brief(y_test, y_proba)
        rows.append(
            {
                "Model": key,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "AUC": auc_value,
            }
        )
        details[key] = {
            "metrics": metrics,
            "pipeline": pipeline,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "auc": auc_value,
            "classes": [str(value) for value in pipeline.classes_],
        }

    return {
        "table": pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS),
        "details": details,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "config": {
            "task": "classification",
            "model_keys": model_keys,
            "test_size": test_size,
            "random_state": random_state,
            "stratify": stratify,
        },
    }


def compare_regressors(
    X,
    y,
    model_keys=None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train every regressor on the same split and preprocessor.

    Args:
        X: Feature DataFrame (raw, untransformed).
        y: Numeric target Series.
        model_keys: Display names of the regressors to compare; defaults to
            all models in ``REGRESSORS``.
        test_size: Fraction of rows held out for evaluation.
        random_state: Seed for the reproducible split.

    Returns:
        A dict with the comparison ``table`` (one row per model), ``details``
        (per-model metrics, fitted pipeline, predictions), the shared split
        arrays, and the ``config`` describing the run.

    Raises:
        ValueError: When the split cannot be created or no models are chosen.
    """
    model_keys = list(model_keys) if model_keys is not None else list(REGRESSOR_KEYS)
    if not model_keys:
        raise ValueError("Choose at least one model to compare.")

    X_train, X_test, y_train, y_test = _shared_split(
        X, y, test_size, random_state, None
    )
    preprocessor = build_default_preprocessor(X_train)

    rows = []
    details = {}
    for key in model_keys:
        estimator = _with_random_state(build_regressor(key), random_state)
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        rows.append(
            {
                "Model": key,
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "R2": metrics["r2"],
            }
        )
        details[key] = {
            "metrics": metrics,
            "pipeline": pipeline,
            "y_pred": y_pred,
        }

    return {
        "table": pd.DataFrame(rows, columns=REGRESSION_COLUMNS),
        "details": details,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "config": {
            "task": "regression",
            "model_keys": model_keys,
            "test_size": test_size,
            "random_state": random_state,
            "stratify": False,
        },
    }


def best_model(table: pd.DataFrame) -> str:
    """Return the display name of the top-ranked model.

    Classification ranks on accuracy, then F1; regression ranks on R2 (higher
    is better). Used only to highlight a candidate, never as the final word.
    """
    if "Accuracy" in table.columns:
        return str(
            table.sort_values(["Accuracy", "F1"], ascending=False).iloc[0]["Model"]
        )
    return str(table.sort_values("R2", ascending=False).iloc[0]["Model"])


def _seeded_constructor(kind: str, key: str, seed: int) -> str:
    """Return the constructor snippet for a model, seeded when supported.

    Keeps the generated script reproducible in exactly the same way the app
    pins ``random_state`` on randomized estimators.
    """
    if kind == "classification":
        estimator = build_classifier(key)
        code = classifier_constructor_code(key)
    else:
        estimator = build_regressor(key)
        code = regressor_constructor_code(key)
    if "random_state" in estimator.get_params(deep=False):
        code = f"{code[:-1]}, random_state={seed})"
    return code


def comparison_code(kind: str, config: dict) -> str:
    """Return a copy-paste-ready Python script comparing several models.

    Args:
        kind: ``"classification"`` or ``"regression"``.
        config: Dict with ``target``, ``features``, ``model_keys``,
            ``test_size`` and ``random_state``.
    """
    target = config["target"]
    features = config["features"]
    model_keys = config["model_keys"]
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    quoted_features = ", ".join(repr(feature) for feature in features)

    if kind == "classification":
        metric_import = (
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score"
        )
        row_lines = []
        for key in model_keys:
            constructor = _seeded_constructor("classification", key, random_state)
            row_lines.append(
                f"# {key}\n"
                f"pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', {constructor})])\n"
                "pipeline.fit(X_train, y_train)\n"
                "y_pred = pipeline.predict(X_test)\n"
                "average = 'binary' if len(pipeline.classes_) == 2 else 'macro'\n"
                "row = {'Model': " + repr(key) + ",\n"
                "       'Accuracy': accuracy_score(y_test, y_pred),\n"
                "       'Precision': precision_score(y_test, y_pred, average=average, zero_division=0),\n"
                "       'Recall': recall_score(y_test, y_pred, average=average, zero_division=0),\n"
                "       'F1': f1_score(y_test, y_pred, average=average, zero_division=0)}\n"
                "if len(pipeline.classes_) == 2 and hasattr(pipeline, 'predict_proba'):\n"
                "    from sklearn.metrics import roc_auc_score\n"
                "    row['AUC'] = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1], pos_label=pipeline.classes_[1])\n"
                "results.append(row)"
            )
        summary = (
            "best = table.sort_values(['Accuracy', 'F1'], ascending=False).iloc[0]\n"
            "print('Top model by accuracy:', best['Model'])"
        )
    else:
        metric_import = (
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"
        )
        row_lines = []
        for key in model_keys:
            constructor = _seeded_constructor("regression", key, random_state)
            row_lines.append(
                f"# {key}\n"
                f"pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', {constructor})])\n"
                "pipeline.fit(X_train, y_train)\n"
                "y_pred = pipeline.predict(X_test)\n"
                "row = {'Model': " + repr(key) + ",\n"
                "       'MAE': mean_absolute_error(y_test, y_pred),\n"
                "       'RMSE': mean_squared_error(y_test, y_pred) ** 0.5,\n"
                "       'R2': r2_score(y_test, y_pred)}\n"
                "results.append(row)"
            )
        summary = (
            "best = table.sort_values('R2', ascending=False).iloc[0]\n"
            "print('Top model by R2:', best['Model'])"
        )

    per_model = "\n".join(row_lines)

    return (
        f"# Compare {len(model_keys)} models for {target!r} on the SAME split\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n"
        f"{metric_import}\n"
        "\n"
        f"target = {target!r}\n"
        f"features = [{quoted_features}]\n"
        "X = df[features]\n"
        "y = df[target]\n"
        "X = X[y.notna()]\n"
        "y = y[y.notna()]\n"
        "\n"
        "# ONE split shared by every model -> fair, reproducible comparison\n"
        f"X_train, X_test, y_train, y_test = train_test_split(\n"
        f"    X, y, test_size={test_size}, random_state={random_state}\n"
        ")\n"
        "\n"
        "# ONE preprocessor fitted on the training set only, reused by every model\n"
        "numeric = X_train.select_dtypes(include='number').columns.tolist()\n"
        "categorical = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n"
        "transformers = []\n"
        "if numeric:\n"
        "    transformers.append(('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric))\n"
        "if categorical:\n"
        "    transformers.append(('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical))\n"
        "preprocessor = ColumnTransformer(transformers, remainder='drop')\n"
        "\n"
        "# Train every model on the identical data\n"
        "results = []\n"
        f"{per_model}\n"
        "\n"
        "# Build the comparison table and rank the models\n"
        "table = pd.DataFrame(results)\n"
        "print(table.round(4))\n"
        f"{summary}\n"
    )
