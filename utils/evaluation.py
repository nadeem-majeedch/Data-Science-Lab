"""Evaluation helpers used by the Model Evaluation module.

Everything here is a pure function with no Streamlit dependency so it can be
unit-tested directly. The helpers operate on the self-contained results dicts
produced by :func:`utils.model_training.train_classifier` and
:func:`utils.regression_training.train_regressor`, so the Model Evaluation page
can go *deeper* than the training pages: ROC curves and AUC, cross-validation,
and residual diagnostics, all with per-metric educational content.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from utils.model_training import build_default_preprocessor

# Educational content rendered for each headline metric. Kept in one place so
# the Classification, Regression and Model Evaluation pages all teach the
# same definitions.
METRIC_GUIDANCE: dict[str, str] = {
    "accuracy": (
        "**Accuracy** is the fraction of test rows predicted correctly. It is "
        "easy to understand but can mislead on imbalanced data: a model that "
        "always predicts the majority class can look very accurate while "
        "being useless for the minority class."
    ),
    "precision": (
        "**Precision** answers: *of the rows the model labeled positive, how "
        "many really were positive?* High precision means few false alarms. "
        "Use it when a wrong positive is costly (e.g. flagging a good loan "
        "as risky)."
    ),
    "recall": (
        "**Recall** (sensitivity) answers: *of the rows that really are "
        "positive, how many did the model catch?* High recall means few "
        "missed cases. Use it when missing a positive is costly (e.g. not "
        "detecting a disease)."
    ),
    "f1": (
        "**F1** is the harmonic mean of precision and recall. It is a single "
        "number that balances the two and is especially useful when you "
        "cannot trade one off against the other, or when the classes are "
        "imbalanced."
    ),
    "auc": (
        "**AUC** (area under the ROC curve) measures how well the model "
        "separates the classes across *every* decision threshold. 0.5 is "
        "random guessing, 1.0 is a perfect separator. AUC is threshold-free, "
        "but it can hide problems in a specific operating region that you "
        "actually care about."
    ),
    "mae": (
        "**MAE** (mean absolute error) is the average absolute difference "
        "between predictions and actual values. It is measured in the units "
        "of the target, which makes it the easiest error to explain to "
        "non-technical audiences. Every error is weighted equally."
    ),
    "mse": (
        "**MSE** (mean squared error) averages the *squared* differences. "
        "Because errors are squared, large errors dominate the score - the "
        "model is punished much harder for a big miss than for several small "
        "ones."
    ),
    "rmse": (
        "**RMSE** is the square root of MSE, which brings the value back into "
        "the units of the target. It keeps MSE's harsh penalty on large "
        "errors while remaining interpretable. Compare it with MAE: a large "
        "gap between the two means the errors are uneven - a few big misses "
        "inflate RMSE."
    ),
    "r2": (
        "**R2** (coefficient of determination) is the share of the target's "
        "variance explained by the model. 1.0 is a perfect fit, 0.0 means the "
        "model is no better than predicting the mean, and negative values mean "
        "it is worse. R2 is scale-free, so it is handy for comparing models on "
        "the same target."
    ),
}

# Why a single metric is never the whole story - rendered prominently on both
# the Evaluation and Comparison pages.
HIGHEST_NOT_BEST = (
    "**The model with the highest single metric is not automatically the best "
    "model for your problem.**\n\n"
    "- Different metrics answer different questions. A model can win on "
    "accuracy while losing on recall - and recall may be the metric that "
    "matters for your use case.\n"
    "- **Imbalanced data** can make accuracy and even AUC look flattering "
    "while the model fails on the rare class.\n"
    "- A model that wins on the test set by a tiny margin may just be "
    "lucky; **cross-validation** gives a more stable estimate.\n"
    "- Consider the **cost of errors**, not just their count. A mislabeled "
    "cancer scan is very different from a mislabeled spam email.\n\n"
    "Look at the whole picture: the table, the charts, and what the problem "
    "actually requires."
)


def get_evaluation_source():
    """Return ``(kind, results)`` for the model currently in session.

    ``kind`` is ``"classification"`` or ``"regression"``; ``results`` is the
    self-contained dict stored by the corresponding training page. Returns
    ``(None, None)`` when no model has been trained yet.
    """
    import streamlit as st

    for kind, key in (
        ("classification", "classification_results"),
        ("regression", "regression_results"),
    ):
        if key in st.session_state:
            return kind, st.session_state[key]
    return None, None


def encode_labels(y_true):
    """Return label-encoded (0..n-1) integers for ``y_true``."""
    return LabelEncoder().fit_transform(np.asarray(y_true))


def roc_auc_brief(y_true, y_proba):
    """Return a single AUC number, or ``None`` when probabilities are absent.

    Binary problems use the standard binary AUC; multi-class problems use the
    one-vs-rest macro average. Returns ``None`` when the model exposes no
    ``predict_proba`` or when the score cannot be computed.
    """
    if y_proba is None:
        return None
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2 or y_proba.shape[1] < 2:
        return None
    labels = encode_labels(y_true)
    if len(np.unique(labels)) < 2:
        return None
    try:
        if y_proba.shape[1] == 2:
            return float(roc_auc_score(labels, y_proba[:, 1]))
        return float(
            roc_auc_score(labels, y_proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        return None


def roc_curves(y_true, y_proba, classes):
    """Return one ROC curve dict per class (one-vs-rest).

    Each dict holds ``class``, ``fpr``, ``tpr`` and ``auc`` so the page can
    plot every curve on the same axes. Returns an empty list when the model
    exposes no probabilities.
    """
    if y_proba is None:
        return []
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2:
        return []
    labels = encode_labels(y_true)
    curves = []
    for index, label in enumerate(classes):
        if index >= y_proba.shape[1]:
            continue
        positive = labels == index
        if positive.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(positive, y_proba[:, index])
        curves.append(
            {
                "class": str(label),
                "fpr": fpr,
                "tpr": tpr,
                "auc": float(auc(fpr, tpr)),
            }
        )
    return curves


def cross_validate(
    build_estimator,
    X,
    y,
    task: str = "classification",
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Cross-validate a model builder and return per-fold scores.

    Args:
        build_estimator: Zero-argument callable returning an unfitted sklearn
            estimator (the chosen algorithm).
        X: Raw feature DataFrame.
        y: Target Series.
        task: ``"classification"`` (stratified folds) or ``"regression"``.
        n_folds: Number of folds (each fold re-fits a fresh, leak-free
            preprocessor from its own training portion).
        random_state: Seed for the fold shuffling.

    Returns:
        A dict with ``scores`` (list), ``mean``, ``std``, ``min``, ``max``
        and ``folds``.

    Raises:
        ValueError: When the fold strategy cannot be applied to the data
            (e.g. a class with fewer members than folds).
    """
    if task == "classification":
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    scores = []
    try:
        if task == "classification":
            values, counts = np.unique(np.asarray(y), return_counts=True)
            if counts.min() < n_folds:
                raise ValueError(
                    "every target class has fewer members than there are folds. "
                    "Reduce the number of folds or use a larger, more balanced "
                    "dataset."
                )
        for train_index, val_index in splitter.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            preprocessor = build_default_preprocessor(X_train)
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", build_estimator())])
            pipeline.fit(X_train, y_train)
            scores.append(float(pipeline.score(X_val, y_val)))
    except ValueError as exc:
        raise ValueError(
            f"Cross-validation failed: {exc}. For classification, every class "
            "needs at least as many members as there are folds - reduce the "
            "number of folds or use a larger dataset."
        ) from exc

    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "folds": n_folds,
    }


def residual_statistics(predictions: pd.DataFrame) -> dict:
    """Summarize the residuals from an actual/predicted/residual table."""
    residuals = pd.to_numeric(predictions["Residual"])
    return {
        "mean": float(residuals.mean()),
        "std": float(residuals.std(ddof=0)),
        "min": float(residuals.min()),
        "median": float(residuals.median()),
        "max": float(residuals.max()),
    }


def evaluation_code(kind: str, config: dict) -> str:
    """Return a copy-paste-ready Python script mirroring the evaluation steps.

    Args:
        kind: ``"classification"`` or ``"regression"``.
        config: The ``config`` dict stored with the training results, or a
            dict with the same keys (``model_key``, ``params``, ``target``,
            ``features``, ``random_state``).
    """
    model_key = config["model_key"]
    target = config["target"]
    features = config["features"]
    random_state = config.get("random_state", 42)
    estimator_code = config.get("estimator_code", f"{model_key}()")
    quoted_features = ", ".join(repr(feature) for feature in features)

    if kind == "classification":
        return (
            f"# Evaluate the {model_key} classifier for {target!r}\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold\n"
            "from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score\n"
            "\n"
            f"target = {target!r}\n"
            f"features = [{quoted_features}]\n"
            "X = df[features]\n"
            "y = df[target]\n"
            "X = X[y.notna()]\n"
            "y = y[y.notna()]\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n"
            "\n"
            "# Same leak-free preprocessing as the lab: impute + scale + one-hot\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "numeric = X_train.select_dtypes(include='number').columns.tolist()\n"
            "categorical = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n"
            "transformers = []\n"
            "if numeric:\n"
            "    transformers.append(('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric))\n"
            "if categorical:\n"
            "    transformers.append(('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical))\n"
            "preprocessor = ColumnTransformer(transformers, remainder='drop')\n"
            "\n"
            f"model = Pipeline([('preprocessor', preprocessor), ('classifier', {estimator_code})])\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "y_pred = model.predict(X_test)\n"
            "print(classification_report(y_test, y_pred))\n"
            "print(confusion_matrix(y_test, y_pred))\n"
            "\n"
            "# ROC curve and AUC for binary targets (one-vs-rest for multi-class)\n"
            "y_proba = model.predict_proba(X_test)\n"
            "if len(model.classes_) == 2:\n"
            "    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=model.classes_[1])\n"
            "    print('AUC:', auc(fpr, tpr))\n"
            "else:\n"
            "    print('Macro AUC (one-vs-rest):', roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))\n"
            "\n"
            "# Cross-validation: a more stable estimate than one split\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "scores = cross_val_score(model, X, y, cv=cv)\n"
            "print('CV accuracy: mean %.3f (std %.3f)' % (scores.mean(), scores.std()))\n"
        )

    return (
        f"# Evaluate the {model_key} regressor for {target!r}\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split, cross_val_score, KFold\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
        "\n"
        f"target = {target!r}\n"
        f"features = [{quoted_features}]\n"
        "X = df[features]\n"
        "y = df[target]\n"
        "X = X[y.notna()]\n"
        "y = y[y.notna()]\n"
        "\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42\n"
        ")\n"
        "\n"
        "# Same leak-free preprocessing as the lab: impute + scale + one-hot\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n"
        "numeric = X_train.select_dtypes(include='number').columns.tolist()\n"
        "categorical = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n"
        "transformers = []\n"
        "if numeric:\n"
        "    transformers.append(('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric))\n"
        "if categorical:\n"
        "    transformers.append(('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical))\n"
        "preprocessor = ColumnTransformer(transformers, remainder='drop')\n"
        "\n"
        f"model = Pipeline([('preprocessor', preprocessor), ('regressor', {estimator_code})])\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "y_pred = model.predict(X_test)\n"
        "print('MAE:', mean_absolute_error(y_test, y_pred))\n"
        "print('MSE:', mean_squared_error(y_test, y_pred))\n"
        "print('RMSE:', mean_squared_error(y_test, y_pred) ** 0.5)\n"
        "print('R2:', r2_score(y_test, y_pred))\n"
        "\n"
        "# Residual diagnostics: errors should scatter randomly around zero\n"
        "residuals = y_test - y_pred\n"
        "print('Residual mean: %.4f (should be near 0)' % residuals.mean())\n"
        "print('Residual std: %.4f' % residuals.std())\n"
        "\n"
        "# Cross-validation: a more stable estimate than one split\n"
        "cv = KFold(n_splits=5, shuffle=True, random_state=42)\n"
        "scores = cross_val_score(model, X, y, cv=cv)\n"
        "print('CV R2: mean %.3f (std %.3f)' % (scores.mean(), scores.std()))\n"
    )
