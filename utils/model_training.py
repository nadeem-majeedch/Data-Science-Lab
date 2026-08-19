"""Training helpers used by the Classification module.

Every function here is a pure function with no Streamlit or UI dependencies,
which keeps them easy to unit-test. The central entry point is
:func:`train_classifier`, which builds a leak-free train/test split, applies a
``ColumnTransformer`` (or a caller-supplied preprocessor), fits the classifier
inside a ``Pipeline``, and returns a self-contained results dict for the UI.

Data leakage note: any preprocessor - the default one built here or one
supplied by the caller - is **fitted on the training set only** and then
applied unchanged to the test set, so the evaluation is honest.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_CLASSES = 20

NUMERIC_DTYPES = ["number"]
CATEGORICAL_DTYPES = ["object", "category", "string"]


def validate_classification_target(df, target: str) -> dict:
    """Validate that ``target`` is usable as a classification target.

    Args:
        df: The dataset to classify on.
        target: Column name to predict.

    Returns:
        A dict with ``n_classes``, ``classes``, ``counts``, ``missing`` and
        ``dtype`` describing the target.

    Raises:
        ValueError: With a helpful message when the target is missing, empty,
            has fewer than two classes, or looks like a regression target
            (a numeric column with many unique values).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The dataset must be a pandas DataFrame.")
    if target not in df.columns:
        raise ValueError(f"Target column `{target}` not found in the dataset.")

    series = df[target]
    valid = df.dropna(subset=[target])
    if len(valid) == 0:
        raise ValueError(f"Target column `{target}` has no usable values.")

    target_values = valid[target]
    counts = target_values.value_counts()
    n_classes = len(counts)

    if n_classes < 2:
        raise ValueError(
            f"Target column `{target}` has only one class. Classification "
            "needs at least two classes to predict between."
        )

    if pd.api.types.is_numeric_dtype(series) and n_classes > MAX_CLASSES:
        raise ValueError(
            f"Target column `{target}` looks like a numeric (regression) "
            f"target with {n_classes} unique values. Pick a categorical "
            "column with a small number of classes, or open the Regression "
            "module for continuous outcomes."
        )

    return {
        "n_classes": int(n_classes),
        "classes": [str(value) for value in counts.index],
        "counts": counts,
        "missing": int(series.isna().sum()),
        "dtype": str(series.dtype),
    }


def categorical_feature_columns(df) -> list[str]:
    """Return columns suitable for categorical encoding (text/category)."""
    return df.select_dtypes(include=CATEGORICAL_DTYPES).columns.tolist()


def build_default_preprocessor(X):
    """Build a leak-free ColumnTransformer from the feature dtypes of ``X``.

    Numeric features are median-imputed and standardized; categorical features
    are mode-imputed and one-hot encoded (tolerating unseen categories). Any
    other column types (e.g. raw datetime columns) are dropped.

    The returned transformer must be **fitted on the training set only**.
    """
    numeric = X.select_dtypes(include=NUMERIC_DTYPES).columns.tolist()
    categorical = categorical_feature_columns(X)

    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )

    if not transformers:
        return None
    return ColumnTransformer(transformers, remainder="drop")


def transformed_feature_names(preprocessor):
    """Return the transformed feature names of a fitted preprocessor.

    Returns ``None`` when the preprocessor cannot report feature names.
    """
    if preprocessor is None:
        return None
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except (AttributeError, NotImplementedError, ValueError):
        return None


def train_classifier(
    X,
    y,
    estimator,
    preprocessor=None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> dict:
    """Train a classifier on a leak-free split and return a results dict.

    Args:
        X: Feature DataFrame (raw, untransformed).
        y: Target Series.
        estimator: An unfitted sklearn classifier.
        preprocessor: Optional ``ColumnTransformer``. When ``None``, a default
            one is built from the dtypes of ``X``. Either way it is fitted on
            the **training set only**.
        test_size: Fraction of rows held out for evaluation.
        random_state: Seed for reproducible splits.
        stratify: Whether to preserve class proportions in both splits.

    Returns:
        A dict with the fitted ``pipeline``, the split arrays, predictions,
        probabilities, ``metrics``, a classification-report DataFrame, a
        confusion-matrix DataFrame, ``classes``, ``feature_names`` and
        train/test accuracy.

    Raises:
        ValueError: When there are no usable rows or features, or when the
            split cannot be created.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Features must be provided as a pandas DataFrame.")

    if isinstance(y, pd.Series):
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

    if len(X) == 0:
        raise ValueError("No valid rows remain after dropping missing targets.")
    if X.shape[1] == 0:
        raise ValueError("Select at least one feature column to train on.")

    stratify_y = y if stratify else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )
    except ValueError as exc:
        raise ValueError(
            "The train/test split failed. When stratification is enabled, "
            "every target class needs at least two members. Reduce the test "
            "size or disable stratification."
        ) from exc

    if preprocessor is None:
        preprocessor = build_default_preprocessor(X_train)

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_proba = None

    raw_classes = list(pipeline.classes_)
    classes = [str(value) for value in raw_classes]
    metrics = classification_metrics(y_test, y_pred, n_classes=len(classes))

    return {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "feature_names": transformed_feature_names(preprocessor),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "classes": classes,
        "report": classification_report_frame(y_test, y_pred, raw_classes),
        "confusion_matrix": confusion_matrix_frame(y_test, y_pred, raw_classes),
        "train_score": float(pipeline.score(X_train, y_train)),
        "test_score": float(pipeline.score(X_test, y_test)),
    }


def classification_metrics(y_true, y_pred, n_classes: int | None = None) -> dict:
    """Return accuracy, precision, recall and F1.

    Binary problems use the standard binary averages; multi-class problems use
    macro averages (each class weighted equally, regardless of size).
    """
    if n_classes is None:
        n_classes = len(set(y_true))
    average = "binary" if n_classes == 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "average": average,
    }


def classification_report_frame(y_true, y_pred, labels) -> pd.DataFrame:
    """Return the classification report as a tidy DataFrame.

    One row per class, plus macro and weighted averages.
    """
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    rows = []
    for label in labels:
        values = report[str(label)]
        rows.append(
            {
                "Class": str(label),
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-score": values["f1-score"],
                "Support": int(values["support"]),
            }
        )
    for avg_key, avg_label in (
        ("macro avg", "Macro average"),
        ("weighted avg", "Weighted average"),
    ):
        values = report[avg_key]
        rows.append(
            {
                "Class": avg_label,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-score": values["f1-score"],
                "Support": int(values["support"]),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_frame(y_true, y_pred, labels) -> pd.DataFrame:
    """Return the confusion matrix as a labeled DataFrame."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )


def predict_sample(pipeline, row) -> dict:
    """Predict a single row (dict or Series) with a fitted pipeline.

    Returns:
        A dict with the ``prediction``, ``probabilities`` (a NumPy array or
        ``None``) and the model's ``classes``.
    """
    frame = pd.DataFrame([row])
    prediction = pipeline.predict(frame)[0]
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(frame)[0]
        except Exception:
            probabilities = None
    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "classes": [str(value) for value in getattr(pipeline, "classes_", [])],
    }


def training_code(
    model_label: str,
    estimator_code: str,
    target: str,
    features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> str:
    """Return a complete, copy-paste-ready Python training script."""
    quoted_features = ", ".join(repr(feature) for feature in features)
    stratify_line = "        stratify=y,\n" if stratify else ""
    return (
        f"# Train a {model_label} classifier to predict {target!r}\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.metrics import classification_report, confusion_matrix\n"
        "\n"
        "# Load your dataset\n"
        "# df = pd.read_csv('your_data.csv')\n"
        "\n"
        f"target = {target!r}\n"
        f"features = [{quoted_features}]\n"
        "\n"
        "X = df[features]\n"
        "y = df[target]\n"
        "\n"
        "# Drop rows whose target is missing\n"
        "X = X[y.notna()]\n"
        "y = y[y.notna()]\n"
        "\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y,\n"
        f"    test_size={test_size},\n"
        f"    random_state={random_state},\n"
        f"{stratify_line}"
        ")\n"
        "\n"
        "# Auto preprocessing: impute + scale numeric, impute + one-hot encode text\n"
        "numeric = X_train.select_dtypes(include='number').columns.tolist()\n"
        "categorical = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()\n"
        "transformers = []\n"
        "if numeric:\n"
        "    transformers.append(('numeric', Pipeline([\n"
        "        ('imputer', SimpleImputer(strategy='median')),\n"
        "        ('scaler', StandardScaler()),\n"
        "    ]), numeric))\n"
        "if categorical:\n"
        "    transformers.append(('categorical', Pipeline([\n"
        "        ('imputer', SimpleImputer(strategy='most_frequent')),\n"
        "        ('encoder', OneHotEncoder(handle_unknown='ignore')),\n"
        "    ]), categorical))\n"
        "preprocessor = ColumnTransformer(transformers, remainder='drop')\n"
        "\n"
        f"model = Pipeline([('preprocessor', preprocessor), ('classifier', {estimator_code})])\n"
        "\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "y_pred = model.predict(X_test)\n"
        "print(classification_report(y_test, y_pred))\n"
        "print(confusion_matrix(y_test, y_pred))\n"
    )
