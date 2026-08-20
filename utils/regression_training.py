"""Training helpers used by the Regression module.

Every function here is a pure function with no Streamlit or UI dependencies.
The central entry point is :func:`train_regressor`, which builds a leak-free
train/test split, applies a ``ColumnTransformer`` (or a caller-supplied
preprocessor) fitted on the training set only, fits the regressor inside a
``Pipeline``, and returns a self-contained results dict for the UI.

The preprocessor helpers are reused from ``utils.model_training`` so both
modeling modules share the same safe, leak-free preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.model_training import (
    build_default_preprocessor,
    categorical_feature_columns,
    transformed_feature_names,
)


def validate_regression_target(df, target: str) -> dict:
    """Validate that ``target`` is usable as a regression target.

    Returns:
        A dict with ``n_values``, ``missing``, ``dtype``, ``n_unique``,
        ``min``, ``max`` and ``mean`` describing the target.

    Raises:
        ValueError: When the target is missing, empty, non-numeric, or holds
            a single constant value.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The dataset must be a pandas DataFrame.")
    if target not in df.columns:
        raise ValueError(f"Target column `{target}` not found in the dataset.")

    series = df[target]
    valid = df.dropna(subset=[target])
    if len(valid) == 0:
        raise ValueError(f"Target column `{target}` has no usable values.")

    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"Target column `{target}` is not numeric. Regression predicts "
            "continuous numeric outcomes; for discrete categories open the "
            "Classification module instead."
        )

    n_unique = int(series.nunique(dropna=True))
    if n_unique < 2:
        raise ValueError(
            f"Target column `{target}` holds a single constant value, so "
            "there is nothing to predict. Pick a column that varies."
        )

    return {
        "n_values": int(len(valid)),
        "missing": int(series.isna().sum()),
        "dtype": str(series.dtype),
        "n_unique": n_unique,
        "min": float(valid[target].min()),
        "max": float(valid[target].max()),
        "mean": float(valid[target].mean()),
    }


def regression_metrics(y_true, y_pred) -> dict:
    """Return MAE, MSE, RMSE and R2 for the given predictions."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def prediction_frame(y_true, y_pred) -> pd.DataFrame:
    """Return a DataFrame with Actual, Predicted, and Residual columns."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    return pd.DataFrame(
        {
            "Actual": actual,
            "Predicted": predicted,
            "Residual": actual - predicted,
        }
    )


def train_regressor(
    X,
    y,
    estimator,
    preprocessor=None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train a regressor on a leak-free split and return a results dict.

    Args:
        X: Feature DataFrame (raw, untransformed).
        y: Numeric target Series.
        estimator: An unfitted sklearn regressor.
        preprocessor: Optional ``ColumnTransformer``. When ``None``, a default
            one is built from the dtypes of ``X``. Either way it is fitted on
            the **training set only**.
        test_size: Fraction of rows held out for evaluation.
        random_state: Seed for reproducible splits.

    Returns:
        A dict with the fitted ``pipeline``, the split arrays, predictions, a
        ``predictions`` table (actual vs predicted vs residual), ``metrics``,
        ``classes`` (empty for regression), ``feature_names``, and train/test
        R2.

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

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
    except ValueError as exc:
        raise ValueError(
            "The train/test split failed. Reduce the test size or check the "
            "dataset."
        ) from exc

    if preprocessor is None:
        preprocessor = build_default_preprocessor(X_train)

    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    return {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "feature_names": transformed_feature_names(preprocessor),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": regression_metrics(y_test, y_pred),
        "predictions": prediction_frame(y_test, y_pred),
        "train_score": float(pipeline.score(X_train, y_train)),
        "test_score": float(pipeline.score(X_test, y_test)),
    }


def predict_sample(pipeline, row) -> float:
    """Predict a single row (dict or Series) with a fitted pipeline."""
    frame = pd.DataFrame([row])
    return float(pipeline.predict(frame)[0])


def training_code(
    model_label: str,
    estimator_code: str,
    target: str,
    features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """Return a complete, copy-paste-ready Python training script."""
    quoted_features = ", ".join(repr(feature) for feature in features)
    return (
        f"# Train a {model_label} regressor to predict {target!r}\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
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
        f"model = Pipeline([('preprocessor', preprocessor), ('regressor', {estimator_code})])\n"
        "\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "y_pred = model.predict(X_test)\n"
        "print('MAE:', mean_absolute_error(y_test, y_pred))\n"
        "print('MSE:', mean_squared_error(y_test, y_pred))\n"
        "print('RMSE:', mean_squared_error(y_test, y_pred) ** 0.5)\n"
        "print('R2:', r2_score(y_test, y_pred))\n"
    )
