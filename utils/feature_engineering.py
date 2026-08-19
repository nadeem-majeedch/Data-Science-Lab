"""Feature engineering helpers used by the Feature Engineering module.

Every function here is a pure function operating on a pandas DataFrame with
no Streamlit or UI dependencies, which keeps them easy to unit-test.

Two layers are provided:

* concrete transformations (``create_numeric_feature``,
  ``apply_math_transform``, ``bin_numeric``, ...) that add or remove columns
* :func:`apply_feature_op`, a thin dispatcher over "operation" dicts so the
  UI can record, replay, and undo a chain of operations on the dataset
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures

NUMERIC_OPERATIONS = ["sum", "difference", "product", "ratio"]

MATH_TRANSFORMS = ["log", "sqrt", "square"]

MATH_SUFFIXES = {"log": "_log", "sqrt": "_sqrt", "square": "_squared"}

BINNING_METHODS = ["equal width", "quantile"]

DATETIME_PARTS = ["year", "month", "day", "weekday"]

TEXT_PARTS = ["length", "word count"]


def _validate_columns(df, columns) -> list[str]:
    """Return the selected columns that actually exist in the DataFrame."""
    return [column for column in columns if column in df.columns]


def create_numeric_feature(df, col_a, col_b, operation="sum", name=None):
    """Create a new numeric feature by combining two columns.

    Args:
        df: Input DataFrame.
        col_a: First numeric column.
        col_b: Second numeric column.
        operation: ``sum``, ``difference``, ``product`` or ``ratio``.
        name: Optional name for the new column.

    Returns:
        ``(result, info)`` where info contains the new column name.

    Raises:
        ValueError: If a column is missing, or ``ratio`` divides by zero.
    """
    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError("Both columns must exist in the dataset.")

    default_operator = {
        "sum": "plus",
        "difference": "minus",
        "product": "times",
        "ratio": "over",
    }
    column = name or f"{col_a}_{default_operator[operation]}_{col_b}"
    result = df.copy()

    if operation == "sum":
        result[column] = df[col_a] + df[col_b]
    elif operation == "difference":
        result[column] = df[col_a] - df[col_b]
    elif operation == "product":
        result[column] = df[col_a] * df[col_b]
    else:  # ratio
        if (df[col_b] == 0).any():
            raise ValueError(
                f"`{col_b}` contains zero, which would create undefined "
                "divisions. Filter or fix those rows first."
            )
        result[column] = df[col_a] / df[col_b]

    return result, {"column": column, "operation": operation}


def apply_math_transform(df, columns, method="log"):
    """Apply a unary math transform, adding new columns ``<col><suffix>``.

    Original columns are kept untouched so the transformation is easy to
    inspect and undo.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to transform.
        method: ``log``, ``sqrt`` or ``square``.

    Returns:
        ``(result, info)``.

    Raises:
        ValueError: If the transform is undefined for any value (log of a
            non-positive number, sqrt of a negative number).
    """
    columns = _validate_columns(df, columns)
    result = df.copy()
    suffix = MATH_SUFFIXES[method]
    new_columns = []

    for column in columns:
        series = df[column]
        if method == "log":
            if (series <= 0).any():
                raise ValueError(
                    f"Cannot compute log of `{column}`: it contains "
                    "non-positive values. Shift the values first."
                )
            transformed = np.log(series)
        elif method == "sqrt":
            if (series < 0).any():
                raise ValueError(
                    f"Cannot compute sqrt of `{column}`: it contains negative "
                    "values."
                )
            transformed = np.sqrt(series)
        else:  # square
            transformed = series**2
        new_column = f"{column}{suffix}"
        result[new_column] = transformed
        new_columns.append(new_column)

    return result, {"method": method, "columns": new_columns}


def bin_numeric(df, column, n_bins=4, method="equal width", labels=None):
    """Bin a numeric column into discrete categories.

    Args:
        df: Input DataFrame.
        column: Numeric column to bin.
        n_bins: Number of bins.
        method: ``equal width`` or ``quantile``.
        labels: Optional list of labels for the bins.

    Returns:
        ``(result, info)`` with a new ``<column>_binned`` categorical column.
    """
    if column not in df.columns:
        raise ValueError(f"Column `{column}` not found in the dataset.")

    result = df.copy()
    series = df[column]
    if method == "equal width":
        binned = pd.cut(series, bins=n_bins, labels=labels, include_lowest=True)
    else:
        binned = pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")

    result[f"{column}_binned"] = binned
    return result, {
        "column": f"{column}_binned",
        "n_bins": n_bins,
        "method": method,
    }


def extract_datetime_features(df, column, parts):
    """Extract calendar parts from a datetime column into new columns.

    Args:
        df: Input DataFrame.
        column: A datetime or date-parseable column.
        parts: Subset of ``["year", "month", "day", "weekday"]``. Weekday is
            Monday=0 through Sunday=6.

    Returns:
        ``(result, info)``.

    Raises:
        ValueError: If the column cannot be parsed as dates.
    """
    if column not in df.columns:
        raise ValueError(f"Column `{column}` not found in the dataset.")

    result = df.copy()
    parsed = pd.to_datetime(df[column], errors="coerce")
    if parsed.isna().all():
        raise ValueError(
            f"`{column}` could not be parsed as a date. Pick a column that "
            "contains dates (e.g. YYYY-MM-DD)."
        )

    new_columns = []
    for part in parts:
        new_column = f"{column}_{part}"
        if part == "year":
            result[new_column] = parsed.dt.year
        elif part == "month":
            result[new_column] = parsed.dt.month
        elif part == "day":
            result[new_column] = parsed.dt.day
        elif part == "weekday":
            result[new_column] = parsed.dt.weekday
        new_columns.append(new_column)

    return result, {"columns": new_columns, "source": column}


def extract_text_features(df, column, parts):
    """Extract basic text statistics into new columns.

    Args:
        df: Input DataFrame.
        column: A text column.
        parts: Subset of ``["length", "word count"]``.

    Returns:
        ``(result, info)``.
    """
    if column not in df.columns:
        raise ValueError(f"Column `{column}` not found in the dataset.")

    result = df.copy()
    text = df[column].astype(str)
    new_columns = []

    if "length" in parts:
        new_column = f"{column}_length"
        result[new_column] = text.str.len()
        new_columns.append(new_column)
    if "word count" in parts:
        new_column = f"{column}_word_count"
        result[new_column] = text.str.split().str.len()
        new_columns.append(new_column)

    return result, {"columns": new_columns, "source": column}


def create_interaction(df, col_a, col_b):
    """Add a product interaction feature ``<col_a>_x_<col_b>``."""
    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError("Both columns must exist in the dataset.")

    result = df.copy()
    column = f"{col_a}_x_{col_b}"
    result[column] = df[col_a] * df[col_b]
    return result, {"column": column}


def create_polynomial_features(df, columns, degree=2):
    """Add polynomial and interaction terms with sklearn PolynomialFeatures.

    Only terms that are not already present as original columns are added, so
    the original features are never duplicated.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to expand.
        degree: Polynomial degree (2 or 3).

    Returns:
        ``(result, info)`` with the list of added columns.

    Raises:
        ValueError: If any selected column contains missing values.
    """
    columns = _validate_columns(df, columns)
    result = df.copy()

    if not columns:
        return result, {"note": "No columns selected."}

    missing = [c for c in columns if df[c].isna().any()]
    if missing:
        listed = ", ".join(f"`{c}`" for c in missing)
        raise ValueError(
            "Cannot compute polynomial features for columns with missing "
            f"values: {listed}. Handle missing values first."
        )

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    matrix = poly.fit_transform(df[columns])
    names = poly.get_feature_names_out(columns)

    added = []
    for name, values in zip(names, matrix.T):
        if name in result.columns:
            continue
        clean = name.replace(" ", "_x_").replace("^", "_pow")
        result[clean] = values
        added.append(clean)

    return result, {"columns": added, "degree": degree, "source": columns}


def select_by_variance(df, columns, threshold=0.0):
    """Drop selected numeric columns whose variance is below a threshold.

    A threshold of ``0`` removes only constant columns (no information).

    Returns:
        ``(result, info)`` with the list of dropped columns.

    Raises:
        ValueError: If any selected column contains missing values.
    """
    columns = _validate_columns(df, columns)
    result = df.copy()

    if not columns:
        return result, {"note": "No columns selected."}

    subset = result[columns]
    if subset.isna().any().any():
        raise ValueError(
            "Cannot compute variance with missing values. Handle missing "
            "values before running feature selection."
        )

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(subset)
    keep = subset.columns[selector.get_support()]
    dropped = [c for c in columns if c not in keep]
    result = result.drop(columns=dropped)
    return result, {
        "dropped_columns": dropped,
        "threshold": threshold,
        "kept_columns": keep.tolist(),
    }


def select_by_correlation(df, target, threshold=0.2):
    """Keep numeric features with a strong correlation to a numeric target.

    Features whose absolute Pearson correlation with the target is below the
    threshold are dropped.

    Returns:
        ``(result, info)`` including a ``correlations`` Series.

    Raises:
        ValueError: If the target is missing or not numeric.
    """
    if target not in df.columns:
        raise ValueError(f"Target column `{target}` not found in the dataset.")

    result = df.copy()
    target_series = df[target]
    if not pd.api.types.is_numeric_dtype(target_series):
        raise ValueError(
            "Correlation-based selection needs a numeric target column. Pick "
            "a numeric column to predict."
        )

    numeric = result.select_dtypes(include=["number"]).columns.tolist()
    features = [c for c in numeric if c != target]
    if not features:
        return result, {"note": "No numeric features to evaluate."}

    correlations = result[features].corrwith(target_series).dropna()
    keep = correlations[correlations.abs() >= threshold].index.tolist()
    dropped = [c for c in features if c not in keep]
    result = result.drop(columns=dropped)
    return result, {
        "dropped_columns": dropped,
        "kept_columns": keep,
        "threshold": threshold,
        "correlations": correlations,
    }


def feature_importance(model, feature_names):
    """Return a sorted DataFrame of feature importances for a fitted model.

    Supports models exposing ``feature_importances_`` (tree ensembles) or a
    single ``coef_`` vector (linear models, using absolute coefficients).

    Args:
        model: A fitted sklearn estimator.
        feature_names: List of feature names matching the model's input.

    Returns:
        A DataFrame with ``Feature`` and ``Importance`` columns sorted by
        importance descending, or ``None`` when the model cannot be
        interpreted.
    """
    if feature_names is None or len(feature_names) == 0:
        return None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).ravel()
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        if coef.size != len(feature_names):
            return None
        values = np.abs(coef)
    else:
        return None

    if values.size != len(feature_names):
        return None

    frame = pd.DataFrame({"Feature": feature_names, "Importance": values})
    return frame.sort_values("Importance", ascending=False).reset_index(drop=True)


# Operation dispatcher --------------------------------------------------------
#
# The UI records each user action as an "operation" dict
# ``{"key", "label", "code", "params"}`` and replays the whole chain from the
# original DataFrame. Undo = drop the last operation; reset = clear the list.


def apply_feature_op(df, op):
    """Apply a single operation dict to a DataFrame.

    Args:
        df: DataFrame to transform.
        op: Operation dict with a ``key`` and ``params``.

    Returns:
        The transformed DataFrame.
    """
    key = op["key"]
    params = op["params"]

    if key == "numeric":
        result, _ = create_numeric_feature(
            df,
            params["col_a"],
            params["col_b"],
            params["operation"],
            name=params.get("name"),
        )
    elif key == "math":
        result, _ = apply_math_transform(df, params["columns"], params["method"])
    elif key == "bin":
        result, _ = bin_numeric(
            df, params["column"], params["n_bins"], params["method"]
        )
    elif key == "datetime":
        result, _ = extract_datetime_features(df, params["column"], params["parts"])
    elif key == "text":
        result, _ = extract_text_features(df, params["column"], params["parts"])
    elif key == "interaction":
        result, _ = create_interaction(df, params["col_a"], params["col_b"])
    elif key == "polynomial":
        result, _ = create_polynomial_features(
            df, params["columns"], params["degree"]
        )
    elif key == "variance":
        result, _ = select_by_variance(
            df, params["columns"], params["threshold"]
        )
    elif key == "correlation":
        result, _ = select_by_correlation(
            df, params["target"], params["threshold"]
        )
    else:
        raise ValueError(f"Unknown feature operation: {key}")

    return result


def operation_effect(before, after, op):
    """Describe what an operation changed between two DataFrames.

    Returns a human-readable summary such as ``Added 3 columns: ...``.
    """
    added = [c for c in after.columns if c not in before.columns]
    removed = [c for c in before.columns if c not in after.columns]

    parts = []
    if added:
        parts.append(f"added {len(added)} column(s): {', '.join(f'`{c}`' for c in added)}")
    if removed:
        parts.append(f"removed {len(removed)} column(s): {', '.join(f'`{c}`' for c in removed)}")
    if not parts:
        parts.append("no column changes (row values may differ)")
    return "; ".join(parts)


# Python code generation ------------------------------------------------------


def numeric_feature_code(col_a, col_b, operation="sum", name=None):
    """Return a Python snippet creating a numeric feature."""
    operator = {"sum": "+", "difference": "-", "product": "*", "ratio": "/"}
    default_name = {
        "sum": f"{col_a}_plus_{col_b}",
        "difference": f"{col_a}_minus_{col_b}",
        "product": f"{col_a}_times_{col_b}",
        "ratio": f"{col_a}_over_{col_b}",
    }[operation]
    column = name or default_name
    return (
        f"# Create a numeric feature: {operation} of {col_a} and {col_b}\n"
        f"df[{column!r}] = df[{col_a!r}] {operator[operation]} df[{col_b!r}]"
    )


def math_transform_code(columns, method="log"):
    """Return a Python snippet applying a math transform."""
    quoted = ", ".join(repr(c) for c in columns)
    suffix = MATH_SUFFIXES[method]
    if method == "square":
        return (
            "# Square selected columns\n"
            f"for col in [{quoted}]:\n"
            f"    df[col + {suffix!r}] = df[col] ** 2"
        )
    if method == "log":
        return (
            "# Apply log transform (requires positive values)\n"
            f"import numpy as np\n"
            f"for col in [{quoted}]:\n"
            f"    df[col + {suffix!r}] = np.log(df[col])"
        )
    return (
        "# Apply sqrt transform (requires non-negative values)\n"
        f"import numpy as np\n"
        f"for col in [{quoted}]:\n"
        f"    df[col + {suffix!r}] = np.sqrt(df[col])"
    )


def binning_code(column, n_bins=4, method="equal width"):
    """Return a Python snippet binning a numeric column."""
    func = "pd.cut" if method == "equal width" else "pd.qcut"
    extra = "" if method == "equal width" else ", duplicates='drop'"
    return (
        f"# Bin {column} into {n_bins} {method} bins\n"
        f"df[{column + '_binned'!r}] = {func}(df[{column!r}], "
        f"bins={n_bins}{extra})"
    )


def datetime_code(column, parts):
    """Return a Python snippet extracting date/time features."""
    lines = [
        "# Extract calendar parts from a date column",
        f"dates = pd.to_datetime(df[{column!r}], errors='coerce')",
    ]
    for part in parts:
        attr = "weekday" if part == "weekday" else part
        lines.append(f"df[{column + '_' + part!r}] = dates.dt.{attr}")
    return "\n".join(lines)


def text_code(column, parts):
    """Return a Python snippet extracting text features."""
    lines = [f"# Extract text features from {column}", f"text = df[{column!r}].astype(str)"]
    if "length" in parts:
        lines.append(f"df[{column + '_length'!r}] = text.str.len()")
    if "word count" in parts:
        lines.append(f"df[{column + '_word_count'!r}] = text.str.split().str.len()")
    return "\n".join(lines)


def interaction_code(col_a, col_b):
    """Return a Python snippet creating an interaction feature."""
    return (
        "# Create an interaction (product) feature\n"
        f"df[{col_a + '_x_' + col_b!r}] = df[{col_a!r}] * df[{col_b!r}]"
    )


def polynomial_code(columns, degree=2):
    """Return a Python snippet creating polynomial features."""
    quoted = ", ".join(repr(c) for c in columns)
    return (
        "# Create polynomial and interaction terms\n"
        "from sklearn.preprocessing import PolynomialFeatures\n"
        f"poly = PolynomialFeatures(degree={degree}, include_bias=False)\n"
        f"poly.fit(df[[{quoted}]])\n"
        f"matrix = poly.transform(df[[{quoted}]])\n"
        f"for name, col in zip(poly.get_feature_names_out([{quoted}]), matrix.T):\n"
        "    if name in df.columns:\n"
        "        continue\n"
        "    df[name.replace(' ', '_x_').replace('^', '_pow')] = col"
    )


def variance_code(columns, threshold=0.0):
    """Return a Python snippet selecting features by variance."""
    quoted = ", ".join(repr(c) for c in columns)
    return (
        "# Keep features whose variance is above the threshold\n"
        "from sklearn.feature_selection import VarianceThreshold\n"
        f"selector = VarianceThreshold(threshold={threshold})\n"
        f"selector.fit(df[[{quoted}]])\n"
        f"keep = [c for c, flag in zip([{quoted}], selector.get_support()) if flag]\n"
        "df = df[keep]"
    )


def correlation_code(target, threshold=0.2):
    """Return a Python snippet selecting features by target correlation."""
    return (
        "# Keep numeric features with |correlation| >= threshold to the target\n"
        f"target = {target!r}\n"
        "numeric = df.select_dtypes(include='number').columns\n"
        "features = [c for c in numeric if c != target]\n"
        "correlations = df[features].corrwith(df[target]).abs()\n"
        f"keep = correlations[correlations >= {threshold}].index\n"
        "df = df[keep.tolist() + [target]]"
    )
