"""Data preprocessing helpers used by the Data Preprocessing module.

Every function here is a pure function operating on a pandas DataFrame with
no Streamlit or UI dependencies, which keeps them easy to unit-test.

The module exposes two complementary ways to work:

* immediate, inspectable transformations (``handle_missing``,
  ``encode_categorical``, ``scale_numeric``, ...) for the interactive UI
* a reusable sklearn :class:`~sklearn.compose.ColumnTransformer`
  (:func:`build_preprocessor`) that reproduces the configured steps so model
  training can apply them to new data

Data leakage note: transformations that *learn* from the data (imputers,
encoders, scalers) must be fitted on the **training** set only and then
applied unchanged to the **test** set. Fitting them on the whole dataset
first leaks test information into training and makes evaluation
over-optimistic. :func:`build_preprocessor` exists to do this correctly.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

import pandas as pd

MISSING_STRATEGIES = [
    "drop rows",
    "drop columns",
    "mean",
    "median",
    "mode",
    "constant",
]

ENCODING_METHODS = ["one-hot", "label"]

SCALING_METHODS = ["StandardScaler", "MinMaxScaler", "RobustScaler"]

SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}

IMPUTER_STRATEGIES = {
    "mean": "mean",
    "median": "median",
    "mode": "most_frequent",
    "constant": "constant",
}


def handle_missing(df, columns, strategy="median", fill_value=None):
    """Apply a missing-value strategy to the selected columns.

    Args:
        df: Input DataFrame.
        columns: Column names to operate on.
        strategy: One of ``MISSING_STRATEGIES``.
        fill_value: Constant used when ``strategy == "constant"``.

    Returns:
        ``(result, info)`` where ``info`` describes what changed.
    """
    columns = [column for column in columns if column in df.columns]
    result = df.copy()

    if not columns:
        return result, {"note": "No columns selected."}

    if strategy == "drop rows":
        before = len(result)
        result = result.dropna(subset=columns)
        return result, {"removed_rows": before - len(result)}

    if strategy == "drop columns":
        result = result.drop(columns=columns)
        return result, {"removed_columns": len(columns)}

    warnings = []
    for column in columns:
        series = result[column]
        if not series.isna().any():
            continue
        if strategy in ("mean", "median") and not pd.api.types.is_numeric_dtype(
            series
        ):
            warnings.append(
                f"`{column}` is not numeric, so `{strategy}` imputation was "
                "skipped. Use 'mode' or 'constant' for non-numeric columns."
            )
            continue
        if strategy == "mean":
            fill = series.mean()
        elif strategy == "median":
            fill = series.median()
        elif strategy == "mode":
            modes = series.mode(dropna=True)
            fill = modes.iloc[0] if len(modes) else None
        else:  # constant
            fill = fill_value
        if fill is None or (isinstance(fill, float) and pd.isna(fill)):
            warnings.append(
                f"`{column}` could not be imputed (every value is missing)."
            )
            continue
        result[column] = series.fillna(fill)

    info = {"strategy": strategy}
    if warnings:
        info["warnings"] = warnings
    return result, info


def remove_duplicates(df):
    """Drop fully duplicated rows.

    Returns:
        ``(result, info)`` with the number of removed rows.
    """
    before = len(df)
    result = df.drop_duplicates()
    return result, {"removed_rows": before - len(result), "kept_rows": len(result)}


def outlier_mask(df, columns, threshold=1.5):
    """Return a boolean DataFrame flagging outliers per column (IQR rule).

    A value is an outlier when it falls below ``Q1 - threshold * IQR`` or
    above ``Q3 + threshold * IQR``.
    """
    mask = pd.DataFrame(index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask[column] = (df[column] < lower) | (df[column] > upper)
    return mask


def outlier_counts(df, columns, threshold=1.5):
    """Return a per-column DataFrame of outlier counts (IQR rule)."""
    mask = outlier_mask(df, columns, threshold)
    rows = [
        {"Column": column, "Outliers": int(mask[column].sum())}
        for column in mask.columns
    ]
    return pd.DataFrame(rows)


def remove_outliers(df, columns, threshold=1.5):
    """Drop rows flagged as outliers in any selected column.

    Returns:
        ``(result, info)`` with the number of removed rows.
    """
    mask = outlier_mask(df, columns, threshold)
    any_outlier = mask.any(axis=1)
    result = df[~any_outlier]
    return result, {"removed_rows": int(any_outlier.sum()), "kept_rows": len(result)}


def encode_categorical(df, columns, method="one-hot"):
    """Encode categorical columns with one-hot or label encoding.

    Returns:
        ``(result, info)``. One-hot encoding adds indicator columns and drops
        the originals; label encoding replaces each column with integer
        category codes.
    """
    columns = [column for column in columns if column in df.columns]
    result = df.copy()

    if not columns:
        return result, {"note": "No categorical columns selected."}

    if method == "one-hot":
        result = pd.get_dummies(result, columns=columns, dtype=int)
    else:  # label
        for column in columns:
            result[column] = result[column].astype("category").cat.codes

    return result, {"method": method, "encoded_columns": columns}


def scale_numeric(df, columns, method="StandardScaler"):
    """Scale numeric columns with a fitted sklearn scaler.

    Returns:
        ``(result, info)`` where ``info`` carries the fitted scaler so the
        exact same transformation can be applied to the test set later.

    Raises:
        ValueError: If any selected column contains missing values; scaling
            before imputation would silently corrupt the result.
    """
    columns = [column for column in columns if column in df.columns]
    result = df.copy()

    if not columns:
        return result, {"note": "No numeric columns selected."}

    nan_columns = [column for column in columns if result[column].isna().any()]
    if nan_columns:
        listed = ", ".join(f"`{c}`" for c in nan_columns)
        raise ValueError(
            "Cannot scale columns that still contain missing values: "
            f"{listed}. Handle missing values first or exclude these columns."
        )

    scaler = SCALERS[method]()
    result[columns] = scaler.fit_transform(result[columns])
    return result, {"method": method, "scaler": scaler, "columns": columns}


def split_train_test(df, target_column, test_size=0.2, random_state=42, stratify=False):
    """Split the processed data into train/test feature and target sets.

    Returns:
        A dict with ``X_train``, ``X_test``, ``y_train``, ``y_test``.

    Raises:
        ValueError: If the target column is missing or stratification cannot
            be applied (every target class needs at least two members).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column `{target_column}` not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
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
            "The train/test split failed. When stratification is enabled "
            "every target class needs at least two members. Reduce the test "
            "size or disable stratification."
        ) from exc

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def build_preprocessor(
    impute_columns=None,
    impute_strategy="median",
    impute_fill_value=None,
    encode_columns=None,
    encode_method="one-hot",
    scale_columns=None,
    scale_method="StandardScaler",
    remainder="passthrough",
):
    """Build a reusable sklearn ColumnTransformer reproducing the config.

    This captures the *learned* part of preprocessing: imputation, encoding
    and scaling. Fit it on the training set only and then apply it to the
    test set unchanged to avoid data leakage::

        preprocessor.fit(X_train)                 # learn from TRAIN only
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)  # never fitted on TEST

    Args:
        impute_columns: Columns to impute missing values in.
        impute_strategy: Strategy passed to SimpleImputer.
        impute_fill_value: Fill value when ``impute_strategy == "constant"``.
        encode_columns: Categorical columns to encode.
        encode_method: ``"one-hot"`` or ``"label"``.
        scale_columns: Numeric columns to scale.
        scale_method: Name of a scaler class in ``SCALERS``.
        remainder: How to treat untouched columns (default ``"passthrough"``).

    Returns:
        A ``ColumnTransformer``, or ``None`` when no transformation was
        requested.
    """
    impute_columns = [c for c in (impute_columns or []) if c]
    encode_columns = [c for c in (encode_columns or []) if c]
    scale_columns = [c for c in (scale_columns or []) if c]

    transformers = []

    if impute_columns:
        imputer = SimpleImputer(strategy=IMPUTER_STRATEGIES[impute_strategy])
        if impute_strategy == "constant":
            imputer = SimpleImputer(strategy="constant", fill_value=impute_fill_value)
        transformers.append(("imputer", imputer, impute_columns))

    if encode_columns:
        if encode_method == "one-hot":
            encoder = OneHotEncoder(handle_unknown="ignore")
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        transformers.append(("encoder", encoder, encode_columns))

    if scale_columns:
        transformers.append(("scaler", SCALERS[scale_method](), scale_columns))

    if not transformers:
        return None

    return ColumnTransformer(transformers, remainder=remainder)


def compare_before_after(before, after):
    """Return a compact DataFrame contrasting key quality metrics."""
    numeric_before = before.select_dtypes(include=["number"]).columns.tolist()
    numeric_after = after.select_dtypes(include=["number"]).columns.tolist()
    return pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Missing cells",
                "Duplicate rows",
                "Numeric columns",
                "Non-numeric columns",
            ],
            "Before": [
                len(before),
                before.shape[1],
                int(before.isna().sum().sum()),
                int(before.duplicated().sum()),
                len(numeric_before),
                len(before.columns) - len(numeric_before),
            ],
            "After": [
                len(after),
                after.shape[1],
                int(after.isna().sum().sum()),
                int(after.duplicated().sum()),
                len(numeric_after),
                len(after.columns) - len(numeric_after),
            ],
        }
    )


# Python code generation ------------------------------------------------------
#
# The UI shows the equivalent Python for every operation so students can copy
# the exact snippet into their own notebooks.


def missing_values_code(columns, strategy="median", fill_value=None):
    """Return a Python snippet implementing the selected missing-value step."""
    quoted = ", ".join(repr(c) for c in columns)
    header = f"# Handle missing values ({strategy})"
    if not columns:
        return header + "\n# No columns selected - nothing to do."
    if strategy == "drop rows":
        code = f"df = df.dropna(subset=[{quoted}])"
    elif strategy == "drop columns":
        code = f"df = df.drop(columns=[{quoted}])"
    elif strategy == "mean":
        code = f"for col in [{quoted}]:\n    df[col] = df[col].fillna(df[col].mean())"
    elif strategy == "median":
        code = (
            f"for col in [{quoted}]:\n"
            "    df[col] = df[col].fillna(df[col].median())"
        )
    elif strategy == "mode":
        code = (
            f"for col in [{quoted}]:\n"
            "    df[col] = df[col].fillna(df[col].mode()[0])"
        )
    else:  # constant
        code = f"df[{quoted}] = df[{quoted}].fillna({fill_value!r})"
    return f"{header}\n{code}"


def duplicates_code():
    """Return a Python snippet that removes fully duplicated rows."""
    return "# Remove fully duplicated rows\ndf = df.drop_duplicates()"


def outliers_code(columns, threshold=1.5):
    """Return a Python snippet that flags and removes outliers (IQR)."""
    quoted = ", ".join(repr(c) for c in columns)
    if not columns:
        return "# Outlier handling\n# No columns selected - nothing to do."
    return (
        "# Flag and remove outliers using the IQR rule\n"
        f"cols = [{quoted}]\n"
        "Q1 = df[cols].quantile(0.25)\n"
        "Q3 = df[cols].quantile(0.75)\n"
        "IQR = Q3 - Q1\n"
        f"is_outlier = ((df[cols] < (Q1 - {threshold} * IQR)) | "
        f"(df[cols] > (Q3 + {threshold} * IQR))).any(axis=1)\n"
        "df = df[~is_outlier]"
    )


def encode_code(columns, method="one-hot"):
    """Return a Python snippet implementing the selected encoding step."""
    quoted = ", ".join(repr(c) for c in columns)
    if not columns:
        return "# Categorical encoding\n# No columns selected - nothing to do."
    if method == "one-hot":
        return (
            "# One-hot encode categorical columns\n"
            f"df = pd.get_dummies(df, columns=[{quoted}], dtype=int)"
        )
    return (
        "# Label encode categorical columns\n"
        f"for col in [{quoted}]:\n"
        "    df[col] = df[col].astype('category').cat.codes"
    )


def scale_code(columns, method="StandardScaler"):
    """Return a Python snippet implementing the selected scaling step."""
    quoted = ", ".join(repr(c) for c in columns)
    if not columns:
        return "# Feature scaling\n# No columns selected - nothing to do."
    return (
        f"# Scale numeric columns with {method}\n"
        f"from sklearn.preprocessing import {method}\n"
        f"scaler = {method}()\n"
        f"df[{quoted}] = scaler.fit_transform(df[{quoted}])"
    )


def split_code(target, test_size=0.2, random_state=42, stratify=False):
    """Return a Python snippet implementing the train/test split."""
    stratify_line = "    stratify=y," if stratify else ""
    return (
        "# Split into train and test sets\n"
        "from sklearn.model_selection import train_test_split\n"
        f"X = df.drop(columns={target!r})\n"
        f"y = df[{target!r}]\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        f"    X, y,\n"
        f"    test_size={test_size},\n"
        f"    random_state={random_state},\n"
        f"{stratify_line}\n"
        ")"
    )


def preprocessor_code(
    impute_columns=None,
    impute_strategy="median",
    encode_columns=None,
    encode_method="one-hot",
    scale_columns=None,
    scale_method="StandardScaler",
    remainder="passthrough",
):
    """Return a Python snippet that builds the reusable preprocessor."""
    parts = [
        "# Build a reusable preprocessor - fit on TRAIN only, then transform TEST",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.preprocessing import (",
        "    OneHotEncoder, OrdinalEncoder, MinMaxScaler, RobustScaler, StandardScaler,",
        ")",
        "transformers = []",
    ]

    impute_columns = [c for c in (impute_columns or []) if c]
    encode_columns = [c for c in (encode_columns or []) if c]
    scale_columns = [c for c in (scale_columns or []) if c]

    if impute_columns:
        quoted = ", ".join(repr(c) for c in impute_columns)
        parts.append(
            f"transformers.append(('imputer', "
            f"SimpleImputer(strategy={IMPUTER_STRATEGIES[impute_strategy]!r}), "
            f"[{quoted}]))"
        )
    if encode_columns:
        quoted = ", ".join(repr(c) for c in encode_columns)
        if encode_method == "one-hot":
            encoder = "OneHotEncoder(handle_unknown='ignore')"
        else:
            encoder = "OrdinalEncoder()"
        parts.append(f"transformers.append(('encoder', {encoder}, [{quoted}]))")
    if scale_columns:
        quoted = ", ".join(repr(c) for c in scale_columns)
        parts.append(
            f"transformers.append(('scaler', {scale_method}(), [{quoted}]))"
        )

    parts.extend(
        [
            f"preprocessor = ColumnTransformer(transformers, remainder={remainder!r})",
            "preprocessor.fit(X_train)  # learn from TRAINING data only",
            "X_train_t = preprocessor.transform(X_train)",
            "X_test_t = preprocessor.transform(X_test)  # TEST is transformed, not fitted",
        ]
    )
    return "\n".join(parts)
