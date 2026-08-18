"""Dataset analysis helpers used by the Dataset Explorer module.

Every function here is a pure function operating on a pandas DataFrame, with
no Streamlit or UI dependencies, which keeps them easy to unit-test.
"""

import pandas as pd


def format_bytes(size_bytes: int) -> str:
    """Format a byte count into a human-readable string."""
    size = float(size_bytes)
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def memory_usage_bytes(df: pd.DataFrame) -> int:
    """Return the total memory used by the DataFrame, including object data."""
    return int(df.memory_usage(deep=True).sum())


def dataframe_info(df: pd.DataFrame) -> dict:
    """Return high-level metadata about a DataFrame.

    Returns a dict with keys ``rows``, ``columns``, ``memory_bytes`` and
    ``dtypes`` (a column name -> dtype string mapping).
    """
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_bytes": memory_usage_bytes(df),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of numeric column names."""
    return df.select_dtypes(include=["number"]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of non-numeric column names."""
    return df.select_dtypes(exclude=["number"]).columns.tolist()


def head(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the first ``n`` rows of the DataFrame."""
    return df.head(n)


def tail(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the last ``n`` rows of the DataFrame."""
    return df.tail(n)


def random_sample(df: pd.DataFrame, n: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Return a random sample of ``n`` rows using a fixed seed by default."""
    return df.sample(n=min(n, len(df)), random_state=random_state)


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all numeric columns.

    Includes count, mean, std, min, quartiles, max and the missing count.
    Returns an empty DataFrame when there are no numeric columns.
    """
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()
    stats = numeric.describe().T.reset_index()
    stats = stats.rename(columns={"index": "Column"})
    stats["Missing"] = numeric.isna().sum().values
    return stats


def categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column summaries for all non-numeric columns.

    Includes data type, non-null count, number of unique values, the most
    frequent value, its frequency, and the missing count.
    """
    categorical = df.select_dtypes(exclude=["number"])
    rows = []
    for column in categorical.columns:
        series = df[column]
        counts = series.value_counts(dropna=True)
        rows.append(
            {
                "Column": column,
                "Data type": str(series.dtype),
                "Count": int(series.count()),
                "Unique": int(series.nunique()),
                "Top": counts.index[0] if len(counts) else None,
                "Frequency": int(counts.iloc[0]) if len(counts) else 0,
                "Missing": int(series.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing value counts and percentages."""
    rows = []
    total = len(df)
    for column in df.columns:
        missing = int(df[column].isna().sum())
        rows.append(
            {
                "Column": column,
                "Missing count": missing,
                "Missing %": round(missing / total * 100, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def duplicate_rows(df: pd.DataFrame) -> int:
    """Return the number of fully duplicated rows."""
    return int(df.duplicated().sum())


def unique_counts(df: pd.DataFrame) -> pd.Series:
    """Return the number of unique values per column."""
    return df.nunique()


def constant_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that contain a single value (or are entirely empty)."""
    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=True) <= 1
    ]


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a combined per-column data-quality summary.

    One row per column with data type, missing count and percentage, number
    of unique values, and a flag for constant (no-information) columns.
    """
    rows = []
    total = len(df)
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "Column": column,
                "Data type": str(series.dtype),
                "Missing count": missing,
                "Missing %": round(missing / total * 100, 2) if total else 0.0,
                "Unique values": int(series.nunique()),
                "Constant": bool(series.nunique(dropna=True) <= 1),
            }
        )
    return pd.DataFrame(rows)


def data_quality_score(df: pd.DataFrame) -> float:
    """Return the share of non-missing cells, between 0 and 1."""
    if df.shape[0] * df.shape[1] == 0:
        return 0.0
    return float(1 - df.isna().sum().sum() / (df.shape[0] * df.shape[1]))


# EDA-specific helpers --------------------------------------------------------

HIGH_CARDINALITY_THRESHOLD = 50


def top_correlations(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Return the strongest absolute Pearson correlations between numeric columns.

    Returns a DataFrame with ``Column A``, ``Column B`` and ``Correlation``,
    sorted by absolute correlation descending, excluding self-pairs. Only
    columns with at least two paired non-null values are considered.
    """
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["Column A", "Column B", "Correlation"])

    corr = numeric.corr(numeric_only=True)
    pairs = []
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            pairs.append({"Column A": col_a, "Column B": col_b, "Correlation": value})
    if not pairs:
        return pd.DataFrame(columns=["Column A", "Column B", "Correlation"])

    ranked = pd.DataFrame(pairs).reindex(
        columns=["Column A", "Column B", "Correlation"]
    )
    ranked["Abs"] = ranked["Correlation"].abs()
    ranked = ranked.sort_values("Abs", ascending=False).head(k)
    return ranked.drop(columns=["Abs"]).reset_index(drop=True)


def eda_summary(df: pd.DataFrame) -> dict:
    """Return a compact automatic EDA summary of the dataset.

    The returned dict contains high-level facts computed from the data. These
    are neutral observations for educational use, not conclusions about any
    real-world question.
    """
    missing_total = int(df.isna().sum().sum())
    total_cells = df.shape[0] * df.shape[1]
    high_cardinality = [
        column
        for column in df.columns
        if str(df[column].dtype) in ("object", "category")
        and df[column].nunique() > HIGH_CARDINALITY_THRESHOLD
    ]
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": missing_total,
        "missing_cells_pct": round(missing_total / total_cells * 100, 2)
        if total_cells
        else 0.0,
        "duplicate_rows": duplicate_rows(df),
        "numeric_columns": len(numeric_columns(df)),
        "categorical_columns": len(categorical_columns(df)),
        "constant_columns": constant_columns(df),
        "high_cardinality_columns": high_cardinality,
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }
