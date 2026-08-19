"""Feature Engineering module.

This page lets students enrich the active dataset by chaining *operations*:
creating numeric features, applying math transforms, binning, extracting
date/time and text features, creating interaction and polynomial terms, and
selecting features. Every operation shows its equivalent Python code, previews
the new columns, and can be undone or reset at any time.

When a modeling module has trained a model, this page also displays the
feature importances extracted from that model.
"""

import streamlit as st

from utils import (
    get_module,
    render_education,
    render_page_header,
    render_page_link,
    render_page_sidebar,
    render_sidebar_footer,
)
from utils.data_analysis import categorical_columns, numeric_columns
from utils.feature_engineering import (
    BINNING_METHODS,
    DATETIME_PARTS,
    MATH_TRANSFORMS,
    NUMERIC_OPERATIONS,
    TEXT_PARTS,
    apply_feature_op,
    binning_code,
    correlation_code,
    create_interaction,
    datetime_code,
    extract_datetime_features,
    extract_text_features,
    feature_importance,
    interaction_code,
    math_transform_code,
    numeric_feature_code,
    operation_effect,
    polynomial_code,
    text_code,
    variance_code,
)
from utils.session import (
    get_current_dataset,
    get_current_dataset_name,
    get_feature_ops,
    get_trained_model,
    get_trained_model_features,
    set_feature_ops,
)

_MODULE = get_module("Feature Engineering")

OPERATION_TYPES = [
    "Create numeric feature",
    "Mathematical transformation",
    "Bin numeric variable",
    "Extract date/time features",
    "Extract text features",
    "Create interaction feature",
    "Create polynomial features",
    "Select features by variance",
    "Select features by correlation",
]


def render_dataset_banner(df, name: str | None) -> None:
    """Show a compact banner describing the dataset being enriched."""
    st.caption(
        f"Engineering features for: **{name or 'current dataset'}** "
        f"({df.shape[0]:,} rows x {df.shape[1]} columns)"
    )


def render_quick_overview(df) -> None:
    """Show the headline shape metrics before any engineering."""
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", df.shape[0])
    col_b.metric("Columns", df.shape[1])
    col_c.metric("Numeric columns", len(numeric_columns(df)))
    col_d.metric("Non-numeric columns", len(categorical_columns(df)))


def fold_ops(df, ops):
    """Replay the operation chain from the original DataFrame.

    If a stored operation no longer makes sense for the current dataset (for
    example after switching datasets), replay stops gracefully and the rest of
    the chain is skipped.
    """
    working = df
    for op in ops:
        try:
            working = apply_feature_op(working, op)
        except (ValueError, KeyError) as exc:
            st.error(
                f"Operation '{op['label']}' could not be applied to the current "
                f"dataset: {exc}"
            )
            break
    return working


def render_builder(df) -> dict | None:
    """Render the operation controls and return a configured operation.

    Args:
        df: The current working DataFrame (used for column options and
            pre-validation).

    Returns:
        An operation dict, or ``None`` when the operation cannot be built.
    """
    op_type = st.selectbox("Operation type", OPERATION_TYPES, key="fe_op_type")

    if op_type == "Create numeric feature":
        numeric = numeric_columns(df)
        if len(numeric) < 2:
            st.info("Creating a numeric feature needs at least two numeric columns.")
            return None
        col_a = st.selectbox("Column A", numeric, key="fe_num_a")
        remaining = [c for c in numeric if c != col_a] or numeric
        col_b = st.selectbox("Column B", remaining, key="fe_num_b", index=0)
        operation = st.selectbox(
            "Operation", NUMERIC_OPERATIONS, key="fe_num_op"
        )
        name = st.text_input(
            "New column name (optional)", key="fe_num_name"
        ).strip() or None
        st.markdown(
            f"This adds a new column `{name or col_a + ' ' + operation + ' ' + col_b}` "
            f"combining `{col_a}` and `{col_b}`."
        )
        render_education(
            "Creating numeric features",
            "Combining existing columns into a new one can expose relationships "
            "a model cannot see on its own. For example, a BMI is a ratio of "
            "weight and height, and total price is a product of quantity and "
            "unit price. Ratios and products are especially useful when two "
            "columns together matter more than either alone.",
        )
        code = numeric_feature_code(col_a, col_b, operation, name)
        return {
            "key": "numeric",
            "label": f"Numeric feature: {col_a} {operation} {col_b}",
            "code": code,
            "params": {
                "col_a": col_a,
                "col_b": col_b,
                "operation": operation,
                "name": name,
            },
        }

    if op_type == "Mathematical transformation":
        numeric = numeric_columns(df)
        if not numeric:
            st.info("This dataset has no numeric columns to transform.")
            return None
        method = st.selectbox("Transformation", MATH_TRANSFORMS, key="fe_math_method")
        columns = st.multiselect("Columns to transform", numeric, key="fe_math_cols")
        if not columns:
            st.info("Select at least one numeric column.")
            return None
        suffix = {"log": "_log", "sqrt": "_sqrt", "square": "_squared"}[method]
        st.markdown(
            f"This adds one new column per selected column, e.g. "
            f"`{columns[0]}{suffix}`. The originals stay untouched."
        )
        render_education(
            "Mathematical transformations",
            "Skewed numeric features confuse distance-based models. Log and "
            "sqrt compress large values and pull distributions toward "
            "symmetry, while squaring emphasizes differences.\n\n"
            "- **log** needs strictly positive values.\n"
            "- **sqrt** needs non-negative values.\n"
            "- **square** is always defined.\n\n"
            "The new columns are added alongside the originals so you can "
            "compare and undo them.",
        )
        code = math_transform_code(columns, method)
        return {
            "key": "math",
            "label": f"{method.title()} transform: {', '.join(columns)}",
            "code": code,
            "params": {"columns": columns, "method": method},
        }

    if op_type == "Bin numeric variable":
        numeric = numeric_columns(df)
        if not numeric:
            st.info("This dataset has no numeric columns to bin.")
            return None
        column = st.selectbox("Column to bin", numeric, key="fe_bin_col")
        n_bins = st.slider("Number of bins", 2, 10, 4, key="fe_bin_bins")
        method = st.selectbox("Binning method", BINNING_METHODS, key="fe_bin_method")
        st.markdown(
            f"This adds a categorical column `{column}_binned` with "
            f"{n_bins} bins."
        )
        render_education(
            "Binning numerical variables",
            "Binning converts a numeric column into ordered categories (bins). "
            "It is useful when only the broad magnitude matters (low / medium / "
            "high) rather than the exact value.\n\n"
            "- **Equal width** splits the value range into bins of the same "
            "size; bins may end up with very different counts.\n"
            "- **Quantile** splits the data so every bin holds roughly the same "
            "number of rows.\n\n"
            "The result is categorical, so it must be encoded before modeling.",
        )
        code = binning_code(column, n_bins, method)
        return {
            "key": "bin",
            "label": f"Bin {column} ({method}, {n_bins} bins)",
            "code": code,
            "params": {"column": column, "n_bins": n_bins, "method": method},
        }

    if op_type == "Extract date/time features":
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        text_cols = categorical_columns(df)
        candidates = datetime_cols + text_cols
        if not candidates:
            st.info("This dataset has no date or text columns to extract from.")
            return None
        column = st.selectbox("Date column", candidates, key="fe_dt_col")
        parts = st.multiselect(
            "Parts to extract",
            DATETIME_PARTS,
            default=["year", "month"],
            key="fe_dt_parts",
        )
        if not parts:
            st.info("Select at least one part to extract.")
            return None
        st.markdown(
            f"This adds columns like `{column}_year` and `{column}_month`. "
            "Weekday is 0 for Monday through 6 for Sunday."
        )
        render_education(
            "Date/time feature extraction",
            "Dates are rich but not directly usable by models. Splitting a "
            "date into year, month, day, and weekday exposes recurring "
            "patterns such as seasonality or weekly cycles that models can "
            "learn from.",
        )
        code = datetime_code(column, parts)
        return {
            "key": "datetime",
            "label": f"Extract {', '.join(parts)} from {column}",
            "code": code,
            "params": {"column": column, "parts": parts},
        }

    if op_type == "Extract text features":
        text_cols = categorical_columns(df)
        if not text_cols:
            st.info("This dataset has no text columns to analyze.")
            return None
        column = st.selectbox("Text column", text_cols, key="fe_txt_col")
        parts = st.multiselect(
            "Features to extract",
            TEXT_PARTS,
            default=["length"],
            key="fe_txt_parts",
        )
        if not parts:
            st.info("Select at least one text feature.")
            return None
        render_education(
            "String/text basic features",
            "Raw text must become numbers before modeling. Simple numeric "
            "summaries of text are a cheap first step:\n\n"
            "- **length** counts characters in the string.\n"
            "- **word count** counts words separated by spaces.\n\n"
            "These can capture verbosity in reviews, response length in "
            "surveys, and similar signals.",
        )
        code = text_code(column, parts)
        return {
            "key": "text",
            "label": f"Text features from {column}: {', '.join(parts)}",
            "code": code,
            "params": {"column": column, "parts": parts},
        }

    if op_type == "Create interaction feature":
        numeric = numeric_columns(df)
        if len(numeric) < 2:
            st.info("An interaction feature needs at least two numeric columns.")
            return None
        col_a = st.selectbox("Column A", numeric, key="fe_int_a")
        remaining = [c for c in numeric if c != col_a] or numeric
        col_b = st.selectbox("Column B", remaining, key="fe_int_b", index=0)
        st.markdown(
            f"This adds a product column `{col_a}_x_{col_b}`. Unlike the "
            "numeric-feature builder, this is only ever a product."
        )
        render_education(
            "Interaction features",
            "An interaction feature multiplies two columns. It lets a model "
            "learn that the effect of one feature depends on the value of "
            "another - for example, that marketing spend only helps when the "
            "customer segment is right.",
        )
        code = interaction_code(col_a, col_b)
        return {
            "key": "interaction",
            "label": f"Interaction: {col_a} x {col_b}",
            "code": code,
            "params": {"col_a": col_a, "col_b": col_b},
        }

    if op_type == "Create polynomial features":
        numeric = numeric_columns(df)
        if not numeric:
            st.info("This dataset has no numeric columns for polynomials.")
            return None
        columns = st.multiselect(
            "Columns to expand",
            numeric,
            key="fe_poly_cols",
        )
        degree = st.slider("Polynomial degree", 2, 3, 2, key="fe_poly_degree")
        if not columns:
            st.info("Select at least one numeric column.")
            return None
        render_education(
            "Polynomial features",
            "Polynomial features add squared terms (``x^2``), cubes, and every "
            "pairwise interaction (``a*b``). They help linear models fit "
            "curved relationships. Watch out: the number of terms grows "
            "quickly, and the features become highly correlated with each "
            "other.",
        )
        code = polynomial_code(columns, degree)
        return {
            "key": "polynomial",
            "label": f"Polynomial features (degree {degree}): {', '.join(columns)}",
            "code": code,
            "params": {"columns": columns, "degree": degree},
        }

    if op_type == "Select features by variance":
        numeric = numeric_columns(df)
        if not numeric:
            st.info("This dataset has no numeric columns to select from.")
            return None
        columns = st.multiselect(
            "Columns to evaluate", numeric, key="fe_var_cols"
        )
        threshold = st.number_input(
            "Variance threshold (0 removes constant columns)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="fe_var_threshold",
        )
        if not columns:
            st.info("Select at least one numeric column.")
            return None
        render_education(
            "Variance threshold selection",
            "A feature with zero (or very low) variance contains almost no "
            "information - every row has the same value. ``VarianceThreshold`` "
            "drops such features. A threshold of 0 removes only constant "
            "columns; higher thresholds remove progressively flatter features. "
            "Scale your features first, because variance depends on the scale "
            "of the column.",
        )
        code = variance_code(columns, threshold)
        return {
            "key": "variance",
            "label": f"Select by variance (threshold {threshold})",
            "code": code,
            "params": {"columns": columns, "threshold": threshold},
        }

    # Select features by correlation
    target = st.selectbox(
        "Target column (numeric, what you want to predict)",
        list(df.columns),
        key="fe_corr_target",
    )
    threshold = st.slider(
        "Minimum absolute correlation",
        min_value=0.0,
        max_value=0.9,
        value=0.2,
        step=0.05,
        key="fe_corr_threshold",
    )
    st.markdown(
        f"Keeps numeric features whose absolute correlation with "
        f"`{target}` is at least {threshold}. The target itself is always kept."
    )
    render_education(
        "Correlation-based selection",
        "Features that barely move with the target are unlikely to help "
        "predict it. This selector computes the Pearson correlation between "
        "every numeric feature and the (numeric) target, then drops features "
        "whose absolute correlation falls below the threshold.\n\n"
        "Correlation only captures *linear* relationships, so a weak "
        "correlation does not always mean the feature is useless - and a "
        "strong one does not guarantee causality.",
    )
    code = correlation_code(target, threshold)
    return {
        "key": "correlation",
        "label": f"Select by correlation with {target} (>= {threshold})",
        "code": code,
        "params": {"target": target, "threshold": threshold},
    }


def render_workflow(df, ops) -> None:
    """Render the builder, action buttons, and operation history."""
    st.subheader("Add an operation")

    working = fold_ops(df, ops)
    op = render_builder(working)

    col_apply, col_undo, col_reset = st.columns([1, 1, 1])
    if col_apply.button("Apply operation", key="fe_apply", type="primary"):
        if op is None:
            st.warning("Configure the operation first.")
        else:
            try:
                apply_feature_op(working, op)
            except ValueError as exc:
                st.error(str(exc))
            else:
                ops.append(op)
                set_feature_ops(ops)
                st.success(f"Applied: {op['label']}")
    if col_undo.button("Undo last operation", key="fe_undo"):
        if ops:
            ops.pop()
            set_feature_ops(ops)
            st.info("Undid the last operation.")
        else:
            st.info("Nothing to undo yet.")
    if col_reset.button("Reset all operations", key="fe_reset"):
        set_feature_ops([])
        st.info("Reset all operations.")

    working = fold_ops(df, get_feature_ops())

    if working.shape[1] > df.shape[1]:
        added = [c for c in working.columns if c not in df.columns]
        st.caption(
            f"Working dataset now has **{working.shape[0]:,} rows x "
            f"{working.shape[1]} columns** (added {len(added)} column(s))."
        )

    st.markdown("---")
    render_history(df, ops)


def render_history(df, ops) -> None:
    """Render each applied operation with its code and effect."""
    st.subheader("Operation history")

    if not ops:
        st.info(
            "No operations applied yet. Build one above and click "
            "**Apply operation**."
        )
        return

    current = df
    for index, op in enumerate(ops, start=1):
        after = fold_ops(current, [op])
        effect = operation_effect(current, after, op)
        with st.expander(f"{index}. {op['label']}", expanded=index == len(ops)):
            st.markdown(effect)
            st.code(op["code"], language="python")
            added = [c for c in after.columns if c not in current.columns]
            if added:
                st.markdown("**Preview of the new columns**")
                st.dataframe(after[added].head(5), width="stretch")
        current = after


def render_feature_importance() -> None:
    """Display feature importances from a trained model, if one is available."""
    st.subheader("Feature importance")

    model = get_trained_model()
    features = get_trained_model_features()

    if model is None:
        st.info(
            "No trained model available yet. Once a modeling module trains a "
            "model, its feature importances will appear here so you can judge "
            "which engineered features actually matter."
        )
        render_education(
            "Feature importance",
            "Feature importance ranks how much each feature contributes to a "
            "trained model's predictions.\n\n"
            "- **Tree-based models** (random forests, gradient boosting) expose "
            "``feature_importances_``, computed from how often and how much a "
            "feature splits the data.\n"
            "- **Linear models** expose ``coef_``; the absolute coefficient is "
            "used here as a rough proxy for importance.\n\n"
            "Importance helps you drop uninformative engineered features and "
            "keep your model simple.",
        )
        return

    if not features:
        st.warning(
            "A model is available but no feature names were recorded, so "
            "importances cannot be labeled."
        )
        return

    importance = feature_importance(model, features)
    if importance is None:
        st.warning(
            "This model type does not expose interpretable feature "
            "importances (no ``feature_importances_`` or ``coef_``)."
        )
        return

    st.dataframe(importance, width="stretch")
    top = importance.iloc[0]["Feature"]
    st.caption(
        f"Highest-ranked feature: `{top}`. Values are model-internal scores, "
        "not conclusions about your data."
    )


def render_full_code(ops) -> None:
    """Render one combined, copy-paste-ready Python script."""
    st.subheader("Complete Python code")
    if not ops:
        st.info("Apply at least one operation to see the full code.")
        return
    parts = [
        "# Complete feature engineering script - copy into your own notebook",
        "import pandas as pd",
        "",
        "# Load your dataset",
        "# df = pd.read_csv('your_data.csv')",
        "",
    ]
    for op in ops:
        parts.append(op["code"])
        parts.append("")
    with st.container(border=True):
        st.code("\n".join(parts), language="python")


def render_download(df) -> None:
    """Render the download button for the enriched dataset."""
    st.subheader("Download engineered dataset")
    st.download_button(
        "Download enriched dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="engineered_data.csv",
        mime="text/csv",
        key="fe_download",
    )
    st.caption(
        f"The downloaded file contains {df.shape[0]:,} rows x {df.shape[1]} columns."
    )


def main() -> None:
    """Assemble the Feature Engineering page."""
    render_page_sidebar(_MODULE)
    render_page_header(_MODULE.title, _MODULE.subtitle, help_text=_MODULE.help_text)

    df = get_current_dataset()
    name = get_current_dataset_name()

    if df is None:
        st.info(
            "No dataset loaded yet. Open the **Dataset Explorer** in the "
            "sidebar to upload a dataset or load one of the sample datasets "
            "first."
        )
        render_page_link("pages/1_Dataset_Explorer.py", "Go to Dataset Explorer")
        render_sidebar_footer()
        return

    render_dataset_banner(df, name)
    render_quick_overview(df)

    st.markdown("---")
    ops = get_feature_ops()
    render_workflow(df, ops)

    working = fold_ops(df, get_feature_ops())

    st.markdown("---")
    render_feature_importance()
    render_full_code(ops)
    render_download(working)

    render_sidebar_footer()


main()
