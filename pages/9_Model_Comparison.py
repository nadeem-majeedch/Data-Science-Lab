"""Model Comparison module.

This page trains several algorithms side by side on the **same** train/test
split with the **same** preprocessor, so differences in the metrics reflect the
algorithms rather than differences in how the data was handled.

For classification it ranks models on accuracy, precision, recall, F1 and AUC;
for regression on MAE, RMSE and R2. Results appear as a table, as grouped bar
charts, and (for binary classification) as overlaid ROC curves. The table can
be downloaded as CSV and the equivalent Python code is shown.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    get_module,
    render_education,
    render_page_header,
    render_page_link,
    render_page_sidebar,
    render_sidebar_footer,
)
from utils.data_analysis import numeric_columns
from utils.evaluation import HIGHEST_NOT_BEST, METRIC_GUIDANCE
from utils.feature_engineering import apply_feature_op
from utils.model_comparison import (
    best_model,
    compare_classifiers,
    compare_regressors,
    comparison_code,
)
from utils.model_training import MAX_CLASSES, validate_classification_target
from utils.models import MODEL_KEYS
from utils.regressors import REGRESSOR_KEYS
from utils.regression_training import validate_regression_target
from utils.session import get_current_dataset, get_current_dataset_name, get_feature_ops

_MODULE = get_module("Model Comparison")

RESULTS_KEY = "comparison_results"

HIGH_CARDINALITY_THRESHOLD = 50


def get_working_dataset(df):
    """Replay the feature-engineering operation chain onto the dataset."""
    working = df
    for op in get_feature_ops():
        try:
            working = apply_feature_op(working, op)
        except (ValueError, KeyError):
            break
    return working


def default_target(df, task: str) -> str | None:
    """Pick a sensible default target for the given task."""
    if task == "classification":
        for column in df.columns:
            series = df[column]
            dtype = str(series.dtype)
            if dtype in ("object", "str", "category", "string"):
                if 2 <= series.nunique(dropna=True) <= MAX_CLASSES:
                    return column
    else:
        for column in numeric_columns(df):
            if df[column].nunique(dropna=True) >= 2:
                return column
    return df.columns[0] if len(df.columns) else None


def render_target_section(working, task) -> tuple[str, dict | None]:
    """Render target selection and return ``(target, target_info)``."""
    st.subheader("1. Choose the target column")
    if task == "classification":
        render_education(
            "What makes a classification target?",
            "The **target** is the column your models are trying to predict. "
            "For classification it must contain a small number of discrete "
            "classes, such as ``yes``/``no``. A numeric column with many "
            "unique values is a regression target.",
        )
    else:
        render_education(
            "What makes a regression target?",
            "For regression the **target** must be a *continuous numeric* "
            "outcome, such as a price, a temperature, or a test score. "
            "Discrete categories belong in the Classification module.",
        )

    default = default_target(working, task)
    target = st.selectbox(
        "Target column (what you want to predict)",
        list(working.columns),
        key="cmp_target",
        index=working.columns.get_loc(default) if default in working.columns else 0,
    )

    validator = (
        validate_classification_target if task == "classification" else validate_regression_target
    )
    try:
        info = validator(working, target)
    except ValueError as exc:
        st.error(str(exc))
        return target, None
    return target, info


def render_feature_section(working, target) -> tuple[list[str], list[str]]:
    """Render feature selection and return ``(numeric, categorical)``."""
    st.subheader("2. Choose the features")
    render_education(
        "Features and automatic preprocessing",
        "Features are the columns the models read to make a prediction. "
        "Numeric features are **median-imputed** and **standardized**; text "
        "features are **mode-imputed** and **one-hot encoded**. Every model "
        "shares the *same* preprocessor, fitted on the training set only, so "
        "the comparison is fair.",
    )

    numeric = [c for c in numeric_columns(working) if c != target]
    categorical = [
        c
        for c in working.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if c != target
    ]
    default_categorical = [
        c for c in categorical if working[c].nunique(dropna=True) <= HIGH_CARDINALITY_THRESHOLD
    ]
    if default_categorical != categorical:
        st.caption(
            "Some text columns (more than 50 unique values, e.g. ID or note "
            "columns) are not selected by default because one-hot encoding "
            "them would create a huge, uninformative feature space."
        )

    selected_numeric = st.multiselect(
        "Numeric features",
        numeric,
        default=numeric,
        key="cmp_features_num",
    )
    selected_categorical = st.multiselect(
        "Categorical features",
        categorical,
        default=default_categorical,
        key="cmp_features_cat",
    )

    features = selected_numeric + selected_categorical
    if not features:
        st.warning("No features selected. Pick at least one feature to compare models.")
    return selected_numeric, selected_categorical


def render_split_section() -> dict:
    """Render the shared train/test split configuration."""
    st.subheader("3. Configure the train/test split")
    render_education(
        "Why one shared split?",
        "Every model trains on the **exact same** training rows and is "
        "evaluated on the **exact same** test rows. If each model used its own "
        "random split, one model could simply get luckier data and look "
        "better than it really is. A single split plus a fixed "
        "``random_state`` keeps the comparison fair and reproducible.",
    )
    col_a, col_b = st.columns(2)
    test_size = col_a.slider(
        "Test set size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        key="cmp_test_size",
    )
    random_state = col_b.number_input(
        "Random state",
        min_value=0,
        max_value=10_000,
        value=42,
        key="cmp_random_state",
    )
    return {"test_size": test_size, "random_state": random_state}


def render_model_section(task: str) -> list[str]:
    """Render the model multiselect and return the chosen keys."""
    st.subheader("4. Choose the models to compare")
    keys = MODEL_KEYS if task == "classification" else REGRESSOR_KEYS
    render_education(
        "How the models run",
        "Every selected algorithm uses its default parameters and is trained "
        "and evaluated on the identical data. Randomized algorithms (random "
        "forests, gradient boosting) are given the same random seed so the "
        "results are reproducible.",
    )
    chosen = st.multiselect(
        "Models",
        keys,
        default=keys,
        key="cmp_models",
    )
    if not chosen:
        st.warning("Choose at least one model to compare.")
    return chosen


def render_compare_button(
    working, task, target, numeric_features, categorical_features, split, model_keys
) -> None:
    """Run the comparison on click and store the results."""
    if st.button("Compare models", type="primary", key="cmp_compare"):
        features = numeric_features + categorical_features
        if not features:
            st.error("Select at least one feature before comparing.")
            return
        try:
            if task == "classification":
                validate_classification_target(working, target)
                results = compare_classifiers(
                    working[features],
                    working[target],
                    model_keys=model_keys,
                    test_size=split["test_size"],
                    random_state=split["random_state"],
                    stratify=True,
                )
            else:
                validate_regression_target(working, target)
                results = compare_regressors(
                    working[features],
                    working[target],
                    model_keys=model_keys,
                    test_size=split["test_size"],
                    random_state=split["random_state"],
                )
            results["config"].update(
                {"target": target, "features": features}
            )
            st.session_state[RESULTS_KEY] = results
            st.success(
                f"Compared {len(results['table'])} models. Top: "
                f"**{best_model(results['table'])}**."
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Comparison failed: {exc}")


def render_comparison_table(results) -> None:
    """Render the ranked comparison table."""
    table = results["table"].copy()
    numeric_cols = table.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        table[numeric_cols] = table[numeric_cols].round(4)
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        f"Top model: **{best_model(results['table'])}**. See the guidance "
        "below before treating that as the winner."
    )
    render_education(
        "Reading the table",
        "\n\n".join(
            METRIC_GUIDANCE[name]
            for name in (
                ["accuracy", "precision", "recall", "f1", "auc"]
                if results["config"]["task"] == "classification"
                else ["mae", "rmse", "r2"]
            )
        ),
    )


def render_classification_charts(results) -> None:
    """Render grouped metric bars and an ROC overlay (binary targets)."""
    st.markdown("**Metric comparison**")
    table = results["table"]
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    long = table.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    long = long[long["Score"].notna()]
    fig = px.bar(
        long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        labels={"Score": "Score (higher is better)", "Model": ""},
        title="Classification metrics across models",
    )
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, width="stretch", key="cmp_classification_bars")

    render_roc_overlay(results)


def render_roc_overlay(results) -> None:
    """Overlay one ROC curve per model for binary classification."""
    y_test = results["y_test"]
    details = results["details"]
    binary = all(
        len(details[key]["classes"]) == 2 and details[key]["y_proba"] is not None
        for key in results["config"]["model_keys"]
    )
    if not binary:
        return
    st.markdown("**ROC curves (one per model)**")
    fig = go.Figure()
    for key in results["config"]["model_keys"]:
        info = details[key]
        proba = info["y_proba"]
        from utils.evaluation import roc_curves

        curves = roc_curves(y_test, proba, info["classes"])
        for curve in curves:
            fig.add_trace(
                go.Scatter(
                    x=curve["fpr"],
                    y=curve["tpr"],
                    mode="lines",
                    name=f"{key} (AUC {curve['auc']:.3f})",
                )
            )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#888"},
            name="Random chance",
        )
    )
    fig.update_layout(
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    st.plotly_chart(fig, width="stretch", key="cmp_roc_overlay")


def render_regression_charts(results) -> None:
    """Render R2 (higher is better) and MAE/RMSE (lower is better) bars."""
    st.markdown("**Metric comparison**")
    table = results["table"]

    r2 = table[["Model", "R2"]]
    fig_r2 = px.bar(
        r2,
        x="Model",
        y="R2",
        labels={"R2": "R2 (higher is better)", "Model": ""},
        title="R2 across models",
        color_discrete_sequence=["#1f77b4"],
    )
    fig_r2.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig_r2, width="stretch", key="cmp_r2_bars")

    errors = table[["Model", "MAE", "RMSE"]].melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Error",
    )
    fig_err = px.bar(
        errors,
        x="Model",
        y="Error",
        color="Metric",
        barmode="group",
        labels={"Error": "Error (lower is better)", "Model": ""},
        title="MAE and RMSE across models",
    )
    fig_err.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig_err, width="stretch", key="cmp_error_bars")


def render_download(results) -> None:
    """Offer the comparison table as a CSV download."""
    table = results["table"].copy()
    numeric_cols = table.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        table[numeric_cols] = table[numeric_cols].round(4)
    st.download_button(
        "Download comparison (CSV)",
        table.to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
        key="cmp_download",
    )


def render_code(results) -> None:
    """Render the equivalent Python code for the comparison."""
    st.markdown("**Show Python code**")
    config = results["config"]
    st.code(comparison_code(config["task"], config), language="python")


def render_results_section(results) -> None:
    """Render every result block after a successful comparison."""
    st.markdown("---")
    st.subheader("5. Results")
    st.caption(
        f"Comparison of {len(results['table'])} model(s) on a shared "
        f"split (test size {results['config']['test_size']}, random state "
        f"{results['config']['random_state']})."
    )
    render_comparison_table(results)
    if results["config"]["task"] == "classification":
        render_classification_charts(results)
    else:
        render_regression_charts(results)
    render_download(results)
    st.markdown("---")
    render_code(results)
    st.markdown("---")
    render_education("Choosing the right metric", HIGHEST_NOT_BEST)


def main() -> None:
    """Assemble the Model Comparison page."""
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

    working = get_working_dataset(df)
    ops = get_feature_ops()
    st.caption(
        f"Comparing models on: **{name or 'current dataset'}** "
        f"({working.shape[0]:,} rows x {working.shape[1]} columns)"
    )
    if ops:
        st.caption(
            f"Using the feature-engineered dataset ({len(ops)} operation(s) "
            "applied in the Feature Engineering module)."
        )

    st.markdown("---")
    task = st.radio(
        "Task type",
        ["Classification", "Regression"],
        key="cmp_task",
        horizontal=True,
    )
    task = "classification" if task == "Classification" else "regression"

    st.markdown("---")
    target, target_info = render_target_section(working, task)
    st.markdown("---")
    numeric_features, categorical_features = render_feature_section(working, target)
    st.markdown("---")
    split = render_split_section()
    st.markdown("---")
    model_keys = render_model_section(task)

    st.markdown("---")
    render_compare_button(
        working, task, target, numeric_features, categorical_features, split, model_keys
    )

    results = st.session_state.get(RESULTS_KEY)
    if target_info is not None and results is not None:
        render_results_section(results)
    else:
        st.markdown("---")
        st.subheader("5. Results")
        st.info("Configure the settings above and click **Compare models** to see results.")

    render_sidebar_footer()


main()
