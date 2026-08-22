"""Model Evaluation module.

This page goes *deeper* on the model you just trained in the Classification or
Regression module. It reads the stored training results from session state and
adds the diagnostics the training pages do not show:

- classification: ROC curves and AUC, a per-metric explainer, and a
  cross-validation summary
- regression: residual statistics and histograms, and a cross-validation
  summary

Every metric comes with an educational explanation, the results can be
downloaded as CSV, and the equivalent Python code is shown.
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
from utils.evaluation import (
    HIGHEST_NOT_BEST,
    METRIC_GUIDANCE,
    cross_validate,
    evaluation_code,
    get_evaluation_source,
    residual_statistics,
    roc_auc_brief,
    roc_curves,
)
from utils.feature_engineering import apply_feature_op
from utils.models import build_classifier
from utils.regressors import build_regressor
from utils.session import get_current_dataset, get_current_dataset_name, get_feature_ops

_MODULE = get_module("Model Evaluation")

RESULTS_KEY = "evaluation_results"

TARGET_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1", "auc"],
    "regression": ["mae", "mse", "rmse", "r2"],
}

METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 score",
    "auc": "AUC",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "r2": "R2",
}


def get_working_dataset(df):
    """Replay the feature-engineering operation chain onto the dataset."""
    working = df
    for op in get_feature_ops():
        try:
            working = apply_feature_op(working, op)
        except (ValueError, KeyError):
            break
    return working


def _build_estimator(kind: str, model_key: str, params: dict, seed: int = 42):
    """Recreate the estimator used for training so CV can rebuild it per fold."""
    if kind == "classification":
        estimator = build_classifier(model_key, params)
    else:
        estimator = build_regressor(model_key, params)
    if "random_state" in estimator.get_params(deep=False):
        estimator.set_params(random_state=seed)
    return estimator


def render_no_model() -> None:
    """Render guidance when no model has been trained yet."""
    st.info(
        "No trained model found. Train a model first in the **Classification** "
        "or **Regression** module, then come back here to evaluate it in "
        "depth."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        render_page_link("pages/5_Classification.py", "Go to Classification")
    with col_b:
        render_page_link("pages/6_Regression.py", "Go to Regression")


def render_source_selector(kind: str) -> None:
    """Show which task is being evaluated, with a switch when both exist."""
    if kind == "classification":
        st.caption("Evaluating the model trained in the **Classification** module.")
    else:
        st.caption("Evaluating the model trained in the **Regression** module.")
    if "classification_results" in st.session_state and "regression_results" in st.session_state:
        st.caption("Both a classifier and a regressor were trained. Use the selector below to switch.")
        chosen = st.radio(
            "Model to evaluate",
            ["Classification", "Regression"],
            key="eval_task",
            horizontal=True,
        )
        return "classification" if chosen == "Classification" else "regression"
    return kind


def render_classification(results) -> None:
    """Render classification evaluation: metrics, ROC/AUC, report, CV."""
    config = results.get("config") or {}
    model_key = config.get("model_key", "Trained model")
    params = config.get("params", {})
    target = config.get("target", "target")
    st.subheader(f"Evaluating **{model_key}** (predicting `{target}`)")

    metrics = results["metrics"]
    auc_value = roc_auc_brief(results["y_test"], results["y_proba"])
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col_b.metric("Precision", f"{metrics['precision']:.3f}")
    col_c.metric("Recall", f"{metrics['recall']:.3f}")
    col_d.metric("F1 score", f"{metrics['f1']:.3f}")
    if auc_value is not None:
        st.metric("AUC", f"{auc_value:.3f}")
    st.caption(
        f"Test set: {len(results['y_test']):,} rows. Train accuracy "
        f"{results['train_score']:.3f} vs test accuracy "
        f"{results['test_score']:.3f}."
    )

    render_education(
        "Reading the metrics",
        "\n\n".join(
            METRIC_GUIDANCE[name] for name in TARGET_METRICS["classification"]
        ),
    )

    st.markdown("**Classification report**")
    st.dataframe(results["report"], width="stretch")

    st.markdown("**Confusion matrix**")
    matrix = results["confusion_matrix"]
    fig = px.imshow(
        matrix.to_numpy(),
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
    )
    st.plotly_chart(fig, width="stretch", key="eval_confusion_plot")
    st.dataframe(matrix, width="stretch")

    render_roc_section(results, model_key)
    render_cv_section("classification", model_key, params, config, results)


def render_roc_section(results, model_key: str) -> None:
    """Render the ROC curves and explain what they mean."""
    st.markdown("**ROC curve**")
    curves = roc_curves(results["y_test"], results["y_proba"], results["classes"])
    if not curves:
        st.info(
            "This model does not expose prediction probabilities, so an ROC "
            "curve cannot be drawn."
        )
        return
    fig = go.Figure()
    for curve in curves:
        fig.add_trace(
            go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode="lines",
                name=f"{curve['class']} (AUC {curve['auc']:.3f})",
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
        title=f"ROC curve - {model_key}",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    st.plotly_chart(fig, width="stretch", key="eval_roc_plot")
    render_education(
        "How to read an ROC curve",
        "The ROC curve plots the **true positive rate** against the **false "
        "positive rate** as the decision threshold moves from 'everything is "
        "positive' (top right) to 'nothing is positive' (bottom left).\n\n"
        "- A curve hugging the top-left corner separates the classes well.\n"
        "- The dashed diagonal is random guessing.\n"
        "- **AUC** is the area under the curve: 0.5 = random, 1.0 = perfect. "
        "It summarizes separation across all thresholds without choosing one.\n"
        "- For multi-class problems each class gets its own one-vs-rest "
        "curve; the AUC shown is the macro average.",
    )


def render_regression(results) -> None:
    """Render regression evaluation: metrics, residuals, CV."""
    config = results.get("config") or {}
    model_key = config.get("model_key", "Trained model")
    params = config.get("params", {})
    target = config.get("target", "target")
    st.subheader(f"Evaluating **{model_key}** (predicting `{target}`)")

    metrics = results["metrics"]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("MAE", f"{metrics['mae']:.3f}")
    col_b.metric("MSE", f"{metrics['mse']:.3f}")
    col_c.metric("RMSE", f"{metrics['rmse']:.3f}")
    col_d.metric("R2", f"{metrics['r2']:.3f}")
    st.caption(
        f"Test set: {len(results['y_test']):,} rows. Train R2 "
        f"{results['train_score']:.3f} vs test R2 "
        f"{results['test_score']:.3f}."
    )

    render_education(
        "Reading the metrics",
        "\n\n".join(METRIC_GUIDANCE[name] for name in TARGET_METRICS["regression"]),
    )

    render_actual_vs_predicted(results)
    render_residuals(results)
    render_cv_section("regression", model_key, params, config, results)


def render_actual_vs_predicted(results) -> None:
    """Render the actual-vs-predicted scatter with an ideal line."""
    st.markdown("**Actual vs predicted**")
    frame = results["predictions"]
    fig = px.scatter(
        frame,
        x="Actual",
        y="Predicted",
        labels={"Actual": "Actual value", "Predicted": "Predicted value"},
        title="Actual vs predicted on the test set",
        color_discrete_sequence=["#1f77b4"],
        opacity=0.8,
    )
    bounds = [
        float(min(frame["Actual"].min(), frame["Predicted"].min())),
        float(max(frame["Actual"].max(), frame["Predicted"].max())),
    ]
    fig.add_trace(
        go.Scatter(
            x=bounds,
            y=bounds,
            mode="lines",
            line={"dash": "dash", "color": "#d62728"},
            name="Ideal fit",
        )
    )
    st.plotly_chart(fig, width="stretch", key="eval_avp_plot")


def render_residuals(results) -> None:
    """Render residual diagnostics: plot, histogram, and statistics."""
    st.markdown("**Residual analysis**")
    frame = results["predictions"]
    fig = px.scatter(
        frame,
        x="Predicted",
        y="Residual",
        labels={"Predicted": "Predicted value", "Residual": "Residual (actual - predicted)"},
        title="Residuals against predicted values",
        color_discrete_sequence=["#2ca02c"],
        opacity=0.8,
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="Zero residual",
    )
    st.plotly_chart(fig, width="stretch", key="eval_residual_plot")

    hist = px.histogram(
        frame,
        x="Residual",
        nbins=min(30, max(10, len(frame) // 4)),
        title="Distribution of residuals",
        color_discrete_sequence=["#9467bd"],
        labels={"Residual": "Residual"},
    )
    st.plotly_chart(hist, width="stretch", key="eval_residual_hist")

    stats = residual_statistics(frame)
    stats_frame = pd.DataFrame(
        {
            "Statistic": ["Mean", "Std dev", "Min", "Median", "Max"],
            "Value": [
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['median']:.4f}",
                f"{stats['max']:.4f}",
            ],
        }
    )
    st.dataframe(stats_frame, width="stretch", hide_index=True)
    render_education(
        "How to read residuals",
        "The **residual** is the prediction error: ``actual - predicted``. "
        "Good residuals scatter randomly around zero with roughly constant "
        "spread.\n\n"
        "- A **mean near 0** means the model is not systematically biased.\n"
        "- A **funnel** (spreading) shape signals non-constant variance "
        "(heteroscedasticity).\n"
        "- A **curve** means a missed non-linear pattern.\n"
        "- A few extreme points far from zero are outliers worth "
        "investigating.",
    )


def render_cv_section(kind: str, model_key: str, params: dict, config: dict, results) -> None:
    """Run and render a cross-validation summary on the training data."""
    st.markdown("**Cross-validation**")
    render_education(
        "Why cross-validate?",
        "A single train/test split is a single roll of the dice. "
        "Cross-validation splits the training data into folds, trains the "
        "model several times - each time holding out a different fold - and "
        "averages the scores. The mean is a much more stable estimate of how "
        "the model will behave on new data, and the standard deviation shows "
        "how much it varies from fold to fold.",
    )
    n_folds = st.slider(
        "Number of folds",
        min_value=2,
        max_value=10,
        value=5,
        key="eval_cv_folds",
    )
    seed = config.get("random_state", 42)
    try:
        summary = cross_validate(
            lambda: _build_estimator(kind, model_key, params, seed),
            results["X_train"],
            results["y_train"],
            task=kind,
            n_folds=n_folds,
            random_state=seed,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    score_name = "Accuracy" if kind == "classification" else "R2"
    col_a.metric(f"CV {score_name} (mean)", f"{summary['mean']:.3f}")
    col_b.metric(f"CV {score_name} (std)", f"{summary['std']:.3f}")
    col_c.metric("Best fold", f"{summary['max']:.3f}")
    col_d.metric("Worst fold", f"{summary['min']:.3f}")

    fold_frame = pd.DataFrame(
        {"Fold": [f"Fold {i + 1}" for i in range(len(summary["scores"]))],
         score_name: summary["scores"]}
    )
    st.dataframe(fold_frame, width="stretch", hide_index=True)

    train_score = results["train_score"]
    gap = abs(train_score - summary["mean"])
    if kind == "classification" and train_score - summary["mean"] > 0.15:
        st.warning(
            f"The train accuracy ({train_score:.3f}) is much higher than the "
            f"cross-validated accuracy ({summary['mean']:.3f}). This gap "
            "usually means the model is **overfitting** the training data."
        )
    elif kind == "regression" and train_score - summary["mean"] > 0.15:
        st.warning(
            f"The train R2 ({train_score:.3f}) is much higher than the "
            f"cross-validated R2 ({summary['mean']:.3f}). This gap usually "
            "means the model is **overfitting** the training data."
        )
    st.caption(f"Gap between train and cross-validated score: {gap:.3f}.")


def render_download(results, kind: str) -> None:
    """Offer a CSV download of the most useful results table."""
    if kind == "classification":
        frame = results["report"]
        file_name = "evaluation_classification_report.csv"
        label = "Download classification report (CSV)"
    else:
        frame = results["predictions"]
        file_name = "evaluation_predictions.csv"
        label = "Download predictions (CSV)"
    st.download_button(
        label,
        frame.to_csv().encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key="eval_download",
    )


def render_code(results, kind: str) -> None:
    """Render the equivalent Python code for the evaluation steps."""
    st.markdown("**Show Python code**")
    config = results.get("config") or {}
    code_config = {
        "model_key": config.get("model_key", "Trained model"),
        "params": config.get("params", {}),
        "target": config.get("target", "target"),
        "features": config.get("features", []),
        "random_state": config.get("random_state", 42),
        "estimator_code": "",
    }
    st.code(evaluation_code(kind, code_config), language="python")


def main() -> None:
    """Assemble the Model Evaluation page."""
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
    st.caption(
        f"Evaluating on: **{name or 'current dataset'}** "
        f"({working.shape[0]:,} rows x {working.shape[1]} columns)"
    )

    kind, results = get_evaluation_source()
    if results is None:
        st.markdown("---")
        render_no_model()
        render_sidebar_footer()
        return

    st.markdown("---")
    kind = render_source_selector(kind)
    results = st.session_state[
        "classification_results" if kind == "classification" else "regression_results"
    ]

    st.markdown("---")
    st.subheader("Evaluation")
    if kind == "classification":
        render_classification(results)
    else:
        render_regression(results)

    render_download(results, kind)
    st.markdown("---")
    render_code(results, kind)

    st.markdown("---")
    render_education("Choosing the right metric", HIGHEST_NOT_BEST)

    render_sidebar_footer()


main()
