"""Regression module.

This page lets students train and evaluate a regressor on the active dataset:

1. pick a continuous numeric target column
2. choose numeric and categorical features (categorical columns are one-hot
   encoded automatically inside a leak-free sklearn Pipeline)
3. configure the train/test split
4. select one of seven regressors and tune its parameters
5. train, then inspect MAE/MSE/RMSE/R2, an actual-vs-predicted scatter plot, a
   residual plot, a prediction table, feature importances (where supported),
   and copy the equivalent Python code.

Like the Classification module, the engineered dataset from the Feature
Engineering page is used automatically when operations were applied.
"""

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
from utils.feature_engineering import apply_feature_op, feature_importance
from utils.regressors import (
    REGRESSOR_KEYS,
    build_regressor,
    get_regressor,
    regressor_constructor_code,
)
from utils.regression_training import (
    train_regressor,
    training_code,
    validate_regression_target,
)
from utils.session import (
    get_current_dataset,
    get_current_dataset_name,
    get_feature_ops,
    set_trained_model,
)

_MODULE = get_module("Regression")

RESULTS_KEY = "regression_results"

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


def default_target(df) -> str | None:
    """Pick a sensible default target: the first numeric column that varies."""
    numeric = numeric_columns(df)
    for column in numeric:
        if df[column].nunique(dropna=True) >= 2:
            return column
    return df.columns[0] if len(df.columns) else None


def render_dataset_banner(df, name: str | None) -> None:
    """Show a compact banner describing the dataset being modeled."""
    st.caption(
        f"Building a regressor on: **{name or 'current dataset'}** "
        f"({df.shape[0]:,} rows x {df.shape[1]} columns)"
    )


def render_overview(df) -> None:
    """Show the headline shape metrics for the working dataset."""
    categorical = df.select_dtypes(exclude=["number"]).columns.tolist()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", df.shape[0])
    col_b.metric("Columns", df.shape[1])
    col_c.metric("Numeric columns", len(numeric_columns(df)))
    col_d.metric("Non-numeric columns", len(categorical))


def render_target_section(working) -> tuple[str, dict | None]:
    """Render target selection and return ``(target, target_info)``.

    ``target_info`` is ``None`` when the selected target is not usable; the
    caller should stop before training in that case.
    """
    st.subheader("1. Choose the target column")
    render_education(
        "What is a target column?",
        "The **target** is the column your regressor is trying to predict. "
        "For regression it must be a *continuous numeric* outcome, such as a "
        "price, a temperature, or a test score. A column that holds discrete "
        "categories (like ``yes``/``no``) is a classification target - use "
        "the Classification module for those.",
    )

    target = st.selectbox(
        "Target column (what you want to predict)",
        list(working.columns),
        key="reg_target",
        index=working.columns.get_loc(default_target(working))
        if default_target(working) in working.columns
        else 0,
    )

    try:
        info = validate_regression_target(working, target)
    except ValueError as exc:
        st.error(str(exc))
        return target, None

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Values", info["n_values"])
    col_b.metric("Min", f"{info['min']:.3g}")
    col_c.metric("Max", f"{info['max']:.3g}")
    col_d.metric("Mean", f"{info['mean']:.3g}")
    st.caption(
        f"Distinct values: {info['n_unique']}  |  "
        f"Missing: {info['missing']}  |  Dtype: {info['dtype']}"
    )
    return target, info


def render_feature_section(working, target) -> tuple[list[str], list[str]]:
    """Render feature selection and return ``(numeric_features, categorical_features)``."""
    st.subheader("2. Choose the features")
    render_education(
        "Features and automatic preprocessing",
        "Features are the columns the model reads to make a prediction. "
        "Numeric features are **median-imputed** and **standardized** (mean 0, "
        "standard deviation 1). Text/categorical features are **mode-imputed** "
        "and **one-hot encoded** - each category becomes its own 0/1 column. "
        "This all happens inside a ``ColumnTransformer`` that is fitted on the "
        "training set only, which prevents data leakage.\n\n"
        "Raw date columns are not used automatically; use the Feature "
        "Engineering module to extract year/month/day parts first.",
    )

    numeric = [c for c in numeric_columns(working) if c != target]
    categorical = [
        c
        for c in working.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
        if c != target
    ]
    default_categorical = [
        c for c in categorical if working[c].nunique(dropna=True) <= HIGH_CARDINALITY_THRESHOLD
    ]
    if default_categorical != categorical:
        st.caption(
            "Some text columns (more than 50 unique values, e.g. ID or note "
            "columns) are not selected by default because one-hot encoding "
            "them would create a huge, uninformative feature space. You can "
            "still select them manually."
        )

    selected_numeric = st.multiselect(
        "Numeric features",
        numeric,
        default=numeric,
        key="reg_features_num",
    )
    selected_categorical = st.multiselect(
        "Categorical features",
        categorical,
        default=default_categorical,
        key="reg_features_cat",
    )

    features = selected_numeric + selected_categorical
    if not features:
        st.warning(
            "No features selected. Pick at least one numeric or categorical "
            "feature to train a regressor."
        )
    return selected_numeric, selected_categorical


def render_split_section() -> dict:
    """Render the train/test split configuration."""
    st.subheader("3. Configure the train/test split")
    render_education(
        "Why split the data?",
        "You evaluate a regressor on data it has **never seen**. The dataset "
        "is split into a training set (used to fit the model) and a test set "
        "(used only for the final evaluation). Fitting on the test set, or "
        "letting preprocessing 'learn' from it, would leak information and "
        "make the reported errors unrealistically good.\n\n"
        "- **test_size**: fraction of rows held out for evaluation.\n"
        "- **random_state**: fixes the split so results are reproducible.",
    )
    col_a, col_b = st.columns(2)
    test_size = col_a.slider(
        "Test set size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        key="reg_test_size",
    )
    random_state = col_b.number_input(
        "Random state",
        min_value=0,
        max_value=10_000,
        value=42,
        key="reg_random_state",
    )
    return {"test_size": test_size, "random_state": random_state}


def render_model_section() -> tuple[str, dict]:
    """Render algorithm selection, parameter widgets, and education."""
    st.subheader("4. Choose a regressor")

    model_key = st.selectbox("Algorithm", REGRESSOR_KEYS, key="reg_model")
    spec = get_regressor(model_key)

    st.markdown(f"**Why use {model_key}?**")
    st.markdown(spec["why"])
    render_education(
        "Key parameters",
        spec["key_parameters"],
    )
    render_education(
        "Advantages and limitations",
        f"**Advantages**\n\n{spec['advantages']}\n\n"
        f"**Limitations**\n\n{spec['limitations']}\n\n"
        f"**When to use it**\n\n{spec['when_to_use']}",
    )
    if spec["requires_scaling"]:
        st.caption(
            "This algorithm is sensitive to feature scale - the automatic "
            "preprocessing standardizes numeric features for you."
        )

    params = render_parameter_widgets(spec)
    return model_key, params


def render_parameter_widgets(spec) -> dict:
    """Render one widget per configured parameter and return chosen values."""
    values = {}
    for param in spec["params"]:
        key = f"reg_param_{param['name']}"
        help_text = param.get("help")
        if param["type"] == "int":
            values[param["name"]] = st.slider(
                param["label"],
                min_value=param["min"],
                max_value=param["max"],
                value=param["value"],
                step=param["step"],
                key=key,
                help=help_text,
            )
        elif param["type"] == "float":
            values[param["name"]] = st.number_input(
                param["label"],
                min_value=param["min"],
                max_value=param["max"],
                value=param["value"],
                step=param["step"],
                key=key,
                help=help_text,
            )
        elif param["type"] == "choice":
            options = [option[1] for option in param["options"]]
            index = options.index(param["value"])
            values[param["name"]] = st.selectbox(
                param["label"],
                options,
                index=index,
                key=key,
                help=help_text,
            )
    return values


def render_train_button(
    working, target, numeric_features, categorical_features, split, model_key, params
) -> None:
    """Render the train button; on click, train and store results."""
    if st.button("Train model", type="primary", key="reg_train"):
        features = numeric_features + categorical_features
        if not features:
            st.error("Select at least one feature before training.")
            return
        try:
            validate_regression_target(working, target)
            estimator = build_regressor(model_key, params)
            results = train_regressor(
                working[features],
                working[target],
                estimator,
                test_size=split["test_size"],
                random_state=split["random_state"],
            )
            results["config"] = {
                "model_key": model_key,
                "params": params,
                "target": target,
                "features": features,
                "test_size": split["test_size"],
                "random_state": split["random_state"],
            }
            st.session_state[RESULTS_KEY] = results
            set_trained_model(results["pipeline"], results["feature_names"])
            r2 = results["metrics"]["r2"]
            st.success(f"Trained **{model_key}** - test R2 {r2:.3f}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Training failed: {exc}")


def render_metrics(results) -> None:
    """Render the headline metric tiles."""
    metrics = results["metrics"]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("MAE", f"{metrics['mae']:.3f}")
    col_b.metric("MSE", f"{metrics['mse']:.3f}")
    col_c.metric("RMSE", f"{metrics['rmse']:.3f}")
    col_d.metric("R2", f"{metrics['r2']:.3f}")
    st.caption(
        f"Train R2: {results['train_score']:.3f} vs test R2: "
        f"{results['test_score']:.3f}."
    )
    render_education(
        "Reading the metrics",
        "- **MAE** (mean absolute error): the average absolute difference "
        "between predictions and actual values - easy to interpret, in the "
        "units of the target.\n"
        "- **MSE** (mean squared error): the average squared difference. It "
        "penalizes large errors much harder than small ones.\n"
        "- **RMSE**: the square root of MSE, back in the target's units. It "
        "balances interpretability with a harsher penalty on large errors.\n"
        "- **R2** (coefficient of determination): the share of the target's "
        "variance explained by the model. 1.0 is a perfect fit, 0.0 is no "
        "better than predicting the mean, and negative values mean the model "
        "is worse than the mean.\n\n"
        "Compare train and test R2: a large gap means the model is "
        "overfitting the training data.",
    )


def render_actual_vs_predicted(results) -> None:
    """Render the actual-vs-predicted scatter plot with an ideal line."""
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
    st.plotly_chart(fig, width="stretch", key="reg_actual_vs_pred_plot")
    render_education(
        "How to read this plot",
        "Each point is one test-set row: its x is the true value and its y is "
        "the prediction. Points hugging the dashed **ideal** line are accurate "
        "predictions. Points above the line are over-predictions; points below "
        "are under-predictions. A clear curve away from the line usually means "
        "the model is missing a non-linear pattern.",
    )


def render_residual_plot(results) -> None:
    """Render the residual scatter plot with a zero reference line."""
    st.markdown("**Residual plot**")
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
    st.plotly_chart(fig, width="stretch", key="reg_residual_plot")
    render_education(
        "How to read this plot",
        "The **residual** is the prediction error: ``actual - predicted``. A "
        "good model scatters residuals randomly around the zero line. Patterns "
        "to watch for:\n\n"
        "- A **funnel** shape (spreading out) signals non-constant variance "
        "(heteroscedasticity).\n"
        "- A **curve** means the model is systematically under- or "
        "over-predicting in parts of the range (missed non-linearity).\n"
        "- A few extreme points far from the line are outliers worth "
        "investigating.",
    )


def render_prediction_table(results) -> None:
    """Render the actual-vs-predicted-vs-residual table."""
    st.markdown("**Prediction table**")
    st.dataframe(results["predictions"], width="stretch")
    st.caption(
        f"Showing {len(results['predictions']):,} test-set rows. Residual is "
        "`Actual - Predicted`; rows closest to 0 are the most accurate."
    )


def render_feature_importance(results) -> None:
    """Render feature importances when the model supports them."""
    st.markdown("**Feature importance**")
    importance = feature_importance(results["pipeline"], results["feature_names"])
    if importance is None:
        st.info(
            "This model type does not expose interpretable feature importances "
            "(no ``feature_importances_`` or ``coef_`` for the chosen "
            "settings)."
        )
        return
    st.dataframe(importance, width="stretch")
    st.caption(
        f"Highest-ranked feature: `{importance.iloc[0]['Feature']}`. For "
        "linear models these are absolute coefficients; for tree models they "
        "are built-in importance scores."
    )


def render_code(results) -> None:
    """Render the equivalent Python code for the trained model."""
    st.markdown("**Show Python code**")
    config = results["config"]
    estimator_code = regressor_constructor_code(config["model_key"], config["params"])
    st.caption(
        f"Configured model parameters: `{estimator_code}`"
    )
    script = training_code(
        config["model_key"],
        estimator_code,
        config["target"],
        config["features"],
        config["test_size"],
        config["random_state"],
    )
    st.code(script, language="python")


def render_results_section(results) -> None:
    """Render every result block after a successful training."""
    st.markdown("---")
    st.subheader("5. Results")
    render_metrics(results)
    render_actual_vs_predicted(results)
    render_residual_plot(results)
    render_prediction_table(results)
    render_feature_importance(results)
    render_code(results)


def main() -> None:
    """Assemble the Regression page."""
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
    render_dataset_banner(working, name)
    if ops:
        st.caption(
            f"Using the feature-engineered dataset ({len(ops)} operation(s) "
            "applied in the Feature Engineering module)."
        )
    render_overview(working)

    st.markdown("---")
    target, target_info = render_target_section(working)
    st.markdown("---")
    numeric_features, categorical_features = render_feature_section(working, target)
    st.markdown("---")
    split = render_split_section()
    st.markdown("---")
    model_key, params = render_model_section()

    st.markdown("---")
    render_train_button(
        working, target, numeric_features, categorical_features, split, model_key, params
    )

    results = st.session_state.get(RESULTS_KEY)
    if target_info is not None and results is not None:
        render_results_section(results)
    else:
        st.markdown("---")
        st.subheader("5. Results")
        st.info("Configure the settings above and click **Train model** to see results.")

    render_sidebar_footer()


main()
