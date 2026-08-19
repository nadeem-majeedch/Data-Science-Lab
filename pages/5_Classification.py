"""Classification module.

This page lets students train and evaluate a classifier on the active dataset:

1. pick a categorical (low-cardinality) target column
2. choose numeric and categorical features (categorical columns are one-hot
   encoded automatically inside a leak-free sklearn Pipeline)
3. configure the train/test split
4. select one of seven algorithms and tune its parameters
5. train, then inspect metrics, the classification report, the confusion
   matrix, feature importances (where supported), make predictions on sample
   rows, and copy the equivalent Python code.

The engineered dataset from the Feature Engineering page (if any operations
were applied) is used automatically, so models see the enriched features.
"""

import pandas as pd
import plotly.express as px
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
from utils.models import (
    MODEL_KEYS,
    build_classifier,
    classifier_constructor_code,
    get_model,
)
from utils.model_training import (
    MAX_CLASSES,
    predict_sample,
    train_classifier,
    training_code,
    validate_classification_target,
)
from utils.session import (
    get_current_dataset,
    get_current_dataset_name,
    get_feature_ops,
    set_trained_model,
)

_MODULE = get_module("Classification")

RESULTS_KEY = "classification_results"

MAX_MANUAL_FEATURES = 8


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
    """Pick a sensible default target: a categorical column with 2-20 classes."""
    for column in df.columns:
        series = df[column]
        dtype = str(series.dtype)
        if dtype in ("object", "str", "category", "string"):
            if 2 <= series.nunique(dropna=True) <= MAX_CLASSES:
                return column
    return df.columns[0] if len(df.columns) else None


def render_dataset_banner(df, name: str | None) -> None:
    """Show a compact banner describing the dataset being modeled."""
    st.caption(
        f"Building a classifier on: **{name or 'current dataset'}** "
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
        "The **target** is the column your classifier is trying to predict. "
        "For classification it must contain a small number of discrete "
        "classes, for example ``yes``/``no`` or ``A``/``B``/``C``. A numeric "
        "column with many unique values (like a price) is a *regression* "
        "target - use the Regression module for those.",
    )

    target = st.selectbox(
        "Target column (what you want to predict)",
        list(working.columns),
        key="clf_target",
        index=working.columns.get_loc(default_target(working))
        if default_target(working) in working.columns
        else 0,
    )

    try:
        info = validate_classification_target(working, target)
    except ValueError as exc:
        st.error(str(exc))
        return target, None

    count_col, info_col = st.columns([1, 2])
    count_col.metric("Classes", info["n_classes"])
    info_col.caption(
        f"Targets: {', '.join(info['classes'][:6])}"
        f"{' ...' if info['n_classes'] > 6 else ''}  |  "
        f"{info['missing']} missing target value(s)"
    )
    st.dataframe(
        info["counts"].rename("Count").to_frame(),
        width="stretch",
    )
    return target, info


def render_feature_section(working, target) -> tuple[list[str], list[str]]:
    """Render feature selection and return ``(numeric_features, categorical_features)``."""
    st.subheader("2. Choose the features")
    render_education(
        "Features and automatic preprocessing",
        "Features are the columns the model reads to make a prediction. "
        "Numeric features are **median-imputed** (missing values filled) and "
        "**standardized** (mean 0, standard deviation 1). Text/categorical "
        "features are **mode-imputed** and **one-hot encoded** - each category "
        "becomes its own 0/1 column. This all happens inside a "
        "``ColumnTransformer`` that is fitted on the training set only, which "
        "prevents data leakage.\n\n"
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
        c for c in categorical if working[c].nunique(dropna=True) <= 50
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
        key="clf_features_num",
    )
    selected_categorical = st.multiselect(
        "Categorical features",
        categorical,
        default=default_categorical,
        key="clf_features_cat",
    )

    features = selected_numeric + selected_categorical
    if not features:
        st.warning(
            "No features selected. Pick at least one numeric or categorical "
            "feature to train a classifier."
        )
    return selected_numeric, selected_categorical


def render_split_section() -> dict:
    """Render the train/test split configuration."""
    st.subheader("3. Configure the train/test split")
    render_education(
        "Why split the data?",
        "You evaluate a classifier on data it has **never seen**. The dataset "
        "is split into a training set (used to fit the model) and a test set "
        "(used only for the final evaluation). Fitting on the test set, or "
        "letting preprocessing 'learn' from it, would leak information and "
        "make the reported accuracy unrealistically high.\n\n"
        "- **test_size**: fraction of rows held out for evaluation.\n"
        "- **stratify**: preserve the class proportions in both splits. This "
        "is important when one class is rare.\n"
        "- **random_state**: fixes the split so results are reproducible.",
    )
    col_a, col_b, col_c = st.columns(3)
    test_size = col_a.slider(
        "Test set size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        key="clf_test_size",
    )
    random_state = col_b.number_input(
        "Random state",
        min_value=0,
        max_value=10_000,
        value=42,
        key="clf_random_state",
    )
    stratify = col_c.checkbox(
        "Stratify split",
        value=True,
        key="clf_stratify",
    )
    return {
        "test_size": test_size,
        "random_state": random_state,
        "stratify": stratify,
    }


def render_model_section() -> tuple[str, dict]:
    """Render algorithm selection, parameter widgets, and education."""
    st.subheader("4. Choose a classifier")

    model_key = st.selectbox("Algorithm", MODEL_KEYS, key="clf_model")
    spec = get_model(model_key)

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
        key = f"clf_param_{param['name']}"
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
    if st.button("Train model", type="primary", key="clf_train"):
        features = numeric_features + categorical_features
        if not features:
            st.error("Select at least one feature before training.")
            return
        try:
            validate_classification_target(working, target)
            estimator = build_classifier(model_key, params)
            results = train_classifier(
                working[features],
                working[target],
                estimator,
                test_size=split["test_size"],
                random_state=split["random_state"],
                stratify=split["stratify"],
            )
            results["config"] = {
                "model_key": model_key,
                "params": params,
                "target": target,
                "features": features,
                "test_size": split["test_size"],
                "random_state": split["random_state"],
                "stratify": split["stratify"],
            }
            st.session_state[RESULTS_KEY] = results
            set_trained_model(results["pipeline"], results["feature_names"])
            accuracy = results["metrics"]["accuracy"]
            st.success(f"Trained **{model_key}** - test accuracy {accuracy:.3f}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Training failed: {exc}")


def render_metrics(results) -> None:
    """Render the headline metric tiles."""
    metrics = results["metrics"]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col_b.metric("Precision", f"{metrics['precision']:.3f}")
    col_c.metric("Recall", f"{metrics['recall']:.3f}")
    col_d.metric("F1 score", f"{metrics['f1']:.3f}")
    average = metrics["average"]
    st.caption(
        f"Precision, recall and F1 use the **{average}** average "
        "(binary for two classes, macro otherwise). Train accuracy: "
        f"{results['train_score']:.3f} vs test accuracy: {results['test_score']:.3f}."
    )
    render_education(
        "Reading the metrics",
        "- **Accuracy**: fraction of test rows predicted correctly.\n"
        "- **Precision**: of the rows predicted as class X, how many really "
        "were X. High precision means few false alarms.\n"
        "- **Recall**: of the rows that really are X, how many were caught. "
        "High recall means few missed cases.\n"
        "- **F1**: the harmonic mean of precision and recall - a balance "
        "between the two.\n\n"
        "Accuracy alone can mislead on imbalanced data (a model that always "
        "predicts the majority class can look accurate). Precision, recall "
        "and F1 tell you more.",
    )


def render_report_and_confusion(results) -> None:
    """Render the classification report and the confusion matrix."""
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
    st.plotly_chart(fig, width="stretch", key="clf_confusion_plot")
    st.dataframe(matrix, width="stretch")
    render_education(
        "How to read a confusion matrix",
        "Each row is an **actual** class and each column a **predicted** "
        "class. Cells on the diagonal are correct predictions; cells off the "
        "diagonal are mistakes. Looking at *where* the mistakes land shows "
        "you which classes are confused with each other - for example whether "
        "a model frequently mistakes class B for class A.",
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
        f"Highest-ranked feature: `{importance.iloc[0]['Feature']}`. Values "
        "are model-internal scores, not conclusions about your data."
    )


def render_prediction(results, working, features) -> None:
    """Render the prediction playground: sample rows or custom inputs."""
    st.markdown("**Make a prediction**")
    mode = st.radio(
        "Prediction mode",
        ["Sample row from test set", "Enter your own values"],
        key="clf_pred_mode",
    )

    if mode == "Sample row from test set":
        render_sample_row_prediction(results)
    else:
        render_custom_prediction(results, working, features)


def render_sample_row_prediction(results) -> None:
    """Predict on a chosen row of the test set and compare with the truth."""
    X_test = results["X_test"]
    indices = list(X_test.index)
    selected = st.selectbox(
        "Test set row",
        indices,
        key="clf_pred_row",
        format_func=lambda index: f"Row {index}",
    )
    position = indices.index(selected)
    actual = results["y_test"].iloc[position]
    predicted = results["y_pred"][position]
    st.markdown(f"**Actual class:** `{actual}`  |  **Predicted class:** `{predicted}`")
    st.dataframe(X_test.loc[[selected]], width="stretch")
    if results["y_proba"] is not None:
        render_probabilities(results, position)


def render_custom_prediction(results, working, features) -> None:
    """Predict on values the student enters by hand."""
    numeric = [c for c in numeric_columns(working) if c in features]
    categorical = [
        c for c in working.columns if c in features and c not in numeric
    ]

    if len(features) > MAX_MANUAL_FEATURES:
        st.info(
            f"This dataset has {len(features)} features, which is too many to "
            "enter by hand. Use the *sample row* mode instead, or the Feature "
            "Engineering module to narrow the feature set."
        )
        return

    row = {}
    X_test = results["X_test"]
    for column in numeric:
        default = float(X_test[column].median()) if len(X_test) else 0.0
        row[column] = st.number_input(
            f"{column}",
            key=f"clf_in_num_{column}",
            value=default,
        )
    for column in categorical:
        options = sorted(working[column].dropna().astype(str).unique())
        row[column] = st.selectbox(
            f"{column}",
            options,
            key=f"clf_in_cat_{column}",
        )

    if st.button("Predict", key="clf_predict"):
        outcome = predict_sample(results["pipeline"], row)
        st.success(f"Predicted class: `{outcome['prediction']}`")
        if outcome["probabilities"] is not None:
            proba_frame = pd.DataFrame(
                {"Class": outcome["classes"], "Probability": outcome["probabilities"]}
            )
            st.markdown("**Predicted probabilities**")
            st.dataframe(proba_frame, width="stretch")


def render_probabilities(results, position: int) -> None:
    """Show the predicted class probabilities for one test row."""
    proba = results["y_proba"][position]
    frame = pd.DataFrame({"Class": results["classes"], "Probability": proba})
    st.markdown("**Predicted probabilities**")
    st.dataframe(frame, width="stretch")


def render_code(results, current_config) -> None:
    """Render the equivalent Python code for the last (or current) training."""
    st.markdown("**Show Python code**")
    config = (results or {}).get("config") or current_config
    estimator_code = classifier_constructor_code(config["model_key"], config["params"])
    script = training_code(
        config["model_key"],
        estimator_code,
        config["target"],
        config["features"],
        config["test_size"],
        config["random_state"],
        config["stratify"],
    )
    st.code(script, language="python")


def render_results_section(results) -> None:
    """Render every result block after a successful training."""
    st.markdown("---")
    st.subheader("5. Results")
    render_metrics(results)
    render_report_and_confusion(results)
    render_feature_importance(results)
    st.markdown("---")
    render_prediction(results, results["X_test"], results["config"]["features"])
    st.markdown("---")
    render_code(results, results["config"])


def main() -> None:
    """Assemble the Classification page."""
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
    elif results is None:
        current_config = {
            "model_key": model_key,
            "params": params,
            "target": target,
            "features": numeric_features + categorical_features,
            "test_size": split["test_size"],
            "random_state": split["random_state"],
            "stratify": split["stratify"],
        }
        st.markdown("---")
        st.subheader("5. Results")
        st.info("Configure the settings above and click **Train model** to see results.")
        render_code(None, current_config)

    render_sidebar_footer()


main()
