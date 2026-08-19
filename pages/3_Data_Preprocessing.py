"""Data Preprocessing module.

This page turns the raw dataset selected in the Dataset Explorer into a clean,
analysis-ready DataFrame. Students follow a step-by-step cleaning workflow:

1. Handle missing values     2. Remove duplicates     3. Handle outliers
4. Encode categoricals       5. Scale numerics        6. Train/test split

Every step shows the exact Python code it uses so students can copy it into
their own notebooks, and the module ends by building a reusable sklearn
``ColumnTransformer`` that later modeling modules reuse.

Data leakage is explained and handled properly: learned transformations are
captured in the preprocessor, which is fitted on the *training* set only and
then applied unchanged to the test set.
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
from utils.preprocessing import (
    ENCODING_METHODS,
    MISSING_STRATEGIES,
    SCALING_METHODS,
    build_preprocessor,
    compare_before_after,
    duplicates_code,
    encode_categorical,
    encode_code,
    handle_missing,
    missing_values_code,
    outlier_counts,
    outliers_code,
    preprocessor_code,
    remove_duplicates,
    remove_outliers,
    scale_code,
    scale_numeric,
    split_code,
    split_train_test,
)
from utils.session import (
    get_current_dataset,
    get_current_dataset_name,
    set_preprocessor,
    set_train_test_split,
)

_MODULE = get_module("Data Preprocessing")


def render_dataset_banner(df, name: str | None) -> None:
    """Show a compact banner describing the dataset being cleaned."""
    st.caption(
        f"Cleaning: **{name or 'current dataset'}** "
        f"({df.shape[0]:,} rows x {df.shape[1]} columns)"
    )


def render_quick_overview(df) -> None:
    """Show the four headline quality metrics before any cleaning."""
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", df.shape[0])
    col_b.metric("Columns", df.shape[1])
    col_c.metric("Missing cells", int(df.isna().sum().sum()))
    col_d.metric("Duplicate rows", int(df.duplicated().sum()))


def render_step_header(number: int, title: str, description: str) -> None:
    """Render a numbered step heading with a short explanation."""
    st.markdown(f"**Step {number} - {title}**")
    st.caption(description)


def render_code(code: str) -> None:
    """Render a Python snippet inside a bordered container."""
    with st.container(border=True):
        st.caption("Python code for this step")
        st.code(code, language="python")


def render_step_result(label: str, info: dict) -> None:
    """Render the outcome of applying a step plus any warnings."""
    details = ", ".join(f"{key}: {value}" for key, value in info.items())
    st.success(f"{label} applied. ({details})")
    for warning in info.get("warnings", []):
        st.warning(warning)


def render_missing_step(working, snippets) -> tuple:
    """Render and apply the missing-values step.

    Returns ``(updated_df, config)``.
    """
    config = {"enabled": False}

    with st.expander("Step 1 - Handle missing values", expanded=True):
        render_step_header(
            1,
            "Handle missing values",
            "Decide what to do with the cells that contain no value. Dropping "
            "rows or columns removes data; imputation fills gaps with a "
            "statistic so you keep every row.",
        )
        enabled = st.checkbox("Enable this step", key="pp_missing_enabled")
        if not enabled:
            return working, config
        config["enabled"] = True

        candidate_columns = working.columns[working.isna().any()].tolist()
        if not candidate_columns:
            st.success("No missing values found in this dataset.")
            return working, config

        strategy = st.selectbox(
            "Imputation strategy",
            MISSING_STRATEGIES,
            key="pp_missing_strategy",
        )
        columns = st.multiselect(
            "Columns to fix",
            candidate_columns,
            default=candidate_columns,
            key="pp_missing_cols",
        )
        fill_value = None
        if strategy == "constant":
            fill_value = st.text_input("Fill value", value="0", key="pp_missing_fill")

        config.update(
            {
                "strategy": strategy,
                "columns": columns,
                "fill_value": fill_value,
            }
        )

        if not columns:
            st.info("Select at least one column to continue.")
            return working, config

        result, info = handle_missing(working, columns, strategy, fill_value)
        render_step_result("Missing-value handling", info)
        st.dataframe(result.head(5), width="stretch")
        code = missing_values_code(columns, strategy, fill_value)
        render_code(code)
        snippets.append(("Missing values", code))

        render_education(
            "Missing values",
            "Missing values appear as NaN (Not a Number) and silently break "
            "most analyses and models.\n\n"
            "- **Drop rows**: removes whole observations; safe when few rows "
            "are affected, wasteful when many are.\n"
            "- **Drop columns**: removes a feature entirely; use when a column "
            "is mostly empty or not informative.\n"
            "- **Mean / median**: fill with a central value of the *same* "
            "column. Median resists outliers better than the mean.\n"
            "- **Mode**: fill with the most frequent value; useful for "
            "categorical columns.\n"
            "- **Constant**: fill with a fixed value you choose.\n\n"
            "Imputers are *learned* statistics, so they must be fitted on the "
            "training set only to avoid data leakage.",
        )
        return result, config


def render_duplicates_step(working, snippets) -> tuple:
    """Render and apply the duplicates step.

    Returns ``(updated_df, config)``.
    """
    config = {"enabled": False}

    with st.expander("Step 2 - Remove duplicates", expanded=True):
        render_step_header(
            2,
            "Remove duplicates",
            "Identical rows usually result from data-entry errors or repeated "
            "imports. They inflate counts and bias models, so they are removed.",
        )
        enabled = st.checkbox("Enable this step", key="pp_dup_enabled")
        if not enabled:
            return working, config
        config["enabled"] = True

        result, info = remove_duplicates(working)
        render_step_result("Duplicate removal", info)
        st.dataframe(result.head(5), width="stretch")
        code = duplicates_code()
        render_code(code)
        snippets.append(("Duplicates", code))

        render_education(
            "Duplicate rows",
            "A fully duplicated row is an observation that appears more than "
            "once. Removing duplicates keeps one copy of each unique row. "
            "Unlike imputation this is a deterministic operation, so it does "
            "not cause data leakage when applied before splitting.",
        )
        return result, config


def render_outliers_step(working, snippets) -> tuple:
    """Render and apply the outliers step.

    Returns ``(updated_df, config)``.
    """
    config = {"enabled": False}

    with st.expander("Step 3 - Handle outliers", expanded=True):
        render_step_header(
            3,
            "Handle outliers",
            "Outliers are extreme values that can skew averages and confuse "
            "models. The IQR rule flags values far outside the middle half of "
            "the data.",
        )
        enabled = st.checkbox("Enable this step", key="pp_out_enabled")
        if not enabled:
            return working, config
        config["enabled"] = True

        numeric = numeric_columns(working)
        if not numeric:
            st.info("This dataset has no numeric columns to check for outliers.")
            return working, config

        columns = st.multiselect("Columns to check", numeric, key="pp_out_cols")
        threshold = st.slider(
            "IQR multiplier (1.5 = standard)",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.5,
            key="pp_out_threshold",
        )
        remove = st.checkbox(
            "Remove rows flagged as outliers",
            value=False,
            key="pp_out_remove",
        )
        config.update(
            {
                "columns": columns,
                "threshold": threshold,
                "remove": remove,
            }
        )

        if not columns:
            st.info("Select at least one numeric column to continue.")
            return working, config

        counts = outlier_counts(working, columns, threshold)
        st.dataframe(counts, width="stretch")

        result = working
        if remove:
            result, info = remove_outliers(working, columns, threshold)
            render_step_result("Outlier removal", info)
        else:
            st.caption(
                "Detection only - no rows removed. Tick the box above to "
                "remove them."
            )

        code = outliers_code(columns, threshold)
        render_code(code)
        snippets.append(("Outliers", code))

        render_education(
            "Outliers and the IQR rule",
            "The interquartile range (IQR) is the distance between the 25th "
            "and 75th percentiles. Values below ``Q1 - 1.5*IQR`` or above "
            "``Q3 + 1.5*IQR`` are considered outliers.\n\n"
            "Removing outliers is a judgement call: sometimes they are "
            "measurement errors, sometimes they are the most interesting part "
            "of the data. Explore them in the EDA module before deleting.",
        )
        return result, config


def render_encode_step(working, snippets) -> tuple:
    """Render and apply the categorical encoding step.

    Returns ``(updated_df, config)``.
    """
    config = {"enabled": False}

    with st.expander("Step 4 - Encode categoricals", expanded=True):
        render_step_header(
            4,
            "Encode categorical columns",
            "Machine learning works with numbers. Text categories must be "
            "converted into numbers before modeling.",
        )
        enabled = st.checkbox("Enable this step", key="pp_enc_enabled")
        if not enabled:
            return working, config
        config["enabled"] = True

        categorical = categorical_columns(working)
        if not categorical:
            st.info("This dataset has no non-numeric columns to encode.")
            return working, config

        method = st.selectbox("Encoding method", ENCODING_METHODS, key="pp_enc_method")
        columns = st.multiselect(
            "Columns to encode",
            categorical,
            key="pp_enc_cols",
        )
        config.update({"method": method, "columns": columns})

        if not columns:
            st.info("Select at least one categorical column to continue.")
            return working, config

        result, info = encode_categorical(working, columns, method)
        render_step_result("Categorical encoding", info)
        st.dataframe(result.head(5), width="stretch")
        code = encode_code(columns, method)
        render_code(code)
        snippets.append(("Encoding", code))

        render_education(
            "Categorical encoding",
            "- **One-hot encoding** creates a separate 0/1 column for every "
            "category. It never imposes an artificial order, but it adds many "
            "columns for high-cardinality features.\n"
            "- **Label encoding** replaces each category with an integer "
            "(0, 1, 2, ...). It is compact but implies an order that linear "
            "models may misinterpret, so it suits tree-based models best.\n\n"
            "The encoder is a learned transformer: fit it on the training set "
            "and reuse it on the test set so unseen categories are handled "
            "consistently.",
        )
        return result, config


def render_scale_step(working, snippets) -> tuple:
    """Render and apply the numeric scaling step.

    Returns ``(updated_df, config)``.
    """
    config = {"enabled": False}

    with st.expander("Step 5 - Scale numerics", expanded=True):
        render_step_header(
            5,
            "Scale numeric features",
            "Numeric features often live on very different scales. Scaling "
            "brings them to a comparable range so distance-based and gradient "
            "models treat them fairly.",
        )
        enabled = st.checkbox("Enable this step", key="pp_scale_enabled")
        if not enabled:
            return working, config
        config["enabled"] = True

        numeric = numeric_columns(working)
        if not numeric:
            st.info("This dataset has no numeric columns to scale.")
            return working, config

        method = st.selectbox("Scaling method", SCALING_METHODS, key="pp_scale_method")
        columns = st.multiselect("Columns to scale", numeric, key="pp_scale_cols")
        config.update({"method": method, "columns": columns})

        if not columns:
            st.info("Select at least one numeric column to continue.")
            return working, config

        try:
            result, info = scale_numeric(working, columns, method)
        except ValueError as exc:
            st.error(str(exc))
            return working, config
        render_step_result("Feature scaling", info)
        st.dataframe(result.head(5), width="stretch")
        code = scale_code(columns, method)
        render_code(code)
        snippets.append(("Scaling", code))

        render_education(
            "Feature scaling",
            "- **StandardScaler** subtracts the mean and divides by the "
            "standard deviation, producing values centered on 0 with unit "
            "variance.\n"
            "- **MinMaxScaler** squeezes values into a fixed range, usually "
            "[0, 1]. It is sensitive to outliers.\n"
            "- **RobustScaler** uses the median and quartiles, so extreme "
            "values barely affect it.\n\n"
            "Scaling is a learned transformation: the scaler parameters are "
            "fit on the training set and applied unchanged to the test set.",
        )
        return result, config


def render_split_step(working, snippets) -> tuple:
    """Render and apply the train/test split step.

    The split does not modify the downloadable dataset; it is stored in
    session state so modeling modules can reuse it. Returns ``(working, config)``.
    """
    config = {"enabled": False}

    with st.expander("Step 6 - Train/test split", expanded=True):
        render_step_header(
            6,
            "Train/test split",
            "Hold out part of the data to measure how well the model will "
            "perform on data it has never seen. The target column defines "
            "what you want to predict.",
        )
        enabled = st.checkbox("Enable this step", key="pp_split_enabled")
        if not enabled:
            return working, config
        config["enabled"] = True

        columns = list(working.columns)
        if len(columns) < 2:
            st.info("A split needs at least one feature column plus a target.")
            return working, config

        target = st.selectbox("Target column (what to predict)", columns, key="pp_split_target")
        test_size = st.slider(
            "Test size (fraction held out)",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="pp_split_test_size",
        )
        random_state = st.number_input(
            "Random state (reproducibility seed)",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            key="pp_split_random_state",
        )
        stratify = st.checkbox(
            "Stratify (keep class proportions in both sets)",
            value=False,
            key="pp_split_stratify",
        )
        config.update(
            {
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
                "stratify": stratify,
            }
        )

        try:
            split = split_train_test(
                working,
                target,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            st.error(str(exc))
            return working, config

        set_train_test_split(split)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Train rows", len(split["X_train"]))
        col_b.metric("Test rows", len(split["X_test"]))
        col_c.metric("Features", split["X_train"].shape[1])
        col_d.metric("Target classes", split["y_train"].nunique())

        st.markdown("**Training features (first 5 rows)**")
        st.dataframe(split["X_train"].head(5), width="stretch")
        st.markdown("**Training target (first 5 rows)**")
        st.dataframe(split["y_train"].head(5), width="stretch")

        code = split_code(target, test_size, random_state, stratify)
        render_code(code)
        snippets.append(("Train/test split", code))

        render_education(
            "Train/test split and data leakage",
            "The train/test split reserves a slice of the data to honestly "
            "measure model performance.\n\n"
            "**Data leakage** happens when information from the test set "
            "reaches the model during training - for example by fitting the "
            "scaler on all the data *before* splitting. The model then looks "
            "better than it really is.\n\n"
            "The fix: split first, then fit every learned transformer "
            "(imputer, encoder, scaler) on the training set only and apply it "
            "unchanged to the test set. This module's reusable preprocessor "
            "below does exactly that.",
        )
        return working, config


def render_before_after(before, after) -> None:
    """Render the before/after quality comparison table."""
    st.subheader("Before vs. after")
    st.dataframe(compare_before_after(before, after), width="stretch")
    st.info(
        "The numbers above compare the original dataset with the result of "
        "every enabled step. They are factual observations, not a verdict on "
        "whether your cleaning is 'good'."
    )


def render_full_code(snippets) -> None:
    """Render one combined, copy-paste-ready Python script."""
    st.subheader("Complete Python code")
    if not snippets:
        st.info("Enable at least one step to see the full code.")
        return
    parts = [
        "# Complete preprocessing script - copy into your own notebook",
        "import pandas as pd",
        "",
        "# Load your dataset",
        "# df = pd.read_csv('your_data.csv')",
        "",
    ]
    for _, code in snippets:
        parts.append(code)
        parts.append("")
    with st.container(border=True):
        st.code("\n".join(parts), language="python")


def render_reusable_pipeline(df, configs: dict) -> None:
    """Build and store the reusable preprocessor used by modeling modules.

    The preprocessor captures the *learned* transformations (imputation,
    encoding, scaling). It is fitted on the **raw** training features using
    the same split settings, so it reproduces those steps for any new data
    later.

    Args:
        df: The original (pre-processing) DataFrame.
        configs: Dict of step name -> config from the workflow.
    """
    st.subheader("Reusable preprocessor")
    missing = configs.get("missing", {})
    encode = configs.get("encode", {})
    scale = configs.get("scale", {})

    impute_columns = []
    impute_strategy = "median"
    if missing.get("enabled") and missing.get("strategy") in (
        "mean",
        "median",
        "mode",
        "constant",
    ):
        impute_columns = missing.get("columns", [])
        impute_strategy = missing.get("strategy", "median")

    encode_columns = encode.get("columns", []) if encode.get("enabled") else []
    encode_method = encode.get("method", "one-hot")
    scale_columns = scale.get("columns", []) if scale.get("enabled") else []
    scale_method = scale.get("method", "StandardScaler")

    render_education(
        "What is this and why does it matter?",
        "The preprocessor bundles every *learned* transformation you "
        "configured - imputation, encoding and scaling - into one sklearn "
        "``ColumnTransformer``. Modeling modules apply it like this:\n\n"
        "1. ``preprocessor.fit(X_train)`` learns the statistics from the "
        "training set only.\n"
        "2. ``preprocessor.transform(X_test)`` applies the exact same "
        "transformation to the test set without re-learning.\n\n"
        "This is the correct, leak-free way to preprocess data for machine "
        "learning.",
    )

    if not (impute_columns or encode_columns or scale_columns):
        st.info(
            "No learned transformations selected yet. Enable an imputation, "
            "encoding, or scaling step above to build the preprocessor."
        )
        return

    preprocessor = build_preprocessor(
        impute_columns=impute_columns,
        impute_strategy=impute_strategy,
        impute_fill_value=missing.get("fill_value"),
        encode_columns=encode_columns,
        encode_method=encode_method,
        scale_columns=scale_columns,
        scale_method=scale_method,
    )

    if preprocessor is None:
        st.info("Nothing to preprocess with the current configuration.")
        return

    code = preprocessor_code(
        impute_columns=impute_columns,
        impute_strategy=impute_strategy,
        encode_columns=encode_columns,
        encode_method=encode_method,
        scale_columns=scale_columns,
        scale_method=scale_method,
    )
    render_code(code)

    split_config = configs.get("split", {})
    if split_config.get("enabled") and split_config.get("target"):
        try:
            raw_split = split_train_test(
                df,
                split_config["target"],
                test_size=split_config.get("test_size", 0.2),
                random_state=split_config.get("random_state", 42),
                stratify=split_config.get("stratify", False),
            )
            preprocessor.fit(raw_split["X_train"])
            set_preprocessor(preprocessor)
            st.success(
                "Preprocessor fitted on the **raw training** data and saved. "
                "Later modeling modules will reuse it to transform new data, "
                "so the test set stays untouched during fitting."
            )
        except (ValueError, KeyError) as exc:
            st.warning(
                f"The preprocessor could not be fitted on the training data: "
                f"{exc}. Enable Step 6 (or check the target column) to try again."
            )
            set_preprocessor(preprocessor)
    else:
        set_preprocessor(preprocessor)
        st.warning(
            "Preprocessor built but **not fitted** yet. Enable Step 6 so it "
            "can be fitted on the training data; until then, models will fit "
            "it themselves."
        )


def render_download(working) -> None:
    """Render the download button for the processed dataset."""
    st.subheader("Download processed dataset")
    if len(working) == 0:
        st.warning(
            "The processed dataset is empty - check that you have not removed "
            "every row with the cleaning steps."
        )
    st.download_button(
        "Download processed dataset (CSV)",
        data=working.to_csv(index=False),
        file_name="processed_data.csv",
        mime="text/csv",
        key="pp_download",
    )
    st.caption(f"The downloaded file contains {working.shape[0]:,} rows x {working.shape[1]} columns.")


def render_workflow(df) -> tuple:
    """Run the six steps in order and return the cleaned DataFrame.

    Returns:
        A tuple ``(working, snippets, configs)`` where ``working`` is the
        DataFrame after every enabled cleaning step, ``snippets`` is a list of
        ``(step_name, code)`` tuples for the enabled steps, and ``configs``
        maps each step name to its configuration dict.
    """
    snippets: list[tuple[str, str]] = []
    configs: dict[str, dict] = {}

    st.subheader("Cleaning workflow")
    render_education(
        "How this workflow works",
        "Each step below can be toggled on or off. The steps run from top to "
        "bottom, and every enabled step is applied to the result of the "
        "previous one. Step 6 (the split) is applied last and does not change "
        "the downloadable dataset - it only sets up the train/test data for "
        "modeling.",
    )

    working = df
    working, configs["missing"] = render_missing_step(working, snippets)
    working, configs["duplicates"] = render_duplicates_step(working, snippets)
    working, configs["outliers"] = render_outliers_step(working, snippets)
    working, configs["encode"] = render_encode_step(working, snippets)
    working, configs["scale"] = render_scale_step(working, snippets)
    working, configs["split"] = render_split_step(working, snippets)

    return working, snippets, configs


def main() -> None:
    """Assemble the Data Preprocessing page."""
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
    working, snippets, configs = render_workflow(df)

    st.markdown("---")
    render_before_after(df, working)
    render_full_code(snippets)
    render_reusable_pipeline(df, configs)
    render_download(working)

    render_sidebar_footer()


main()
