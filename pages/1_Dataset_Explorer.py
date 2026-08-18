"""Dataset Explorer module.

Students can load a CSV or Excel dataset (uploaded or from the bundled
sample datasets), inspect its structure, preview the rows, explore
descriptive statistics, run data-quality checks, and download the result.
The active dataset is stored in session state so later modules can reuse it.
"""

import streamlit as st

from utils import (
    get_module,
    render_education,
    render_page_header,
    render_page_sidebar,
    render_sidebar_footer,
)
from utils.data_analysis import (
    categorical_columns,
    categorical_stats,
    constant_columns,
    data_quality_score,
    dataframe_info,
    duplicate_rows,
    format_bytes,
    head,
    memory_usage_bytes,
    missing_values,
    numeric_columns,
    numeric_stats,
    quality_report,
    random_sample,
    tail,
    unique_counts,
)
from utils.data_loader import DataLoadError, list_sample_datasets, load_dataset, load_sample_dataset
from utils.session import get_current_dataset, get_current_dataset_name, set_current_dataset

_MODULE = get_module("Dataset Explorer")


def choose_dataset():
    """Render the sidebar dataset controls and return ``(df, name)``."""
    source = st.sidebar.radio(
        "Data source",
        ["Sample dataset", "Upload your own"],
        key="ds_source",
    )

    if source == "Upload your own":
        uploaded = st.sidebar.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx"],
            key="ds_upload",
        )
        if uploaded is not None:
            try:
                dataframe = load_dataset(uploaded)
                set_current_dataset(uploaded.name, dataframe)
                st.sidebar.success(f"Loaded '{uploaded.name}'")
            except DataLoadError as exc:
                st.sidebar.error(str(exc))
    else:
        samples = list_sample_datasets()
        if not samples:
            st.sidebar.warning("No sample datasets found in datasets/samples/.")
        else:
            samples_by_name = {sample.name: sample for sample in samples}
            chosen = st.sidebar.selectbox(
                "Sample dataset",
                list(samples_by_name),
                key="ds_sample_name",
            )
            if st.sidebar.button("Load sample", key="ds_load_sample"):
                try:
                    dataframe = load_sample_dataset(samples_by_name[chosen])
                    set_current_dataset(chosen, dataframe)
                    st.sidebar.success(f"Loaded '{chosen}'")
                except DataLoadError as exc:
                    st.sidebar.error(str(exc))

    return get_current_dataset(), get_current_dataset_name()


def render_overview(name: str, df) -> None:
    """Render dataset name, shape, memory usage and data types."""
    info = dataframe_info(df)
    completeness = data_quality_score(df)

    st.subheader("Dataset overview")
    st.caption(f"Currently loaded: **{name}**")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", info["rows"])
    col_b.metric("Columns", info["columns"])
    col_c.metric("Memory usage", format_bytes(info["memory_bytes"]))
    col_d.metric("Completeness", f"{completeness * 100:.1f}%")

    st.markdown("**Data types**")
    dtype_rows = [
        {"Column": column, "Data type": dtype}
        for column, dtype in info["dtypes"].items()
    ]
    st.dataframe(dtype_rows, width="stretch")

    render_education(
        "Dataset shape and size",
        "**Rows** are the observations (each student, sale, or measurement), and "
        "**columns** are the features describing them. Knowing the shape tells "
        "you how much data you have and whether it is wide (many features) or "
        "long (many observations).\n\n"
        "**Memory usage** shows how much RAM the dataset occupies. Deep string "
        "columns are the main memory hogs, which matters on limited machines. "
        "**Completeness** is the percentage of cells that are filled in - 100% "
        "means no missing values anywhere.",
    )


def render_preview(df) -> None:
    """Render the full dataset in an interactive table."""
    st.subheader("Full dataset")
    st.dataframe(df, width="stretch")
    render_education(
        "Reading an interactive table",
        "The interactive table lets you scroll, sort by any column (click the "
        "header) and search the data. Sorting is only for inspection - it does "
        "not change the stored dataset. This is the fastest way to get a feel "
        "for your data before formal analysis.",
    )


def render_head_tail_sample(df) -> None:
    """Render head, tail and random sample views in tabs."""
    st.subheader("Head, tail and random sample")

    tab_head, tab_tail, tab_sample = st.tabs(["Head", "Tail", "Random sample"])

    with tab_head:
        st.dataframe(head(df), width="stretch")
        render_education(
            "Why look at the head?",
            "The **head** shows the first rows of the dataset, which is useful "
            "for checking column names, formats, and what a typical row looks "
            "like. Be careful: the first rows are not always representative of "
            "the whole dataset.",
        )

    with tab_tail:
        st.dataframe(tail(df), width="stretch")
        render_education(
            "Why look at the tail?",
            "The **tail** shows the last rows. Comparing the head and tail is a "
            "quick way to confirm the data was collected consistently from "
            "start to finish and to catch appended or malformed rows at the end.",
        )

    with tab_sample:
        st.dataframe(random_sample(df), width="stretch")
        render_education(
            "Why use a random sample?",
            "A **random sample** picks rows without regard to position. It gives "
            "you a more representative snapshot than the head or tail when rows "
            "are ordered by time, ID, or some other pattern. A fixed seed keeps "
            "the sample reproducible.",
        )


def render_numeric_stats(df) -> None:
    """Render descriptive statistics for numeric columns."""
    st.subheader("Numeric descriptive statistics")

    stats = numeric_stats(df)
    if stats.empty:
        st.info("This dataset has no numeric columns.")
    else:
        st.dataframe(stats, width="stretch")

    render_education(
        "Reading descriptive statistics",
        "**Count** is the number of non-missing values. **Mean** is the average, "
        "**std** (standard deviation) measures spread, and the percentiles "
        "(25%, 50%, 75%) describe the distribution. The **median** (50%) is "
        "more robust than the mean when there are extreme values. Compare the "
        "mean and median: a big gap often signals outliers. The **Missing** "
        "column shows how many values are absent for each variable.",
    )


def render_categorical_stats(df) -> None:
    """Render descriptive statistics for categorical columns."""
    st.subheader("Categorical descriptive statistics")

    stats = categorical_stats(df)
    if stats.empty:
        st.info("This dataset has no categorical columns.")
    else:
        st.dataframe(stats, width="stretch")

    render_education(
        "Understanding categorical statistics",
        "Categorical columns hold labels rather than numbers. **Count** is the "
        "number of filled values, **Unique** the number of distinct categories, "
        "and **Top** / **Frequency** the most common value and how often it "
        "occurs. A column with very few unique values (like a country code) is "
        "different from a free-text column with thousands of unique values.",
    )


def render_quality(name: str, df) -> None:
    """Render the data-quality checks and summary."""
    st.subheader("Data quality")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Duplicate rows", duplicate_rows(df))
    col_b.metric("Missing cells", int(df.isna().sum().sum()))
    col_c.metric("Constant columns", len(constant_columns(df)))

    st.markdown("**Missing values per column**")
    st.dataframe(missing_values(df), width="stretch")
    render_education(
        "Why missing values matter",
        "Missing values (NaN) can break calculations and bias machine-learning "
        "models. The table shows how many and what percentage of each column's "
        "values are missing. Columns with a high missing percentage may need to "
        "be dropped or imputed (filled in) - you will practice both in the Data "
        "Preprocessing module.",
    )

    st.markdown("**Unique values per column**")
    unique = unique_counts(df).rename("Unique values").reset_index()
    unique = unique.rename(columns={"index": "Column"})
    st.dataframe(unique, width="stretch")
    render_education(
        "Why unique values matter",
        "The number of **unique values** reveals a column's role: identifiers "
        "like order numbers have one value per row, while features like a "
        "country column have few. It also flags suspicious data - an "
        "unexpectedly high or low number of unique values can indicate data "
        "entry errors.",
    )

    if constant_columns(df):
        st.markdown("**Constant columns**")
        st.warning(
            "These columns contain a single value (or are entirely empty) and "
            "carry no information: "
            + ", ".join(f"`{column}`" for column in constant_columns(df))
            + "."
        )
    render_education(
        "Why constant columns are a problem",
        "A **constant column** has the same value in every row, so it cannot "
        "help distinguish one observation from another. Machine-learning models "
        "learn nothing from it and it can even cause errors. These columns are "
        "usually safe to drop during preprocessing.",
    )

    st.markdown("**Quality summary**")
    st.dataframe(quality_report(df), width="stretch")
    score = data_quality_score(df)
    st.metric("Overall data-quality score", f"{score * 100:.1f}%")
    render_education(
        "Interpreting the quality summary",
        f"The quality summary combines every check into one table. The "
        f"**overall score** is the share of cells that are filled in - "
        f"'{name}' scores {score * 100:.1f}%. Use this page to decide which "
        f"columns need cleaning before you move to the Data Preprocessing "
        f"module.",
    )


def render_download(name: str, df) -> None:
    """Render the download button for the current dataset."""
    st.subheader("Download the dataset")

    stem = name.rsplit(".", 1)[0] if "." in name else name
    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download as CSV",
        data=csv_data,
        file_name=f"{stem}.csv",
        mime="text/csv",
        key="ds_download",
    )
    st.caption(
        "The download reflects the dataset as it is currently displayed: all "
        "rows, all columns, in the same order as the original file."
    )


def main() -> None:
    """Assemble the Dataset Explorer page."""
    render_page_sidebar(_MODULE)
    render_page_header(_MODULE.title, _MODULE.subtitle, help_text=_MODULE.help_text)

    df, name = choose_dataset()

    if df is None:
        st.info(
            "No dataset selected yet. Use the sidebar to load a sample dataset "
            "or upload your own CSV or Excel file."
        )
        render_sidebar_footer()
        return

    render_overview(name, df)
    st.markdown("---")
    render_preview(df)
    st.markdown("---")
    render_head_tail_sample(df)
    st.markdown("---")
    render_numeric_stats(df)
    st.markdown("---")
    render_categorical_stats(df)
    st.markdown("---")
    render_quality(name, df)
    st.markdown("---")
    render_download(name, df)

    render_sidebar_footer()


main()
