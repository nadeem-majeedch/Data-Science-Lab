"""Data Science Lab - Streamlit application entry point.

This is the landing page for the Data Science Lab, a modular educational
workspace for BS Data Science students. All interactive modules live under
``pages/`` and are exposed through Streamlit's native multipage navigation.
"""

import streamlit as st

from utils import APP_SUBTITLE, APP_TAGLINE, APP_TITLE, PAGES

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> None:
    """Populate the sidebar with navigation info and project links."""
    st.sidebar.title(APP_TITLE)
    st.sidebar.caption(APP_SUBTITLE)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    st.sidebar.markdown(
        "Use the pages menu above to explore the modules.\n\n"
        "Each module is a dedicated page of the Data Science Lab."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Getting started")
    st.sidebar.markdown(
        "1. Pick a module from the sidebar.\n"
        "2. Upload or load a dataset from `datasets/`.\n"
        "3. Run the analysis and download the results."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Links")
    st.sidebar.markdown(
        "- [Streamlit Docs](https://docs.streamlit.io)\n"
        "- [Pandas Docs](https://pandas.pydata.org/docs/)\n"
        "- [Scikit-learn Docs](https://scikit-learn.org/stable/)"
    )


def render_hero() -> None:
    """Render the landing page hero section."""
    st.markdown(f"# {APP_TITLE}")
    st.markdown(f"### {APP_SUBTITLE}")
    st.markdown(APP_TAGLINE)

    st.markdown("---")


def render_modules() -> None:
    """Render the module cards grid from the PAGES registry."""
    st.subheader("Explore the modules")

    entries = list(PAGES.items())
    # Arrange cards in rows of three columns.
    for i in range(0, len(entries), 3):
        row = entries[i : i + 3]
        columns = st.columns(3)
        for column, (name, path) in zip(columns, row):
            with column:
                st.markdown(f"### {name}")
                st.markdown(
                    "Explore, preprocess, model, and evaluate data with an "
                    "interactive Streamlit interface."
                )
                st.markdown(f"File: `{path}`")


def render_audience() -> None:
    """Render the intended-audience and how-it-works section."""
    st.subheader("Who is this for?")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            "**For BS Data Science students**, this lab turns textbook concepts "
            "into hands-on practice. Each module maps to a core topic of the "
            "curriculum: data inspection, cleaning, feature engineering, model "
            "training, and evaluation."
        )

    with col_b:
        st.markdown(
            "**For instructors**, this is a ready-made demo environment. Modules "
            "are deliberately kept modular and readable, so they can be extended "
            "with new datasets, models, and exercises."
        )

    st.markdown("---")

    st.subheader("How it works")
    st.markdown(
        "The app is a standard Streamlit multipage application. "
        "`app.py` is the landing page, and every file under `pages/` becomes a "
        "page in the sidebar. Shared helpers live in `utils/`, datasets in "
        "`datasets/`, experiments in `notebooks/`, and generated output in "
        "`reports/`."
    )

    st.markdown(
        "```bash\n"
        "pip install -r requirements.txt\n"
        "streamlit run app.py\n"
        "```"
    )


def main() -> None:
    """Assemble the landing page."""
    render_sidebar()
    render_hero()
    render_modules()
    render_audience()


if __name__ == "__main__":
    main()
