"""Data Science Lab - polished dashboard home page."""

import streamlit as st

from utils import (
    APP_SUBTITLE,
    APP_TAGLINE,
    APP_TITLE,
    APP_VERSION,
    DATA_MODULES,
    LEARNING_STAGES,
    render_module_grid,
    render_sidebar_footer,
)

MODULES_PER_ROW = 2


def render_hero() -> None:
    """Render the dashboard hero section."""
    st.title(APP_TITLE)
    st.markdown(f"### {APP_SUBTITLE}")
    st.markdown(APP_TAGLINE)
    st.markdown("---")


def render_stats() -> None:
    """Render a row of headline metrics."""
    stage_count = len(LEARNING_STAGES)
    module_count = len(DATA_MODULES)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Interactive modules", module_count)
    col_b.metric("Learning stages", stage_count)
    col_c.metric("Project phase", "Foundation")
    col_d.metric("Version", APP_VERSION)
    st.markdown("---")


def render_learning_path() -> None:
    """Render the curriculum roadmap as a horizontal step strip."""
    st.subheader("Your learning path")

    cols = st.columns(len(LEARNING_STAGES))
    for index, (col, (step, goal, _modules)) in enumerate(zip(cols, LEARNING_STAGES)):
        with col:
            with st.container(border=True):
                st.markdown(f"**Step {index + 1}: {step}**")
                st.caption(goal)

    st.markdown(
        "Work through the modules in order, or jump straight to the topic "
        "that matches today's lesson."
    )
    st.markdown("---")


def render_modules() -> None:
    """Render the full module grid."""
    st.subheader("Explore the modules")
    st.markdown(
        "Each module below is a self-contained page of the lab. Click any card "
        "to open it."
    )
    render_module_grid(DATA_MODULES, columns=MODULES_PER_ROW)
    st.markdown("---")


def render_about() -> None:
    """Render the About / intended audience section."""
    st.subheader("About Data Science Lab")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            "**For BS Data Science students**, this lab turns textbook concepts "
            "into hands-on practice. Each module maps to a core topic of the "
            "curriculum: data inspection, cleaning, feature engineering, model "
            "training, and evaluation. No prior coding required to get started."
        )

    with col_b:
        st.markdown(
            "**For instructors**, this is a ready-made demo environment. Modules "
            "are deliberately kept modular and readable, so they can be extended "
            "with new datasets, models, and exercises."
        )

    st.markdown(
        "The app is built entirely with Python and Streamlit. The main script "
        "`app.py` wires up the sidebar navigation, and every file under "
        "`pages/` is a module. Shared helpers live in `utils/`, datasets in "
        "`datasets/`, experiments in `notebooks/`, and generated output in "
        "`reports/`."
    )

    st.markdown(
        "```bash\n"
        "pip install -r requirements.txt\n"
        "streamlit run app.py\n"
        "```"
    )

    render_sidebar_footer()


def main() -> None:
    """Assemble the dashboard."""
    render_hero()
    render_stats()
    render_learning_path()
    render_modules()
    render_about()


main()
