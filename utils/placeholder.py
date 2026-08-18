"""Reusable Streamlit components shared across the app."""

import streamlit as st


def render_placeholder(
    title: str,
    description: str,
    planned_features: list[str] | None = None,
    related_page: str | None = None,
) -> None:
    """Render a consistent "coming soon" layout for not-yet-built pages.

    Args:
        title: Page title shown in the header.
        description: One or two sentences describing the module's purpose.
        planned_features: Bullet points of functionality planned for this module.
        related_page: Optional display name of a related module to link to.
    """
    st.title(title)
    st.markdown(description)

    st.markdown("---")

    planned = planned_features or ["Placeholder module - functionality coming soon."]

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("Planned features")
        for feature in planned:
            st.markdown(f"- {feature}")

    with col_b:
        st.info(
            "This module is a placeholder. It will be implemented as part of "
            "the Data Science Lab learning roadmap. Check back soon."
        )

    if related_page:
        st.caption(f"Related module: {related_page}")

    st.markdown("---")
    st.caption("Data Science Lab - placeholder module")


def render_sidebar_footer() -> None:
    """Render a consistent footer block for the application sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.caption("Data Science Lab")
    st.sidebar.caption("Built with Streamlit for BS Data Science students")
