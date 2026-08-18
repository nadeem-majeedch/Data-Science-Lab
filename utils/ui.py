"""Reusable Streamlit UI helpers for the Data Science Lab app.

These functions keep the visual language of every page consistent: page
headers with titles, subtitles and help text, module cards, placeholders,
and the shared sidebar footer.
"""

import streamlit as st

from utils.config import APP_TITLE, Module


def render_page_header(title: str, subtitle: str, help_text: str | None = None) -> None:
    """Render a consistent page header with title, subtitle, and help.

    Args:
        title: Page title rendered as the main heading.
        subtitle: Short subtitle rendered beneath the title.
        help_text: Optional guidance shown in a collapsed "How to use" expander.
    """
    st.title(title)
    st.markdown(f"### {subtitle}")
    if help_text:
        with st.expander("How to use this module"):
            st.markdown(help_text)
    st.markdown("---")


def render_page_footer() -> None:
    """Render a consistent footer at the bottom of every page."""
    st.markdown("---")
    st.caption(f"{APP_TITLE} - an educational tool for BS Data Science students")


def render_education(title: str, body: str) -> None:
    """Render a collapsible educational explanation block.

    Args:
        title: Short label for the explainer (e.g. "Missing values").
        body: Markdown text explaining the concept and why it matters.
    """
    with st.expander(f"Learn more: {title}"):
        st.markdown(body)


def render_status_badge(status: str) -> None:
    """Render a small text badge describing the module's build status."""
    label = status.replace("_", " ").title()
    st.markdown(f"`Status: {label}`")


def render_module_card(module: Module) -> None:
    """Render a single module card with description and navigation link.

    Args:
        module: The Module metadata to display.
    """
    with st.container(border=True):
        st.markdown(f"### {module.title}")
        st.caption(module.subtitle)
        st.markdown(module.description)
        try:
            # Resolves when the page is registered by the app's navigation.
            st.page_link(module.file, label=f"Open {module.title}")
        except KeyError:
            # Standalone execution (e.g. tests) has no navigation registry.
            st.caption(f"Open {module.title} from the sidebar.")


def render_module_grid(modules: list[Module], columns: int = 2) -> None:
    """Render a grid of module cards.

    Args:
        modules: Modules to display as cards.
        columns: Number of cards per row.
    """
    for i in range(0, len(modules), columns):
        row = modules[i : i + columns]
        cols = st.columns(columns)
        for col, module in zip(cols, row):
            with col:
                render_module_card(module)


def render_module_placeholder(module: Module) -> None:
    """Render a consistent "under construction" page for a module.

    Args:
        module: The Module metadata describing the placeholder page.
    """
    render_page_header(module.title, module.subtitle, help_text=module.help_text)

    st.markdown(module.description)
    render_status_badge(module.status)

    st.markdown("---")

    col_a, col_b = st.columns([2, 1], gap="large")

    with col_a:
        st.subheader("What you will learn")
        for outcome in module.learning_outcomes:
            st.markdown(f"- {outcome}")

    with col_b:
        st.info(
            "This module is under construction. It will be implemented as part "
            "of the Data Science Lab learning roadmap. Check back soon."
        )

    render_page_footer()


def render_placeholder(
    title: str,
    description: str,
    planned_features: list[str] | None = None,
    related_page: str | None = None,
) -> None:
    """Render a "coming soon" page without relying on the module registry.

    Kept for backwards compatibility with the original placeholder pages.
    Prefer :func:`render_module_placeholder` for new pages.

    Args:
        title: Page title shown in the header.
        description: One or two sentences describing the module's purpose.
        planned_features: Bullet points of functionality planned for this module.
        related_page: Optional display name of a related module to link to.
    """
    render_page_header(title, "")

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

    render_page_footer()


def render_page_sidebar(module: Module) -> None:
    """Render module context at the top of the sidebar.

    Args:
        module: The Module currently being displayed.
    """
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Module: {module.title}")
    st.sidebar.caption(module.subtitle)


def render_sidebar_footer() -> None:
    """Render a consistent footer block at the bottom of the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.caption(APP_TITLE)
    st.sidebar.caption("Built with Streamlit for BS Data Science students")
