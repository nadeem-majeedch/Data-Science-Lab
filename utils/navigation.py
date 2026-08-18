"""Sidebar navigation builder for the Data Science Lab app."""

import streamlit as st

from utils.config import MODULES


def build_navigation():
    """Build the grouped sidebar navigation from the module registry.

    Returns:
        A Streamlit ``navigation`` object ready to be run with ``.run()``.
    """
    sections: dict[str, list] = {}
    for module in MODULES:
        page = st.Page(
            module.file,
            title=module.title,
            url_path=module.key,
            default=module.key == "home",
        )
        sections.setdefault(module.section, []).append(page)
    return st.navigation(sections, position="sidebar")
