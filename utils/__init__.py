"""Utility package for the Data Science Lab Streamlit app."""

from utils.config import (
    APP_SUBTITLE,
    APP_TAGLINE,
    APP_TITLE,
    APP_VERSION,
    DATA_MODULES,
    LEARNING_STAGES,
    MODULES,
    MODULES_BY_KEY,
    NAV_SECTIONS,
    get_module,
)
from utils.navigation import build_navigation
from utils.ui import (
    render_module_card,
    render_module_grid,
    render_module_placeholder,
    render_page_footer,
    render_page_header,
    render_page_sidebar,
    render_placeholder,
    render_sidebar_footer,
    render_status_badge,
)

__all__ = [
    "APP_SUBTITLE",
    "APP_TAGLINE",
    "APP_TITLE",
    "APP_VERSION",
    "DATA_MODULES",
    "LEARNING_STAGES",
    "MODULES",
    "MODULES_BY_KEY",
    "NAV_SECTIONS",
    "get_module",
    "build_navigation",
    "render_module_card",
    "render_module_grid",
    "render_module_placeholder",
    "render_page_footer",
    "render_page_header",
    "render_page_sidebar",
    "render_placeholder",
    "render_sidebar_footer",
    "render_status_badge",
]
