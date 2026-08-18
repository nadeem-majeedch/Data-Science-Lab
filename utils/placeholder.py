"""Backwards-compatible placeholder helpers.

The canonical UI helpers now live in :mod:`utils.ui`. This module re-exports
the historical ``render_placeholder`` and ``render_sidebar_footer`` names so
existing callers keep working.
"""

from utils.ui import render_placeholder, render_sidebar_footer

__all__ = ["render_placeholder", "render_sidebar_footer"]
