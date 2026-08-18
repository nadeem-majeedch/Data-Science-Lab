"""Data Science Lab - Streamlit application entry point.

This script configures the app-wide page settings and wires up the grouped
sidebar navigation. The dashboard itself lives in ``pages/Home.py``; each
learning module is a dedicated page under ``pages/``.
"""

import streamlit as st

from utils import APP_TITLE, APP_SUBTITLE, build_navigation

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

nav = build_navigation()
nav.run()
