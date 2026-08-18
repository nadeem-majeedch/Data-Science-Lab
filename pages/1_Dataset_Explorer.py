"""Dataset Explorer placeholder module."""

import streamlit as st

from utils import render_placeholder

st.set_page_config(page_title="Dataset Explorer", layout="wide")

render_placeholder(
    title="Dataset Explorer",
    description=(
        "Browse, preview, and inspect datasets used in the lab before any "
        "analysis or modeling begins."
    ),
    planned_features=[
        "Upload datasets (CSV, Excel, JSON)",
        "Preview the first rows with schema inference",
        "Column-level statistics and data types",
        "Missing value and duplicate overview",
        "Quick filtering and sorting controls",
    ],
    related_page="EDA",
)
