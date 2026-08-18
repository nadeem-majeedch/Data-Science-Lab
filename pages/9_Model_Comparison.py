"""Model Comparison placeholder module."""

import streamlit as st

from utils import render_placeholder

st.set_page_config(page_title="Model Comparison", layout="wide")

render_placeholder(
    title="Model Comparison",
    description=(
        "Train several models side by side and rank them on a shared evaluation "
        "framework to pick the best approach."
    ),
    planned_features=[
        "Run multiple models on the same dataset",
        "Side-by-side metric comparison table",
        "Bars and box plots of performance",
        "Ranking with best model export",
    ],
    related_page="Model Evaluation",
)
