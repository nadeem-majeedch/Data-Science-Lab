"""AutoML placeholder module."""

import streamlit as st

from utils import render_placeholder

st.set_page_config(page_title="AutoML", layout="wide")

render_placeholder(
    title="AutoML",
    description=(
        "Automatically search over pipelines and hyperparameters to find strong "
        "models with minimal manual configuration."
    ),
    planned_features=[
        "Automatic task detection (classification / regression)",
        "Automated preprocessing pipeline",
        "Hyperparameter search over multiple algorithms",
        "Best model export and report generation",
    ],
    related_page="Model Comparison",
)
