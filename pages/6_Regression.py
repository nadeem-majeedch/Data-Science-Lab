"""Regression placeholder module."""

import streamlit as st

from utils import render_placeholder

st.set_page_config(page_title="Regression", layout="wide")

render_placeholder(
    title="Regression",
    description=(
        "Model continuous numeric outcomes, such as house prices or sales "
        "forecasts, with interpretable and powerful regressors."
    ),
    planned_features=[
        "Continuous target variable support",
        "Common regressors: linear, ridge, trees, ensembles",
        "Residual analysis and prediction plots",
        "Train / validation / test evaluation",
    ],
    related_page="Classification",
)
