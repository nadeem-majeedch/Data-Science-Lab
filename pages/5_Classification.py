"""Classification placeholder module."""

import streamlit as st

from utils import render_placeholder

st.set_page_config(page_title="Classification", layout="wide")

render_placeholder(
    title="Classification",
    description=(
        "Train and tune classifiers to predict discrete categories, such as "
        "spam detection or disease diagnosis."
    ),
    planned_features=[
        "Support for binary and multi-class targets",
        "Common classifiers: logistic regression, trees, ensembles, SVM",
        "Configurable train / test split",
        "Hyperparameter presets and evaluation on holdout data",
    ],
    related_page="Regression",
)
