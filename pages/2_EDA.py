"""Exploratory Data Analysis placeholder module."""

import streamlit as st

from utils import render_placeholder

st.set_page_config(page_title="EDA", layout="wide")

render_placeholder(
    title="Exploratory Data Analysis (EDA)",
    description=(
        "Understand the structure, distributions, and relationships inside a "
        "dataset through interactive charts and summary statistics."
    ),
    planned_features=[
        "Descriptive statistics per column",
        "Distribution plots for numeric features",
        "Categorical value counts and bar charts",
        "Correlation heatmap between numeric variables",
        "Pairwise scatter plot matrix",
    ],
    related_page="Dataset Explorer",
)
