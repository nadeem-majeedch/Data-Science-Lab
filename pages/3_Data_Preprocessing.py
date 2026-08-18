"""Data Preprocessing placeholder module."""

import streamlit as st

from utils import render_placeholder

st.set_page_config(page_title="Data Preprocessing", layout="wide")

render_placeholder(
    title="Data Preprocessing",
    description=(
        "Clean and transform raw data so it is ready for exploratory analysis "
        "and machine learning pipelines."
    ),
    planned_features=[
        "Handle missing values (drop or impute)",
        "Remove or cap outliers",
        "Encode categorical variables",
        "Scale and normalize numeric features",
        "Split data into train / validation / test sets",
    ],
    related_page="Feature Engineering",
)
