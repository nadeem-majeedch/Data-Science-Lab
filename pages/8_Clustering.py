"""Clustering placeholder module."""

import streamlit as st

from utils import render_placeholder

st.set_page_config(page_title="Clustering", layout="wide")

render_placeholder(
    title="Clustering",
    description=(
        "Discover natural groups and structure in unlabeled data using "
        "unsupervised learning techniques."
    ),
    planned_features=[
        "K-Means and hierarchical clustering",
        "Optional PCA for 2D visualization",
        "Elbow and silhouette analysis",
        "Cluster assignments and centroids export",
    ],
    related_page="EDA",
)
