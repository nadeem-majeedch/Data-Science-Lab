"""Session-state helpers for sharing the current dataset across modules.

The Dataset Explorer stores the active dataset under a well-known session
state key so later modules (EDA, Preprocessing, Modeling, ...) can pick up
exactly the same DataFrame without re-uploading it.
"""

import streamlit as st

DATASET_STATE_KEY = "dataset"

DATASET_NAME_KEY = "dataset_name"


def get_current_dataset():
    """Return the currently selected DataFrame, or ``None`` if none is set."""
    return st.session_state.get(DATASET_STATE_KEY)


def get_current_dataset_name() -> str | None:
    """Return the name of the currently selected dataset, or ``None``."""
    return st.session_state.get(DATASET_NAME_KEY)


def set_current_dataset(name: str, dataframe) -> None:
    """Store the active dataset and its display name in session state."""
    st.session_state[DATASET_STATE_KEY] = dataframe
    st.session_state[DATASET_NAME_KEY] = name


def clear_current_dataset() -> None:
    """Remove the currently selected dataset from session state."""
    st.session_state.pop(DATASET_STATE_KEY, None)
    st.session_state.pop(DATASET_NAME_KEY, None)
