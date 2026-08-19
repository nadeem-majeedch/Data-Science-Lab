"""Session-state helpers for sharing the current dataset across modules.

The Dataset Explorer stores the active dataset under a well-known session
state key so later modules (EDA, Preprocessing, Modeling, ...) can pick up
exactly the same DataFrame without re-uploading it.
"""

import streamlit as st

DATASET_STATE_KEY = "dataset"

DATASET_NAME_KEY = "dataset_name"

PREPROCESSOR_KEY = "preprocessor"

SPLIT_KEY = "train_test_split"

FEATURE_OPS_KEY = "feature_ops"

TRAINED_MODEL_KEY = "trained_model"

TRAINED_MODEL_FEATURES_KEY = "trained_model_features"


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


def get_preprocessor():
    """Return the fitted sklearn preprocessor, or ``None`` if not built yet.

    The preprocessor is created by the Data Preprocessing module and reused by
    later modeling modules so that train/test transformations stay consistent
    and leak-free.
    """
    return st.session_state.get(PREPROCESSOR_KEY)


def set_preprocessor(preprocessor) -> None:
    """Store the fitted sklearn preprocessor in session state."""
    st.session_state[PREPROCESSOR_KEY] = preprocessor


def clear_preprocessor() -> None:
    """Remove the stored preprocessor from session state."""
    st.session_state.pop(PREPROCESSOR_KEY, None)


def get_train_test_split() -> dict | None:
    """Return the stored train/test split dict, or ``None`` if not created."""
    return st.session_state.get(SPLIT_KEY)


def set_train_test_split(split: dict) -> None:
    """Store the train/test split dict in session state."""
    st.session_state[SPLIT_KEY] = split


def clear_train_test_split() -> None:
    """Remove the stored train/test split from session state."""
    st.session_state.pop(SPLIT_KEY, None)


def get_feature_ops() -> list[dict]:
    """Return the list of applied feature-engineering operations."""
    return st.session_state.get(FEATURE_OPS_KEY, [])


def set_feature_ops(ops: list[dict]) -> None:
    """Store the applied feature-engineering operations in session state."""
    st.session_state[FEATURE_OPS_KEY] = list(ops)


def clear_feature_ops() -> None:
    """Remove the feature-engineering operation history."""
    st.session_state.pop(FEATURE_OPS_KEY, None)


def get_trained_model():
    """Return the fitted model trained by a modeling module, or ``None``."""
    return st.session_state.get(TRAINED_MODEL_KEY)


def get_trained_model_features() -> list[str] | None:
    """Return the feature names used to train the stored model, or ``None``."""
    return st.session_state.get(TRAINED_MODEL_FEATURES_KEY)


def set_trained_model(model, features: list[str] | None = None) -> None:
    """Store a fitted model and the feature names it was trained on."""
    st.session_state[TRAINED_MODEL_KEY] = model
    st.session_state[TRAINED_MODEL_FEATURES_KEY] = features


def clear_trained_model() -> None:
    """Remove the stored trained model and its feature names."""
    st.session_state.pop(TRAINED_MODEL_KEY, None)
    st.session_state.pop(TRAINED_MODEL_FEATURES_KEY, None)
