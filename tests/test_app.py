"""Smoke tests for the Data Science Lab Streamlit app.

These tests use Streamlit's ``AppTest`` harness to run each page and assert
that it renders without raising an exception. No machine learning
functionality is exercised.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANDING_PAGE = PROJECT_ROOT / "app.py"
PAGES_DIR = PROJECT_ROOT / "pages"

REQUIRED_PAGES = [
    "Home",
    "Dataset Explorer",
    "EDA",
    "Data Preprocessing",
    "Feature Engineering",
    "Classification",
    "Regression",
    "Model Evaluation",
    "Clustering",
    "Model Comparison",
    "AutoML",
]


def _run_app(path: Path) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=30)
    app.run()
    return app


def test_landing_page_renders_without_errors():
    app = _run_app(LANDING_PAGE)
    assert not app.exception, app.exception
    assert app.title[0].value == "Data Science Lab"
    assert len(app.metric) == 4


def test_all_required_pages_exist():
    existing = {path.stem for path in PAGES_DIR.glob("*.py")}
    for page in REQUIRED_PAGES:
        assert any(page.lower().replace(" ", "_") in name.lower() for name in existing), (
            f"Missing page for module: {page}"
        )


def test_module_registry_is_consistent():
    from utils import MODULES
    from utils.config import get_module

    assert len(MODULES) == 11
    for module in MODULES:
        assert (PROJECT_ROOT / module.file).exists(), f"Missing file: {module.file}"
        assert get_module(module.title) is module


def test_every_page_renders_without_errors():
    pages = sorted(PAGES_DIR.glob("*.py"))
    assert pages, "No placeholder pages found"
    for page in pages:
        app = _run_app(page)
        assert not app.exception, f"{page.name} raised an exception: {app.exception}"


def test_placeholder_pages_share_consistent_header():
    from utils import MODULES

    for module in MODULES:
        if module.key == "home" or module.status != "planned":
            continue
        app = _run_app(PROJECT_ROOT / module.file)
        assert app.title, f"{module.file} rendered no page title"
        assert app.markdown, f"{module.file} rendered no subtitle/description"
        assert any(
            el.value.startswith("What you will learn")
            for el in app.subheader
        ), f"{module.file} is missing the learning outcomes section"


def test_dataset_explorer_renders_without_dataset():
    app = _run_app(PAGES_DIR / "1_Dataset_Explorer.py")
    assert not app.exception, app.exception
    assert app.title[0].value == "Dataset Explorer"
    assert not app.metric  # no metrics until a dataset is loaded


def test_dataset_explorer_loads_sample_dataset():
    app = _run_app(PAGES_DIR / "1_Dataset_Explorer.py")

    app.radio[0].set_value("Sample dataset")
    app.selectbox[0].select("student_grades.csv")
    app.button[0].click()
    app.run()

    assert not app.exception, app.exception
    assert app.metric, "expected overview metrics after loading a dataset"
    assert len(app.dataframe) >= 2  # overview table + full dataset + more
    assert app.session_state["dataset_name"] == "student_grades.csv"


def test_eda_renders_without_dataset():
    app = _run_app(PAGES_DIR / "2_EDA.py")
    assert not app.exception, app.exception
    assert app.title[0].value == "EDA"
    assert not app.metric


def test_eda_renders_with_dataset():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "2_EDA.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    assert not app.exception, app.exception
    assert len(app.metric) == 4  # automatic EDA summary metrics
    assert app.get("plotly_chart"), "expected an interactive histogram"


def test_eda_categorical_bar_chart_interaction():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "2_EDA.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    app.selectbox(key="eda_area").set_value("Categorical")
    app.run()

    app.selectbox(key="eda_cat_chart").set_value("Bar chart")
    app.selectbox(key="eda_cat_x").set_value("subject")
    app.selectbox(key="eda_cat_y").set_value("final")
    app.selectbox(key="eda_cat_agg").set_value("mean")
    app.run()

    assert not app.exception, app.exception
    assert app.get("plotly_chart"), "expected a bar chart after interaction"


def test_preprocessing_renders_without_dataset():
    app = _run_app(PAGES_DIR / "3_Data_Preprocessing.py")
    assert not app.exception, app.exception
    assert app.title[0].value == "Data Preprocessing"
    assert not app.metric


def test_preprocessing_renders_with_dataset():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "3_Data_Preprocessing.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    assert not app.exception, app.exception
    assert len(app.metric) == 4  # headline quality metrics
    assert app.session_state["dataset_name"] == "student_grades.csv"


def test_preprocessing_workflow_stores_preprocessor_and_split():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "3_Data_Preprocessing.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    # Pass 1: enable the learned steps plus the split.
    for key in (
        "pp_missing_enabled",
        "pp_enc_enabled",
        "pp_scale_enabled",
        "pp_split_enabled",
    ):
        app.checkbox(key=key).set_value(True)
    app.run()

    # Pass 2: configure the now-visible widgets.
    app.selectbox(key="pp_missing_strategy").set_value("median")
    app.multiselect(key="pp_missing_cols").set_value(["attendance_pct", "final"])
    app.multiselect(key="pp_enc_cols").set_value(["subject"])
    app.multiselect(key="pp_scale_cols").set_value(["midterm", "final"])
    app.selectbox(key="pp_split_target").set_value("grade")
    app.run()

    assert not app.exception, app.exception
    assert app.session_state["train_test_split"] is not None
    split = app.session_state["train_test_split"]
    assert len(split["X_train"]) > 0 and len(split["X_test"]) > 0
    preprocessor = app.session_state["preprocessor"]
    assert preprocessor is not None
    names = [name for name, _, _ in preprocessor.transformers]
    assert names == ["imputer", "encoder", "scaler"]


def test_feature_engineering_renders_without_dataset():
    app = _run_app(PAGES_DIR / "4_Feature_Engineering.py")
    assert not app.exception, app.exception
    assert app.title[0].value == "Feature Engineering"
    assert not app.metric


def test_feature_engineering_renders_with_dataset():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "4_Feature_Engineering.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    assert not app.exception, app.exception
    assert len(app.metric) == 4
    assert app.session_state["dataset_name"] == "student_grades.csv"


def test_feature_engineering_apply_undo_reset():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "4_Feature_Engineering.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    # Apply a square transform.
    app.selectbox(key="fe_op_type").set_value("Mathematical transformation")
    app.run()
    app.selectbox(key="fe_math_method").set_value("square")
    app.multiselect(key="fe_math_cols").set_value(["midterm", "final"])
    app.button(key="fe_apply").click()
    app.run()

    assert not app.exception, app.exception
    ops = app.session_state["feature_ops"]
    assert len(ops) == 1
    assert ops[0]["key"] == "math"

    # Undo removes it.
    app.button(key="fe_undo").click()
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["feature_ops"] == []

    # Reset is a no-op on an empty history.
    app.button(key="fe_reset").click()
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["feature_ops"] == []


def test_feature_engineering_replays_ops_onto_working_data():
    import pandas as pd

    from utils.feature_engineering import apply_feature_op

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "4_Feature_Engineering.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    # Build a chain: square midterm -> create ratio midterm/final.
    app.selectbox(key="fe_op_type").set_value("Mathematical transformation")
    app.run()
    app.selectbox(key="fe_math_method").set_value("square")
    app.multiselect(key="fe_math_cols").set_value(["midterm"])
    app.button(key="fe_apply").click()
    app.run()

    app.selectbox(key="fe_op_type").set_value("Create numeric feature")
    app.run()
    app.selectbox(key="fe_num_a").set_value("midterm")
    app.selectbox(key="fe_num_b").set_value("final")
    app.selectbox(key="fe_num_op").set_value("ratio")
    app.button(key="fe_apply").click()
    app.run()

    assert not app.exception, app.exception
    ops = app.session_state["feature_ops"]
    assert len(ops) == 2
    # Replaying the chain must give the second op the column the first added.
    working = df.copy()
    for op in ops:
        working = apply_feature_op(working, op)
    assert "midterm_squared" in working.columns
    assert "midterm_over_final" in working.columns


def test_feature_importance_placeholder_when_no_model():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "4_Feature_Engineering.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    assert not app.exception, app.exception
    assert any(
        "No trained model available" in el.value for el in app.info
    ), "expected the feature-importance placeholder hint"


def test_utils_render_placeholder():
    def build_test_page():
        from utils.placeholder import render_placeholder

        render_placeholder(
            title="Test Module",
            description="A test description.",
            planned_features=["Alpha", "Beta"],
        )

    app = AppTest.from_function(build_test_page, default_timeout=30)
    app.run()
    assert not app.exception, app.exception
    assert app.title[0].value == "Test Module"
