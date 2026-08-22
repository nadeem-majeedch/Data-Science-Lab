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


def test_classification_renders_without_dataset():
    app = _run_app(PAGES_DIR / "5_Classification.py")
    assert not app.exception, app.exception
    assert app.title[0].value == "Classification"
    assert not app.metric


def test_classification_renders_with_dataset():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "5_Classification.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    assert not app.exception, app.exception
    labels = [metric.label for metric in app.metric]
    assert "Rows" in labels and "Classes" in labels
    # Default target is the first categorical column with 2-20 classes.
    assert app.selectbox(key="clf_target").value == "subject"


def test_classification_trains_a_model():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "5_Classification.py"), default_timeout=60)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    app.selectbox(key="clf_target").set_value("grade")
    app.run()
    app.button(key="clf_train").click()
    app.run()

    assert not app.exception, app.exception
    assert app.session_state["trained_model"] is not None
    assert app.session_state["classification_results"] is not None
    results = app.session_state["classification_results"]
    assert set(results["metrics"]) >= {"accuracy", "precision", "recall", "f1"}
    assert 0.0 <= results["metrics"]["accuracy"] <= 1.0
    assert len(results["classes"]) == 5  # A/B/C/D/F
    assert app.session_state["trained_model_features"] is not None


def test_classification_train_failure_is_graceful():
    import pandas as pd

    df = pd.DataFrame({"id": list(range(8)), "grade": ["A"] * 8})

    app = AppTest.from_file(str(PAGES_DIR / "5_Classification.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "single_class.csv"
    app.run()

    app.selectbox(key="clf_target").set_value("grade")
    app.run()

    assert not app.exception, app.exception
    # A single-class target renders an error and no model is trained.
    assert any("only one class" in el.value for el in app.error)


def test_classification_sample_row_prediction():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "5_Classification.py"), default_timeout=60)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    app.selectbox(key="clf_target").set_value("grade")
    app.run()
    app.button(key="clf_train").click()
    app.run()

    assert app.selectbox(key="clf_pred_row").value is not None
    assert any(
        "Actual class" in el.value for el in app.markdown
    ), "expected the actual-vs-predicted comparison"


def test_classification_custom_prediction():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "5_Classification.py"), default_timeout=60)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    app.selectbox(key="clf_target").set_value("grade")
    app.run()
    app.button(key="clf_train").click()
    app.run()

    app.radio(key="clf_pred_mode").set_value("Enter your own values")
    app.run()

    assert not app.exception, app.exception
    app.button(key="clf_predict").click()
    app.run()
    assert not app.exception, app.exception
    assert any(
        "Predicted class" in el.value for el in app.success
    ), "expected a predicted class after clicking Predict"


def test_regression_renders_without_dataset():
    app = _run_app(PAGES_DIR / "6_Regression.py")
    assert not app.exception, app.exception
    assert app.title[0].value == "Regression"
    assert not app.metric


def test_regression_renders_with_dataset():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "6_Regression.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    assert not app.exception, app.exception
    labels = [metric.label for metric in app.metric]
    assert "Rows" in labels and "Values" in labels
    # Default target is the first varying numeric column.
    assert app.selectbox(key="reg_target").value == "attendance_pct"


def test_regression_trains_a_model():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "6_Regression.py"), default_timeout=60)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    app.selectbox(key="reg_target").set_value("final")
    app.run()
    app.button(key="reg_train").click()
    app.run()

    assert not app.exception, app.exception
    assert app.session_state["trained_model"] is not None
    assert app.session_state["regression_results"] is not None
    results = app.session_state["regression_results"]
    assert set(results["metrics"]) == {"mae", "mse", "rmse", "r2"}
    assert results["metrics"]["mse"] >= 0.0
    assert list(results["predictions"].columns) == ["Actual", "Predicted", "Residual"]
    assert app.get("plotly_chart"), "expected actual-vs-predicted and residual plots"
    assert app.session_state["trained_model_features"] is not None


def test_regression_train_failure_is_graceful():
    import pandas as pd

    df = pd.DataFrame({"grade": ["A", "B", "C", "D"], "x": [1, 2, 3, 4]})

    app = AppTest.from_file(str(PAGES_DIR / "6_Regression.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "text_target.csv"
    app.run()

    # Switch to the text target -> regression validation must complain.
    app.selectbox(key="reg_target").set_value("grade")
    app.run()

    assert not app.exception, app.exception
    assert any("not numeric" in el.value for el in app.error)


def test_model_evaluation_renders_without_dataset():
    app = _run_app(PAGES_DIR / "7_Model_Evaluation.py")
    assert not app.exception, app.exception
    assert app.title[0].value == "Model Evaluation"
    assert not app.metric


def test_model_evaluation_renders_without_trained_model():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "7_Model_Evaluation.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    assert not app.exception, app.exception
    assert any("No trained model found" in el.value for el in app.info)


def test_model_evaluation_shows_classification_results():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from utils.model_training import train_classifier

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")
    features = ["attendance_pct", "midterm", "final", "subject"]
    results = train_classifier(
        df[features],
        df["grade"],
        LogisticRegression(max_iter=1000),
        random_state=42,
        stratify=True,
    )
    results["config"] = {
        "model_key": "Logistic Regression",
        "params": {"max_iter": 1000, "C": 1.0},
        "target": "grade",
        "features": features,
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    }

    app = AppTest.from_file(str(PAGES_DIR / "7_Model_Evaluation.py"), default_timeout=120)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.session_state["classification_results"] = results
    app.run()

    assert not app.exception, app.exception
    assert any(el.value.startswith("Evaluating") for el in app.subheader)
    assert app.get("plotly_chart"), "expected ROC and confusion-matrix charts"
    assert app.get("download_button"), "expected a CSV download button"


def test_model_evaluation_shows_regression_results():
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from utils.regression_training import train_regressor

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")
    features = ["attendance_pct", "midterm", "final", "subject"]
    results = train_regressor(df[features], df["final"], LinearRegression(), random_state=42)
    results["config"] = {
        "model_key": "Linear Regression",
        "params": {"fit_intercept": True},
        "target": "final",
        "features": features,
        "test_size": 0.2,
        "random_state": 42,
    }

    app = AppTest.from_file(str(PAGES_DIR / "7_Model_Evaluation.py"), default_timeout=120)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.session_state["regression_results"] = results
    app.run()

    assert not app.exception, app.exception
    assert any(el.value.startswith("Evaluating") for el in app.subheader)
    assert app.get("plotly_chart"), "expected residual charts"
    assert app.get("download_button"), "expected a CSV download button"


def test_model_comparison_renders_without_dataset():
    app = _run_app(PAGES_DIR / "9_Model_Comparison.py")
    assert not app.exception, app.exception
    assert app.title[0].value == "Model Comparison"
    assert not app.metric


def test_model_comparison_renders_with_dataset():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "9_Model_Comparison.py"), default_timeout=30)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    assert not app.exception, app.exception
    labels = [metric.label for metric in app.metric]
    assert "Rows" in labels
    # Default task is classification -> default target is a categorical column.
    assert app.selectbox(key="cmp_target").value == "subject"


def test_model_comparison_classification():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "9_Model_Comparison.py"), default_timeout=180)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    app.selectbox(key="cmp_target").set_value("grade")
    app.run()
    app.button(key="cmp_compare").click()
    app.run()

    assert not app.exception, app.exception
    results = app.session_state["comparison_results"]
    assert list(results["table"].columns) == ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    assert len(results["table"]) == 7
    assert results["table"]["AUC"].notna().all()
    assert app.get("download_button"), "expected a CSV download button"


def test_model_comparison_regression():
    import pandas as pd

    df = pd.read_csv(PROJECT_ROOT / "datasets" / "samples" / "student_grades.csv")

    app = AppTest.from_file(str(PAGES_DIR / "9_Model_Comparison.py"), default_timeout=180)
    app.session_state["dataset"] = df
    app.session_state["dataset_name"] = "student_grades.csv"
    app.run()

    app.radio(key="cmp_task").set_value("Regression")
    app.run()
    app.selectbox(key="cmp_target").set_value("final")
    app.run()
    app.button(key="cmp_compare").click()
    app.run()

    assert not app.exception, app.exception
    results = app.session_state["comparison_results"]
    assert list(results["table"].columns) == ["Model", "MAE", "RMSE", "R2"]
    assert len(results["table"]) == 7
    assert app.get("download_button"), "expected a CSV download button"
