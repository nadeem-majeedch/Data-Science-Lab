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


def test_all_required_pages_exist():
    existing = {path.stem for path in PAGES_DIR.glob("*.py")}
    for page in REQUIRED_PAGES:
        assert any(page.lower().replace(" ", "_") in name.lower() for name in existing), (
            f"Missing page for module: {page}"
        )


def test_every_page_renders_without_errors():
    pages = sorted(PAGES_DIR.glob("*.py"))
    assert pages, "No placeholder pages found"
    for page in pages:
        app = _run_app(page)
        assert not app.exception, f"{page.name} raised an exception: {app.exception}"


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
