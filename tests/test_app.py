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
    for page in sorted(PAGES_DIR.glob("*.py")):
        if page.name == "Home.py":
            continue
        app = _run_app(page)
        assert app.title, f"{page.name} rendered no page title"
        assert app.markdown, f"{page.name} rendered no subtitle/description"
        assert any(
            el.value.startswith("What you will learn")
            for el in app.subheader
        ), f"{page.name} is missing the learning outcomes section"


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
