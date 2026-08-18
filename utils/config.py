"""Central configuration constants for the Data Science Lab app."""

APP_TITLE = "Data Science Lab"
APP_SUBTITLE = "An interactive learning workspace for BS Data Science students"
APP_TAGLINE = (
    "Explore, analyze, and model real-world data with an intuitive "
    "point-and-click interface built on Streamlit."
)

PROJECT_REPO = "https://github.com/your-org/data-science-lab"
PROJECT_DOCS = "https://docs.streamlit.io"

# Ordered navigation for the multipage app (display name -> page file).
# The sidebar shows pages automatically based on the pages/ directory.
PAGES = {
    "Dataset Explorer": "pages/1_Dataset_Explorer.py",
    "EDA": "pages/2_EDA.py",
    "Data Preprocessing": "pages/3_Data_Preprocessing.py",
    "Feature Engineering": "pages/4_Feature_Engineering.py",
    "Classification": "pages/5_Classification.py",
    "Regression": "pages/6_Regression.py",
    "Model Evaluation": "pages/7_Model_Evaluation.py",
    "Clustering": "pages/8_Clustering.py",
    "Model Comparison": "pages/9_Model_Comparison.py",
    "AutoML": "pages/10_AutoML.py",
}
