"""Central configuration and module registry for the Data Science Lab app."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Module:
    """Metadata describing a single learning module of the lab.

    Attributes:
        key: Stable unique identifier used for the navigation URL path.
        title: Display name shown in the sidebar and page header.
        subtitle: Short subtitle shown under the page title.
        description: Educational description of what the module teaches.
        learning_outcomes: Bullet points of skills the student will practice.
        help_text: "How to use this module" guidance shown in an expander.
        section: Sidebar navigation group this module belongs to.
        file: Relative path to the module's page script.
        status: Build status used to badge the module (e.g. "planned").
    """

    key: str
    title: str
    subtitle: str
    description: str
    learning_outcomes: list[str]
    section: str
    file: str
    help_text: str = field(default="")
    status: str = field(default="planned")


APP_TITLE = "Data Science Lab"
APP_SUBTITLE = "An interactive learning workspace for BS Data Science students"
APP_TAGLINE = (
    "Explore, analyze, and model real-world data with an intuitive "
    "point-and-click interface built on Streamlit."
)
APP_VERSION = "0.1.0"

PROJECT_REPO = "https://github.com/your-org/data-science-lab"
PROJECT_DOCS = "https://docs.streamlit.io"

# Learning stages shown on the dashboard roadmap.
LEARNING_STAGES = [
    ("Understand", "Inspect and explore datasets", ["Dataset Explorer", "EDA"]),
    ("Prepare", "Clean and engineer features", ["Data Preprocessing", "Feature Engineering"]),
    ("Model", "Train predictive and grouping models", ["Classification", "Regression", "Clustering"]),
    ("Evaluate", "Measure and compare performance", ["Model Evaluation", "Model Comparison"]),
    ("Automate", "Let AutoML do the heavy lifting", ["AutoML"]),
]

MODULES: list[Module] = [
    Module(
        key="home",
        title="Home",
        subtitle="Your dashboard for hands-on data science learning",
        description=(
            "A guided, interactive workspace that walks BS Data Science students "
            "through the full data science workflow: understanding data, "
            "preparing it, modeling, evaluating, and automating."
        ),
        learning_outcomes=[
            "Navigate the modules that map to your data science curriculum",
            "Understand where each stage fits in the end-to-end workflow",
            "Practice data science concepts interactively with real datasets",
        ],
        section="Getting Started",
        file="pages/Home.py",
        status="live",
    ),
    Module(
        key="dataset-explorer",
        title="Dataset Explorer",
        subtitle="Load and inspect datasets with confidence",
        description=(
            "Learn how to bring data into the lab, understand its structure, "
            "and spot quality issues before any analysis begins."
        ),
        learning_outcomes=[
            "Upload datasets in CSV, Excel, or JSON format",
            "Read the schema: data types, shape, and columns",
            "Detect missing values and duplicates",
            "Preview rows and compute quick column statistics",
        ],
        section="Data Understanding",
        file="pages/1_Dataset_Explorer.py",
        status="implemented",
        help_text=(
            "Pick a dataset file, then use the preview and summary views to "
            "understand what you are working with before moving on."
        ),
    ),
    Module(
        key="eda",
        title="EDA",
        subtitle="Uncover patterns with statistics and visuals",
        description=(
            "Learn how to summarize and visualize datasets to reveal "
            "distributions, relationships, and anomalies that guide your "
            "modeling decisions."
        ),
        learning_outcomes=[
            "Compute descriptive statistics per column",
            "Visualize distributions with histograms and box plots",
            "Explore categorical data with count charts",
            "Interpret correlation heatmaps",
            "Spot trends with pair plots",
        ],
        section="Data Understanding",
        file="pages/2_EDA.py",
        status="implemented",
        help_text=(
            "Choose a dataset and the chart types you want. The visualizations "
            "update automatically as you change the options."
        ),
    ),
    Module(
        key="data-preprocessing",
        title="Data Preprocessing",
        subtitle="Clean and transform raw data",
        description=(
            "Learn how to turn messy, raw data into a clean, analysis-ready "
            "format using proven cleaning and transformation steps."
        ),
        learning_outcomes=[
            "Handle missing values by dropping or imputing",
            "Remove or cap outliers safely",
            "Encode categorical variables for machine learning",
            "Scale and normalize numeric features",
            "Split data into training and test sets",
        ],
        section="Data Preparation",
        file="pages/3_Data_Preprocessing.py",
        status="implemented",
        help_text=(
            "Follow the steps from top to bottom: clean, transform, then split. "
            "Each step previews its effect on the data."
        ),
    ),
    Module(
        key="feature-engineering",
        title="Feature Engineering",
        subtitle="Create features that help models learn",
        description=(
            "Learn how to create and select informative features so machine "
            "learning models can learn from the data more effectively."
        ),
        learning_outcomes=[
            "Create derived and polynomial features",
            "Extract useful features from datetime columns",
            "Bin and bucket numeric values",
            "Evaluate feature importance",
            "Select the most informative features",
        ],
        section="Data Preparation",
        file="pages/4_Feature_Engineering.py",
        help_text=(
            "Start from a cleaned dataset, then add new features and inspect "
            "how they change the data."
        ),
    ),
    Module(
        key="classification",
        title="Classification",
        subtitle="Predict categories with machine learning",
        description=(
            "Learn how to train classifiers that assign observations to "
            "categories, such as detecting spam or predicting a diagnosis."
        ),
        learning_outcomes=[
            "Prepare data for classification tasks",
            "Train classifiers such as logistic regression, trees, and ensembles",
            "Understand model decision boundaries",
            "Evaluate with accuracy, precision, recall, and F1",
        ],
        section="Modeling",
        file="pages/5_Classification.py",
        help_text=(
            "Choose a dataset with a categorical target, select a classifier, "
            "then train and evaluate it."
        ),
    ),
    Module(
        key="regression",
        title="Regression",
        subtitle="Predict continuous numeric outcomes",
        description=(
            "Learn how to model continuous outcomes such as prices or demand, "
            "and interpret how well your model fits the data."
        ),
        learning_outcomes=[
            "Train regressors such as linear, ridge, and tree models",
            "Understand residuals and prediction plots",
            "Evaluate with MAE, MSE, RMSE, and R2",
            "Compare models on validation data",
        ],
        section="Modeling",
        file="pages/6_Regression.py",
        help_text=(
            "Choose a dataset with a numeric target, pick a regressor, and "
            "train it to inspect its predictions."
        ),
    ),
    Module(
        key="model-evaluation",
        title="Model Evaluation",
        subtitle="Measure model quality the right way",
        description=(
            "Learn how to assess models with appropriate metrics and "
            "visualizations so you can trust your results and make fair "
            "comparisons."
        ),
        learning_outcomes=[
            "Apply classification metrics and confusion matrices",
            "Interpret ROC and precision-recall curves",
            "Use regression metrics and residual analysis",
            "Run cross-validation",
            "Read learning curves",
        ],
        section="Evaluation",
        file="pages/7_Model_Evaluation.py",
        help_text=(
            "Run an evaluation on a trained model and explore each metric to "
            "understand what it tells you."
        ),
    ),
    Module(
        key="clustering",
        title="Clustering",
        subtitle="Discover groups in unlabeled data",
        description=(
            "Learn how unsupervised learning finds natural structure in data, "
            "grouping similar observations together."
        ),
        learning_outcomes=[
            "Apply K-Means and hierarchical clustering",
            "Choose the number of clusters with elbow and silhouette analysis",
            "Visualize clusters with PCA projections",
            "Interpret cluster profiles and centroids",
        ],
        section="Modeling",
        file="pages/8_Clustering.py",
        help_text=(
            "Load a dataset without a target column, run clustering, and "
            "explore the discovered groups."
        ),
    ),
    Module(
        key="model-comparison",
        title="Model Comparison",
        subtitle="Pick the best model systematically",
        description=(
            "Learn how to train several models side by side and rank them on a "
            "shared evaluation framework to make data-driven choices."
        ),
        learning_outcomes=[
            "Run multiple models on the same dataset",
            "Compare metrics in a side-by-side table",
            "Visualize performance with bars and box plots",
            "Select and export the best model",
        ],
        section="Evaluation",
        file="pages/9_Model_Comparison.py",
        help_text=(
            "Select the models you want to compare, then review the ranking "
            "table and charts."
        ),
    ),
    Module(
        key="automl",
        title="AutoML",
        subtitle="Automate the machine learning pipeline",
        description=(
            "Learn how automated machine learning searches over pipelines and "
            "hyperparameters to find strong models with minimal configuration."
        ),
        learning_outcomes=[
            "Detect the task type automatically",
            "Automate preprocessing pipelines",
            "Search hyperparameters over several algorithms",
            "Export the best model and its report",
        ],
        section="Automation",
        file="pages/10_AutoML.py",
        help_text=(
            "Upload a dataset, set a time budget, and let AutoML find a strong "
            "pipeline for you."
        ),
    ),
]

# Convenience accessors ------------------------------------------------------

MODULES_BY_TITLE: dict[str, Module] = {module.title: module for module in MODULES}
MODULES_BY_KEY: dict[str, Module] = {module.key: module for module in MODULES}

DATA_MODULES: list[Module] = [module for module in MODULES if module.key != "home"]

NAV_SECTIONS: list[str] = list(dict.fromkeys(module.section for module in MODULES))


def get_module(title: str) -> Module:
    """Return the module with the given display title.

    Raises:
        KeyError: If no module matches the provided title.
    """
    return MODULES_BY_TITLE[title]
