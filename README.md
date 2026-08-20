# Data Science Lab

An educational, open-source web application built with **Python** and
**Streamlit** that provides BS Data Science students with an interactive
workspace for exploring, cleaning, modeling, and evaluating data.

This repository contains a growing, modular data science workspace: a landing
page, six fully implemented modules, and placeholders for the remaining
learning modules. Machine learning functionality is implemented incrementally.

## Project purpose

Textbooks explain concepts; this lab lets you practice them. Instead of writing
boilerplate plotting and modeling code from scratch, students can load a
dataset and interactively run the key steps of a data science project through a
clean web interface. The app doubles as a teaching aid for instructors.

## Features

- Professional landing page tailored for BS Data Science students
- Sidebar navigation powered by Streamlit's multipage structure
- Modular, readable codebase split into `pages/`, `utils/`, and `datasets/`
- **Dataset Explorer** (implemented): upload CSV/Excel files or load bundled
  sample datasets, inspect structure, preview rows (head/tail/sample), explore
  numeric and categorical statistics, run data-quality checks, and download
  the dataset
- **EDA** (implemented): automatic EDA summary plus interactive Plotly charts
  for numeric (histogram, box, density, scatter, correlation) and categorical
  (bar, frequency distribution, count plot) columns, missing-value and
  pairwise analysis, with an interpretation section that clearly labels
  automatically generated observations as educational hints
- **Data Preprocessing** (implemented): step-by-step cleaning workflow for
  missing values (drop or impute with mean/median/mode/constant), duplicate
  rows, IQR outlier detection and removal, categorical encoding (one-hot /
  label), numeric scaling (Standard/MinMax/Robust), and a train/test split.
  Every step shows its equivalent Python code, before/after statistics are
  compared, the cleaned dataset can be downloaded, and a reusable sklearn
  ``ColumnTransformer`` is built - fitted on the training set only - so later
  modeling modules transform new data without data leakage
- **Feature Engineering** (implemented): chainable, undoable operations that
  enrich the dataset - new numeric features (sum/difference/product/ratio),
  math transforms (log/sqrt/square), binning (equal width/quantile),
  date/time extraction (year/month/day/weekday), text features (length/word
  count), interaction and polynomial features, plus feature selection by
  variance threshold or correlation with a target. Every operation explains
  itself, previews the new columns, shows its Python code, and can be undone
  or reset. When a modeling module trains a model, its feature importances
  are displayed here
- **Classification** (implemented): train and evaluate classifiers on a
  categorical target. Pick a target and numeric/categorical features (text
  columns are one-hot encoded automatically), configure a leak-free
  train/test split, and choose one of seven algorithms (logistic regression,
  k-nearest neighbors, decision tree, random forest, naive Bayes, support
  vector machine, gradient boosting) with tunable parameters. Results include
  accuracy, precision, recall, F1, a per-class classification report, an
  interactive confusion matrix, feature importances (where supported), a
  prediction playground (sample rows or hand-entered values), and the full
  equivalent Python code. Every algorithm ships with an explainer covering
  why it works, its key parameters, advantages, limitations, and when to use
  it
- **Regression** (implemented): train and evaluate regressors on a continuous
  numeric target. Pick a target and features (text columns are one-hot
  encoded automatically), configure a leak-free train/test split, and choose
  one of seven algorithms (linear, ridge, lasso, decision tree, random
  forest, gradient boosting, KNN) with tunable parameters. Results include
  MAE, MSE, RMSE, R2, an actual-vs-predicted scatter plot with an ideal-fit
  line, a residual plot, an actual/predicted/residual prediction table,
  feature importances (where supported), and the full equivalent Python code
- Placeholder modules for the rest of the learning roadmap:
  - Model Evaluation
  - Clustering
  - Model Comparison
  - AutoML
- Educational explanations embedded in every analysis section
- Smoke tests that verify every page renders without errors

## Installation

Requires Python 3.9+.

```bash
# Clone the repository
git clone <repository-url>
cd Data-Science-Lab

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) install test tooling
pip install pytest
```

## Local execution

```bash
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`. From the sidebar
you can navigate between the landing page and the placeholder modules.

### Running the tests

```bash
pytest
```

The tests use Streamlit's `AppTest` harness to launch each page headlessly and
assert that it renders without exceptions.

## Project structure

```
Data-Science-Lab/
├── README.md              # This file
├── requirements.txt       # Runtime dependencies
├── app.py                 # Landing page (Streamlit entry point)
├── pages/                 # Multipage modules (6 implemented, 5 planned)
│   ├── 1_Dataset_Explorer.py
│   ├── 2_EDA.py
│   ├── 3_Data_Preprocessing.py
│   ├── 4_Feature_Engineering.py
│   ├── 5_Classification.py
│   ├── 6_Regression.py
│   ├── 7_Model_Evaluation.py
│   ├── 8_Clustering.py
│   ├── 9_Model_Comparison.py
│   └── 10_AutoML.py
├── utils/                 # Shared helpers and app configuration
│   ├── __init__.py
│   ├── config.py          # App constants and module registry
│   ├── navigation.py      # Grouped sidebar navigation builder
│   ├── ui.py              # Reusable UI components
│   ├── session.py         # Session-state helpers for the active dataset
│   ├── data_loader.py     # CSV / Excel loading (uploads + samples)
│   ├── data_analysis.py   # Pure dataset analysis functions
│   ├── visualization.py   # Plotly chart builders + interpretation hints
│   ├── preprocessing.py   # Cleaning, encoding, scaling, split, pipeline
│   ├── feature_engineering.py  # Numeric/text/date features, selection, importance
│   ├── models.py          # Classification model registry + educational content
│   ├── model_training.py  # Leak-free pipeline training, metrics, prediction
│   ├── regressors.py      # Regression model registry + educational content
│   ├── regression_training.py  # Leak-free regression training, metrics, code
│   └── placeholder.py     # Backwards-compatible placeholder helper
├── datasets/              # Data files (large files git-ignored)
│   └── samples/           # Bundled sample datasets (CSV + XLSX)
├── notebooks/             # Jupyter notebooks for exploratory work
├── reports/               # Generated reports and exports
└── tests/                 # Smoke tests
    ├── test_app.py
    ├── test_data_loader.py
    ├── test_data_analysis.py
    ├── test_preprocessing.py
    ├── test_feature_engineering.py
    ├── test_models.py
    ├── test_model_training.py
    ├── test_regressors.py
    └── test_regression_training.py
```

## Intended audience

- **BS Data Science students** who want hands-on practice with core concepts:
  data inspection, cleaning, feature engineering, modeling, and evaluation.
- **Instructors** who need a ready-made, extensible demo environment for
  lectures and lab sessions.
- **Open-source contributors** who want to help build an educational tool
  where each module is small, readable, and easy to extend.

## Roadmap

1. **Foundation** (done): landing page, structure, and placeholders.
2. **Dataset Explorer** (done): upload, preview, statistics, data quality.
3. **EDA** (done): interactive visualizations to explore datasets.
4. **Preprocessing** (done): cleaning, encoding, scaling, split, pipeline.
5. **Feature Engineering** (done): create, transform, bin, and select features.
6. **Modeling modules** (in progress): classification and regression are
   done; clustering remains.
7. **Evaluation + Comparison + AutoML**: metrics, ranking, automated search.

## License

Open-source. See the license file added in a future release.
