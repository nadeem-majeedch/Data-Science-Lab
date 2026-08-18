# Data Science Lab

An educational, open-source web application built with **Python** and
**Streamlit** that provides BS Data Science students with an interactive
workspace for exploring, cleaning, modeling, and evaluating data.

This repository currently contains the **project foundation**: a professional
landing page, a modular structure, and placeholder modules for the full data
science workflow. Machine learning functionality will be implemented in later
milestones.

## Project purpose

Textbooks explain concepts; this lab lets you practice them. Instead of writing
boilerplate plotting and modeling code from scratch, students can load a
dataset and interactively run the key steps of a data science project through a
clean web interface. The app doubles as a teaching aid for instructors.

## Features

- Professional landing page tailored for BS Data Science students
- Sidebar navigation powered by Streamlit's multipage structure
- Modular, readable codebase split into `pages/`, `utils/`, and `datasets/`
- Placeholder modules for the entire learning roadmap:
  - Dataset Explorer
  - EDA (Exploratory Data Analysis)
  - Data Preprocessing
  - Feature Engineering
  - Classification
  - Regression
  - Model Evaluation
  - Clustering
  - Model Comparison
  - AutoML
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
├── pages/                 # Multipage modules (10 placeholders)
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
│   ├── config.py          # App constants and page registry
│   └── placeholder.py     # Reusable placeholder layout component
├── datasets/              # Data files (large files git-ignored)
├── notebooks/             # Jupyter notebooks for exploratory work
├── reports/               # Generated reports and exports
└── tests/                 # Smoke tests
    └── test_app.py
```

## Intended audience

- **BS Data Science students** who want hands-on practice with core concepts:
  data inspection, cleaning, feature engineering, modeling, and evaluation.
- **Instructors** who need a ready-made, extensible demo environment for
  lectures and lab sessions.
- **Open-source contributors** who want to help build an educational tool
  where each module is small, readable, and easy to extend.

## Roadmap

1. **Foundation** (current): landing page, structure, and placeholders.
2. **Dataset Explorer + EDA**: upload, preview, and visualize datasets.
3. **Preprocessing + Feature Engineering**: cleaning, encoding, scaling.
4. **Modeling modules**: classification, regression, clustering.
5. **Evaluation + Comparison + AutoML**: metrics, ranking, automated search.

## License

Open-source. See the license file added in a future release.
