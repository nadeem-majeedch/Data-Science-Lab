"""Model registry and educational content for the Classification module.

Every classifier is described by the same metadata shape so the page can
render parameter controls, explainer text, and capability flags from a single
source of truth. Keeping the metadata here (instead of in the page) makes it
trivial to unit-test and to reuse in later modules such as Model Comparison.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODELS: dict[str, dict] = {
    "Logistic Regression": {
        "model_class": LogisticRegression,
        "requires_scaling": True,
        "supports_importance": True,
        "why": (
            "Despite its name, logistic regression is a **linear classifier**, "
            "not a regression method. It models the log-odds of belonging to a "
            "class as a linear combination of the features and turns that into "
            "a probability with the sigmoid function."
        ),
        "key_parameters": (
            "- **C**: inverse of the regularization strength. Smaller C means "
            "stronger regularization and a simpler decision boundary.\n"
            "- **max_iter**: how many iterations the optimization solver is "
            "allowed before it must converge."
        ),
        "advantages": (
            "- Fast to train and predict.\n"
            "- Highly interpretable: coefficients tell you the direction and "
            "size of each feature's effect.\n"
            "- Returns well-calibrated probabilities.\n"
            "- Works well when there are many features."
        ),
        "limitations": (
            "- Assumes an (approximately) linear decision boundary.\n"
            "- Sensitive to correlated features and outliers.\n"
            "- Requires features to be scaled for stable training."
        ),
        "when_to_use": (
            "A strong first model, and the go-to when you need to explain "
            "predictions - for example credit scoring or churn prediction."
        ),
        "params": [
            {
                "name": "max_iter",
                "label": "Maximum iterations",
                "type": "int",
                "min": 100,
                "max": 5000,
                "value": 1000,
                "step": 100,
                "help": "Maximum number of solver iterations. Raise it if the "
                "model warns that it did not converge.",
            },
            {
                "name": "C",
                "label": "Regularization strength (C)",
                "type": "float",
                "min": 0.01,
                "max": 10.0,
                "value": 1.0,
                "step": 0.1,
                "help": "Inverse regularization strength. Smaller values force "
                "a simpler, more regularized model.",
            },
        ],
    },
    "K-Nearest Neighbors": {
        "model_class": KNeighborsClassifier,
        "requires_scaling": True,
        "supports_importance": False,
        "why": (
            "K-nearest neighbors (KNN) predicts a new point by the **majority "
            "vote of its k nearest training points** in feature space. It is "
            "instance-based: no model is really 'learned', the training data "
            "itself is the model."
        ),
        "key_parameters": (
            "- **n_neighbors (k)**: how many neighbors vote. Odd numbers are "
            "commonly chosen to avoid ties; smaller k means a more flexible "
            "boundary, larger k a smoother one.\n"
            "- **weights**: ``uniform`` gives every neighbor one vote; "
            "``distance`` weights votes by how close the neighbor is."
        ),
        "advantages": (
            "- Simple and easy to understand.\n"
            "- No explicit training phase.\n"
            "- Can capture very irregular decision boundaries.\n"
            "- Naturally supports multi-class problems."
        ),
        "limitations": (
            "- Prediction is slow on large datasets (must scan all points).\n"
            "- Highly sensitive to feature scale - features must be scaled.\n"
            "- Suffers from the curse of dimensionality.\n"
            "- Stores the whole training set in memory."
        ),
        "when_to_use": (
            "Small to medium datasets with low dimensionality, or when the "
            "boundary between classes is highly irregular."
        ),
        "params": [
            {
                "name": "n_neighbors",
                "label": "Number of neighbors (k)",
                "type": "int",
                "min": 1,
                "max": 20,
                "value": 5,
                "step": 1,
                "help": "Number of nearest neighbors that vote on each "
                "prediction.",
            },
            {
                "name": "weights",
                "label": "Voting weights",
                "type": "choice",
                "options": [("uniform", "uniform"), ("distance", "distance")],
                "value": "uniform",
                "help": "How to weight the neighbors' votes.",
            },
        ],
    },
    "Decision Tree": {
        "model_class": DecisionTreeClassifier,
        "requires_scaling": False,
        "supports_importance": True,
        "why": (
            "A decision tree recursively splits the data on the feature that "
            "best separates the classes, producing a flowchart of if-else "
            "rules you can literally read and follow by hand."
        ),
        "key_parameters": (
            "- **criterion**: the split quality measure - ``gini`` impurity or "
            "``entropy`` (information gain).\n"
            "- **max_depth**: how deep the tree may grow. Deeper trees fit the "
            "training data better but overfit more easily.\n"
            "- **min_samples_split**: minimum samples required to split a node."
        ),
        "advantages": (
            "- Extremely interpretable - great for communicating rules.\n"
            "- Handles mixed numeric and categorical data.\n"
            "- No feature scaling needed.\n"
            "- Captures non-linear relationships."
        ),
        "limitations": (
            "- Prone to overfitting without depth limits.\n"
            "- Unstable: small changes in data can change the whole tree.\n"
            "- Greedy splitting can miss a globally optimal tree."
        ),
        "when_to_use": (
            "When interpretability matters most, or as a building block inside "
            "ensembles such as random forests and gradient boosting."
        ),
        "params": [
            {
                "name": "criterion",
                "label": "Split criterion",
                "type": "choice",
                "options": [("gini", "gini"), ("entropy", "entropy")],
                "value": "gini",
                "help": "How the tree measures the quality of a split.",
            },
            {
                "name": "max_depth",
                "label": "Max depth (0 = unlimited)",
                "type": "int",
                "min": 0,
                "max": 30,
                "value": 0,
                "step": 1,
                "none_allowed": True,
                "help": "Maximum tree depth. 0 lets the tree grow until all "
                "leaves are pure (risks overfitting).",
            },
            {
                "name": "min_samples_split",
                "label": "Min samples per split",
                "type": "int",
                "min": 2,
                "max": 20,
                "value": 2,
                "step": 1,
                "help": "Minimum number of samples required to split a node. "
                "Higher values produce simpler trees.",
            },
        ],
    },
    "Random Forest": {
        "model_class": RandomForestClassifier,
        "requires_scaling": False,
        "supports_importance": True,
        "why": (
            "A random forest trains **many decision trees**, each on a random "
            "bootstrap sample of the rows and a random subset of the features, "
            "then averages their votes. Each tree sees a slightly different "
            "view of the data, which makes the ensemble far more robust."
        ),
        "key_parameters": (
            "- **n_estimators**: the number of trees in the forest. More trees "
            "usually mean better, more stable performance at the cost of speed.\n"
            "- **max_depth**: per-tree depth limit (0 = unlimited), a key "
            "overfitting control."
        ),
        "advantages": (
            "- Very robust and often a strong default for tabular data.\n"
            "- Reduces the overfitting of a single tree.\n"
            "- Handles mixed features and, to a degree, missing values.\n"
            "- Provides built-in feature importances."
        ),
        "limitations": (
            "- Much less interpretable than a single tree.\n"
            "- Slower to train and predict, and uses more memory.\n"
            "- Can still overfit on small or very noisy datasets."
        ),
        "when_to_use": (
            "A reliable, general-purpose baseline for almost any "
            "classification problem on tabular data."
        ),
        "params": [
            {
                "name": "n_estimators",
                "label": "Number of trees",
                "type": "int",
                "min": 10,
                "max": 500,
                "value": 100,
                "step": 10,
                "help": "Number of decision trees in the forest.",
            },
            {
                "name": "max_depth",
                "label": "Max depth (0 = unlimited)",
                "type": "int",
                "min": 0,
                "max": 30,
                "value": 0,
                "step": 1,
                "none_allowed": True,
                "help": "Maximum depth of each tree. 0 lets trees grow fully.",
            },
        ],
    },
    "Naive Bayes": {
        "model_class": GaussianNB,
        "requires_scaling": False,
        "supports_importance": False,
        "why": (
            "Naive Bayes applies **Bayes' theorem** to classification, assuming "
            "that every feature is conditionally independent of the others "
            "given the class. Gaussian Naive Bayes models each numeric feature "
            "as a normal (Gaussian) distribution per class."
        ),
        "key_parameters": (
            "This implementation exposes **var_smoothing**, which adds a small "
            "value to the variance of every feature to keep probability "
            "estimates stable. The default is almost always fine."
        ),
        "advantages": (
            "- Extremely fast to train and predict.\n"
            "- Needs very little data to produce a working model.\n"
            "- Scales well to high-dimensional problems.\n"
            "- A great cheap baseline."
        ),
        "limitations": (
            "- The independence assumption rarely holds in real data.\n"
            "- Poor with strongly correlated features.\n"
            "- Probabilities can be overconfident."
        ),
        "when_to_use": (
            "Text classification, spam filtering, and other high-dimensional "
            "problems where speed matters and the baseline should be solid."
        ),
        "params": [],
    },
    "Support Vector Machine": {
        "model_class": SVC,
        "requires_scaling": True,
        "supports_importance": True,
        "why": (
            "A support vector machine finds the **hyperplane that best "
            "separates the classes**, maximizing the margin between the "
            "classes. Kernels map the data into a higher-dimensional space so "
            "the separator can also be non-linear."
        ),
        "key_parameters": (
            "- **kernel**: the similarity function used. ``rbf`` is a good "
            "general default; ``linear`` is best for high-dimensional or "
            "linearly separable data.\n"
            "- **C**: regularization trade-off between a wide margin and "
            "avoiding misclassifications."
        ),
        "advantages": (
            "- Effective in high-dimensional spaces.\n"
            "- Works well on small datasets with a clear margin.\n"
            "- Very flexible thanks to kernels."
        ),
        "limitations": (
            "- Sensitive to feature scale - scaling is essential.\n"
            "- Does not scale well to very large datasets.\n"
            "- Harder to interpret than linear or tree models."
        ),
        "when_to_use": (
            "Small to medium datasets with a fairly clean boundary - for "
            "example precomputed feature vectors from text or images."
        ),
        "params": [
            {
                "name": "kernel",
                "label": "Kernel",
                "type": "choice",
                "options": [
                    ("rbf", "rbf"),
                    ("linear", "linear"),
                    ("poly", "poly"),
                    ("sigmoid", "sigmoid"),
                ],
                "value": "rbf",
                "help": "Similarity function that defines the feature space.",
            },
            {
                "name": "C",
                "label": "Regularization strength (C)",
                "type": "float",
                "min": 0.01,
                "max": 10.0,
                "value": 1.0,
                "step": 0.1,
                "help": "Trade-off between a wide margin and fewer "
                "misclassifications.",
            },
        ],
        "fixed_kwargs": {"probability": True},
    },
    "Gradient Boosting": {
        "model_class": GradientBoostingClassifier,
        "requires_scaling": False,
        "supports_importance": True,
        "why": (
            "Gradient boosting trains **weak decision trees one after another**, "
            "each one focused on correcting the mistakes of all the previous "
            "trees (an iterative form of gradient descent in function space)."
        ),
        "key_parameters": (
            "- **n_estimators**: number of boosting stages; more stages mean a "
            "more expressive model.\n"
            "- **learning_rate**: how strongly each tree updates the model. "
            "Smaller values are slower but usually more accurate.\n"
            "- **max_depth**: trees are typically kept shallow (2-4) to avoid "
            "overfitting."
        ),
        "advantages": (
            "- Often the most accurate out-of-the-box method for tabular data.\n"
            "- Handles mixed features and provides feature importances.\n"
            "- Usually beats random forests with careful tuning."
        ),
        "limitations": (
            "- Many hyperparameters to tune.\n"
            "- Sequential training makes it slower to fit.\n"
            "- Overfits easily when the learning rate is too high."
        ),
        "when_to_use": (
            "Structured/tabular problems where predictive accuracy matters "
            "most - the classic winner in tabular competitions."
        ),
        "params": [
            {
                "name": "n_estimators",
                "label": "Number of boosting stages",
                "type": "int",
                "min": 10,
                "max": 300,
                "value": 100,
                "step": 10,
                "help": "Number of boosting rounds.",
            },
            {
                "name": "learning_rate",
                "label": "Learning rate",
                "type": "float",
                "min": 0.01,
                "max": 1.0,
                "value": 0.1,
                "step": 0.05,
                "help": "How strongly each tree contributes. Smaller is slower "
                "but often more accurate.",
            },
            {
                "name": "max_depth",
                "label": "Max depth (0 = unlimited)",
                "type": "int",
                "min": 0,
                "max": 10,
                "value": 3,
                "step": 1,
                "none_allowed": True,
                "help": "Maximum depth of each weak tree. Shallow trees are "
                "typically best.",
            },
        ],
    },
}

MODEL_KEYS: list[str] = list(MODELS.keys())


def get_model(model_key: str) -> dict:
    """Return the metadata dict for a classifier by its display name."""
    return MODELS[model_key]


def default_classifier_params(model_key: str) -> dict:
    """Return ``{param_name: default_value}`` for a classifier."""
    return {param["name"]: param["value"] for param in MODELS[model_key]["params"]}


def build_classifier(model_key: str, values: dict | None = None) -> object:
    """Construct an unfitted sklearn classifier from user-provided parameter values.

    Args:
        model_key: Display name of the model in ``MODELS``.
        values: Optional mapping of parameter name -> value. Any missing
            parameter falls back to its spec default. For parameters flagged
            ``none_allowed``, a value of ``0`` is translated to ``None`` (the
            "unlimited" choice).
    """
    spec = MODELS[model_key]
    values = values or {}
    kwargs = dict(spec.get("fixed_kwargs", {}))
    for param in spec["params"]:
        value = values.get(param["name"], param["value"])
        if param.get("none_allowed") and value == 0:
            value = None
        kwargs[param["name"]] = value
    return spec["model_class"](**kwargs)


def classifier_constructor_code(model_key: str, values: dict | None = None) -> str:
    """Return the Python snippet that builds the chosen classifier.

    ``none_allowed`` parameters whose value is 0 are omitted entirely so the
    generated code uses sklearn's native ``None`` default.
    """
    spec = MODELS[model_key]
    values = values or {}
    kwargs = dict(spec.get("fixed_kwargs", {}))
    for param in spec["params"]:
        value = values.get(param["name"], param["value"])
        if param.get("none_allowed") and value == 0:
            continue
        kwargs[param["name"]] = value
    arguments = ", ".join(f"{name}={value!r}" for name, value in kwargs.items())
    return f"{spec['model_class'].__name__}({arguments})"
