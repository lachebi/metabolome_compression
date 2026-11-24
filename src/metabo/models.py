"""Classifiers and their hyperparameter grids (keys prefixed `clf__`, the pipeline step name).

LR and RF are class-weighted; kNN is not.
"""
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def _sklearn_ge(major, minor):
    p = sklearn.__version__.split(".")
    return (int(p[0]), int(p[1])) >= (major, minor)


def get_classifier(name, random_state=42, n_jobs=-1):
    """n_jobs is the classifier's own threading; use 1 inside a process pool."""
    if name == "lr":
        # Elastic net: scikit-learn >= 1.8 infers it from l1_ratio, older versions
        # need penalty="elasticnet". Both give the same model.
        common = dict(solver="saga", l1_ratio=0.5, C=1.0, class_weight="balanced",
                      max_iter=500, tol=1e-2, random_state=random_state)
        if _sklearn_ge(1, 8):
            return LogisticRegression(**common)
        return LogisticRegression(penalty="elasticnet", **common)
    if name == "rfc":
        return RandomForestClassifier(class_weight="balanced", n_jobs=n_jobs,
                                      random_state=random_state)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=n_jobs)
    raise ValueError(f"unknown model: {name}")


PARAM_GRIDS = {
    "lr":  {"clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.1, 0.5, 0.9]},
    "rfc": {"clf__max_depth": [3, 5, 10], "clf__n_estimators": [200, 400],
            "clf__min_samples_leaf": [1, 5]},
    "knn": {"clf__n_neighbors": [5, 15, 25], "clf__weights": ["uniform", "distance"]},
}
