"""Known-answer tests for the attribution mappings. Run: pytest tests"""
import os
import sys

import numpy as np
import pytest
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from explainability_stage2 import backproject, backproject_importance, check_pca_weights  # noqa: E402
from metabo.models import get_classifier  # noqa: E402


def test_opposite_signs_cancel():
    """Metabolite 0 loads +1 on one component and -1 on the other; with equal
    coefficients its exact weight is zero. Summing absolute values would rank it first."""
    pca = PCA(2)
    pca.components_ = np.array([[1.0, 1.0, 0.0], [-1.0, 0.5, 0.2]]) / np.sqrt(2)
    b = np.array([1.0, 1.0])
    w = backproject("pca", pca, None, b, None)
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert np.argmax(w) == 1
    assert np.argmax(np.abs(pca.components_).T @ np.abs(b)) == 0


def test_pca_weights_reproduce_classifier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 30))
    y = (X[:, 0] - X[:, 1] + rng.normal(size=200) > 0).astype(int)
    pca = PCA(5).fit(X)
    clf = get_classifier("lr", random_state=0, n_jobs=1).fit(pca.transform(X), y)
    check_pca_weights(pca, clf, X)


def test_negative_importance_is_noise():
    pca = PCA(2)
    pca.components_ = np.eye(2, 3)
    imp = backproject_importance("pca", pca, None, np.array([0.3, -0.2]), None)
    assert imp[1] == 0.0 and imp[0] == pytest.approx(0.3)
