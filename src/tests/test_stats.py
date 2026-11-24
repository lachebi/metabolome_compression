"""Known-answer test for the shared Benjamini-Hochberg function. Run: pytest tests

4 copies of this formula existed independently in analysis/ scripts before being
consolidated into metabo.stats.bh (2026-11-22); this guards against the one shared
copy silently diverging from a hand-verified answer in the future.
"""
import numpy as np
import pytest

from metabo.stats import bh


def test_bh_known_answer():
    """p = [0.01, 0.011, 0.012, 0.5] (already sorted): raw q = p*n/rank =
    [0.04, 0.022, 0.016, 0.5], non-monotonic, so the running-minimum step must pull
    the first three down to the smallest later value, 0.016."""
    q = bh([0.01, 0.011, 0.012, 0.5])
    assert q == pytest.approx([0.016, 0.016, 0.016, 0.5])


def test_bh_preserves_input_order():
    """Same 4 p-values, shuffled; q must come back in the same (shuffled) order,
    not sorted order."""
    q = bh([0.5, 0.01, 0.012, 0.011])
    assert q == pytest.approx([0.5, 0.016, 0.016, 0.016])


def test_bh_clips_to_one():
    q = bh([0.9, 0.95, 0.99, 1.0])
    assert np.all(q <= 1.0)
