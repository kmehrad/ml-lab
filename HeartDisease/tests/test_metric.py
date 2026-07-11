"""Tests for the ROC AUC metric — the single source of truth for scoring."""
from __future__ import annotations

import numpy as np

from src.metric import roc_auc


def test_roc_auc_perfect_separation_is_one():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(y, p) == 1.0


def test_roc_auc_inverted_is_zero():
    y = [0, 0, 1, 1]
    p = [0.9, 0.8, 0.2, 0.1]
    assert roc_auc(y, p) == 0.0


def test_roc_auc_random_guess_is_half():
    y = [0, 1, 0, 1]
    p = [0.5, 0.5, 0.5, 0.5]
    assert roc_auc(y, p) == 0.5


def test_roc_auc_shape_agnostic():
    # accepts lists / column vectors / row vectors identically
    y, p = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    a = roc_auc(y, p)
    b = roc_auc(np.array(y).reshape(-1, 1), np.array(p))
    assert np.isclose(a, b) and np.isclose(a, 1.0)
