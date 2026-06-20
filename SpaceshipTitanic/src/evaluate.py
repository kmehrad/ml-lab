"""Metrics and cross-validation utilities for Spaceship Titanic.

The competition is scored on classification **accuracy**. This module will hold
the scoring helpers and cross-validation routines used during model selection.

Nothing is implemented yet — these are placeholders.
"""
from __future__ import annotations


def accuracy(y_true, y_pred) -> float:
    """Return classification accuracy for predicted vs. true labels."""
    # TODO: compute and return accuracy.
    raise NotImplementedError


def cross_validate(model, X, y, n_splits: int = 5):
    """Run stratified k-fold cross-validation and return per-fold scores."""
    # TODO: implement StratifiedKFold CV reporting accuracy.
    raise NotImplementedError
