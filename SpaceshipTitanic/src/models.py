"""Model definitions for Spaceship Titanic (binary classification).

Each factory will return a fresh scikit-learn classifier. Optional gradient
boosting libraries (XGBoost / LightGBM) can be added later and imported lazily
so the project still works without them.

Nothing is implemented yet — these are placeholders.
"""
from __future__ import annotations


def get_model(name: str = "logreg"):
    """Return a fresh classifier estimator by name.

    Planned baselines: logistic regression, random forest, gradient boosting.
    """
    # TODO: map ``name`` to a configured sklearn estimator and return it.
    raise NotImplementedError


def available_models():
    """List the model names that can be built in the current environment."""
    # TODO: return the set of supported model names.
    raise NotImplementedError
