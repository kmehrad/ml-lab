"""Training pipeline entry point for Spaceship Titanic.

Ties together :mod:`data`, :mod:`features`, :mod:`models`, and :mod:`evaluate`
to train a classifier, evaluate it, and write a Kaggle submission to
``data/submissions``.

Nothing is implemented yet — this is a placeholder entry point.
"""
from __future__ import annotations


def main() -> None:
    """Run the end-to-end training pipeline."""
    # TODO: load data -> engineer features -> fit model -> cross-validate
    #       -> predict on test -> write submission CSV.
    raise NotImplementedError


if __name__ == "__main__":
    main()
