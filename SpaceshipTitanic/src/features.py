"""Preprocessing and feature engineering for Spaceship Titanic.

This module will transform the raw passenger records into model-ready features.
Nothing is implemented yet — the functions below are placeholders that describe
the intended steps.

Planned feature work:
- Split ``Cabin`` ("deck/num/side") into separate ``Deck``, ``CabinNum``, ``Side`` columns.
- Parse ``PassengerId`` ("gggg_pp") into travel ``Group`` and member number, and derive group size.
- Handle the spending columns (``RoomService``, ``FoodCourt``, ``ShoppingMall``,
  ``Spa``, ``VRDeck``) — e.g. total spend and a "spent nothing" flag.
- Impute missing values and encode categoricals (``HomePlanet``, ``Destination``,
  ``CryoSleep``, ``VIP``).
"""
from __future__ import annotations


def build_preprocessor():
    """Return a preprocessing pipeline / column transformer for the features."""
    # TODO: construct imputation + encoding + scaling steps.
    raise NotImplementedError


def engineer_features(df):
    """Add engineered features to a raw passenger DataFrame and return it."""
    # TODO: implement Cabin/PassengerId parsing, spend aggregation, etc.
    raise NotImplementedError
