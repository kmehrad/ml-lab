"""Schema-driven features; fitted preprocessing belongs inside each CV fold."""
import pandas as pd

from .data import ID, TARGET


def feature_columns(train: pd.DataFrame) -> list[str]:
    return [c for c in train.columns if c not in (ID, TARGET)]


def numeric_and_categorical(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame if c not in numeric]
    return numeric, categorical


def tree_features(train: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Match categorical codes without using targets; avoid fitting imputers globally."""
    a, b = train.copy(), other.copy()
    _, cats = numeric_and_categorical(a)
    for col in cats:
        levels = pd.Index(pd.concat([a[col], b[col]], ignore_index=True).dropna().unique())
        dtype = pd.CategoricalDtype(categories=levels)
        a[col] = a[col].astype(dtype)
        b[col] = b[col].astype(dtype)
    return a, b, cats
