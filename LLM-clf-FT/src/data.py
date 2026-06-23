"""Data loading and shared paths/constants for the Chatbot Arena task.

The text columns (`prompt`, `response_a`, `response_b`) are JSON-encoded lists of
turns; `parse_turns` decodes them and `join_turns` flattens to a single string.
`model_a`/`model_b` exist only in train and must never become model features
(they are absent from the hidden test set).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SUB = ROOT / "data" / "submissions"

TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]
TEXT_COLS = ["prompt", "response_a", "response_b"]
SEED = 42

# turn separator used when flattening multi-turn conversations
TURN_SEP = "\n\n"


def parse_turns(x) -> list[str]:
    """Decode a JSON-list text cell into a list of strings; robust to bad rows."""
    if isinstance(x, list):
        return [s for s in x if isinstance(s, str)]
    try:
        v = json.loads(x)
    except Exception:
        return [str(x)]
    if isinstance(v, list):
        return [s if isinstance(s, str) else "" for s in v]
    return [str(v)]


def join_turns(x) -> str:
    return TURN_SEP.join(parse_turns(x))


def load_train(with_folds: bool = True) -> pd.DataFrame:
    df = pd.read_csv(RAW / "train.csv")
    if with_folds:
        folds = pd.read_parquet(PROC / "folds.parquet")  # id, label, fold
        df = df.merge(folds, on="id", how="left")
        if df["fold"].isna().any():
            raise RuntimeError(
                "Missing folds for some ids — regenerate data/processed/folds.parquet "
                "via notebooks/01_eda.ipynb."
            )
    return df


def load_test() -> pd.DataFrame:
    # test has only: id, prompt, response_a, response_b (no model_*, no targets)
    return pd.read_csv(RAW / "test.csv")
