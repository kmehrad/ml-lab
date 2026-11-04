"""Load Kaggle CSVs and enforce the competition's minimal schema."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TARGET = "Will_Buy_EV"
ID = "id"


def load_csv(name: str) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; join/accept Kaggle rules, then download S6E9 data")
    return pd.read_csv(path)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, sample = (load_csv(n) for n in ("train.csv", "test.csv", "sample_submission.csv"))
    if ID not in train or ID not in test or TARGET not in train or TARGET in test:
        raise ValueError("unexpected train/test schema")
    if sample.columns.tolist() != [ID, TARGET]:
        raise ValueError("unexpected sample submission columns")
    if len(sample) != len(test) or sample[ID].duplicated().any() or test[ID].duplicated().any():
        raise ValueError("test/sample IDs are not unique or row counts differ")
    if set(sample[ID]) != set(test[ID]):
        raise ValueError("sample/test IDs differ")
    features = [c for c in train if c not in (ID, TARGET)]
    if set(features) != set(test.columns) - {ID}:
        raise ValueError("train/test feature columns differ")
    return train, test, sample


def encode_target(series: pd.Series) -> pd.Series:
    """Map Yes/No or 1/0 to a binary positive-class target."""
    values = set(series.dropna().unique())
    if values == {"Yes", "No"}:
        return series.map({"No": 0, "Yes": 1}).astype(int)
    if values == {0, 1}:
        return series.astype(int)
    raise ValueError(f"unexpected {TARGET} labels: {values}")
