"""Schema / class-list sanity. Data-dependent checks skip cleanly when the
converted metadata isn't present (e.g. running on the dev Mac before download)."""
from __future__ import annotations

import pytest

from src import data


def test_class_list_is_104_unique():
    assert data.N_CLASSES == 104
    assert len(data.CLASSES) == 104
    assert len(set(data.CLASSES)) == 104, "duplicate class names"
    assert all(isinstance(c, str) and c.strip() == c and c for c in data.CLASSES)


def test_expected_counts_positive():
    assert data.N_TRAIN > 0 and data.N_VAL > 0 and data.N_TEST > 0
    assert data.RESOLUTIONS == (192, 224, 331, 512)


@pytest.mark.parametrize("res", data.RESOLUTIONS)
def test_metadata_if_present(res):
    """If a metadata parquet exists, validate its schema, labels and counts."""
    if not data.metadata_path(res).exists():
        pytest.skip(f"metadata_{res}.parquet not present")
    df = data.load_metadata(res)
    assert list(df.columns)[: len(data.METADATA_COLS)] == data.METADATA_COLS or all(
        c in df.columns for c in data.METADATA_COLS
    )
    assert set(df["split"].unique()) <= {"train", "val", "test"}

    labeled = df[df["split"].isin(["train", "val"])]
    assert labeled["label"].between(0, data.N_CLASSES - 1).all()

    test = df[df["split"] == "test"]
    assert test["label"].isna().all(), "test split must be unlabeled"

    counts = df["split"].value_counts().to_dict()
    # Allow small drift but catch gross mismatches.
    assert abs(counts.get("train", 0) - data.N_TRAIN) <= 50
    assert abs(counts.get("val", 0) - data.N_VAL) <= 50
    assert abs(counts.get("test", 0) - data.N_TEST) <= 50
