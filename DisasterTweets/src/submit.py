"""Build a validated submission CSV from saved OOF/test predictions and (optionally) upload.

Usage:
    uv run python -m src.submit --model tfidf_lr               # write outputs/tfidf_lr_submission.csv
    uv run python -m src.submit --model tfidf_lr --submit -m "TF-IDF+LR baseline"

The hard-label threshold is read from ``{model}_metrics.json`` (tuned on OOF F1); pass
``--threshold`` to override. Uploading requires KAGGLE_API_TOKEN and explicit ``--submit``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src import data

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "artifacts"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

COMPETITION = "nlp-getting-started"


def build(model: str, threshold: float | None = None) -> Path:
    test_pred = np.load(ART / f"{model}_test.npy")
    test_ids = np.load(ART / "test_ids.npy")
    with open(ART / f"{model}_metrics.json") as fh:
        meta = json.load(fh)
    thr = threshold if threshold is not None else meta["threshold"]

    labels = (test_pred >= thr).astype(int)
    sub = pd.DataFrame({data.ID_COL: test_ids, data.TARGET: labels})

    # validate against sample_submission (ids + row count + label domain)
    sample = data.load_sample_submission()
    assert len(sub) == len(sample), f"row count {len(sub)} != {len(sample)}"
    assert set(sub[data.ID_COL]) == set(sample[data.ID_COL]), "id set mismatch"
    assert set(sub[data.TARGET].unique()) <= {0, 1}, "labels not in {0,1}"
    sub = sub.set_index(data.ID_COL).loc[sample[data.ID_COL]].reset_index()  # match order

    path = OUT / f"{model}_submission.csv"
    sub.to_csv(path, index=False)
    pos = sub[data.TARGET].mean()
    print(f"wrote {path}  ({len(sub)} rows, threshold {thr:.3f}, "
          f"predicted positive rate {pos:.3f}, OOF F1 {meta['oof_f1']})")
    return path


def upload(path: Path, message: str) -> None:
    cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION,
           "-f", str(path), "-m", message]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tfidf_lr")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--submit", action="store_true", help="upload to Kaggle")
    ap.add_argument("-m", "--message", default=None)
    args = ap.parse_args()

    alias = {"tfidf": "tfidf_lr"}
    model = alias.get(args.model, args.model)

    path = build(model, threshold=args.threshold)
    if args.submit:
        upload(path, args.message or f"{model} submission")


if __name__ == "__main__":
    main()
