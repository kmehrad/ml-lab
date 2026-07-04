"""Build a validated MAP@3 submission from saved bagged test probabilities; optional upload.

Turns a model's `(n_test, 7)` test-probability matrix into the top-3 class-name string per row
(`'C1 C2 C3'`, best-first) via `metric.to_submission_strings`, validates the result against
`sample_submission.csv`, and writes `outputs/{model}_submission.csv`.

Usage:
    uv run python -m src.submit --model xgb
    uv run python -m src.submit --model xgb --submit -m "xgb baseline, OOF 0.34199"
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .metric import to_submission_strings

COMPETITION = "playground-series-s5e6"
ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def build_submission(model: str) -> Path:
    proba = np.load(ART / f"{model}_test.npy")
    test_id = np.load(ART / "test_id.npy")
    classes = np.load(ART / "classes.npy", allow_pickle=True)
    sub = D.load_sample_submission()

    strings = to_submission_strings(proba, classes)
    out = pd.DataFrame({D.ID: test_id, D.TARGET: strings})
    # order rows to match sample_submission exactly
    out = out.set_index(D.ID).reindex(sub[D.ID]).reset_index()

    # --- validate against sample_submission ---
    valid = set(classes)
    assert list(out.columns) == [D.ID, D.TARGET], out.columns
    assert len(out) == len(sub), f"row count {len(out)} != {len(sub)}"
    assert out[D.ID].tolist() == sub[D.ID].tolist(), "id mismatch vs sample_submission"
    assert out[D.TARGET].notna().all(), "missing predictions for some id"
    parts = out[D.TARGET].str.split(" ")
    assert (parts.str.len() == 3).all(), "every row must have exactly 3 class names"
    assert parts.map(lambda p: len(set(p)) == 3).all(), "the 3 names must be distinct"
    assert parts.map(lambda p: all(x in valid for x in p)).all(), "unknown class name present"

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model}_submission.csv"
    out.to_csv(path, index=False)
    top1 = out[D.TARGET].str.split(" ").str[0].value_counts()
    print(f"wrote {path}  rows={len(out)}")
    print("top-1 class distribution:\n" + top1.to_string())
    return path


def submit_to_kaggle(path: Path, message: str) -> None:
    env = dict(os.environ)
    token = Path.home() / ".kaggle" / "access_token"
    if token.exists() and "KAGGLE_API_TOKEN" not in env:
        env["KAGGLE_API_TOKEN"] = token.read_text().strip()
    cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(path), "-m", message]
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="xgb")
    p.add_argument("--submit", action="store_true", help="upload via Kaggle CLI")
    p.add_argument("-m", "--message", default="")
    a = p.parse_args()
    path = build_submission(a.model)
    if a.submit:
        submit_to_kaggle(path, a.message or f"{a.model} submission")
