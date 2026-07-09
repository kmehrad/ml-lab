"""Build a validated balanced-accuracy submission from saved test probabilities; optional upload.

Tunes the per-class decision weights on the model's **OOF** matrix (never the test set), applies them
to the ``(n_test, 3)`` test-probability matrix to pick one class label per row, validates the result
against ``sample_submission.csv``, and writes ``outputs/{model}_submission.csv``
(columns ``id,health_condition``).

Usage:
    uv run python -m src.submit --model blend
    uv run python -m src.submit --model blend --submit -m "gbdt+nn blend, OOF bal_acc 0.xxxx"
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .metric import to_submission_labels, tune_weights, score_proba

COMPETITION = "playground-series-s6e7"
ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def build_submission(model: str) -> Path:
    test_proba = np.load(ART / f"{model}_test.npy")
    oof_proba = np.load(ART / f"{model}_oof.npy")
    y = np.load(ART / "y.npy")
    test_id = np.load(ART / "test_id.npy")
    classes = np.load(ART / "classes.npy", allow_pickle=True)
    sub = D.load_sample_submission()

    # Decision weights are tuned on OOF (leak-free) and applied to the test proba.
    weights = tune_weights(y, oof_proba)
    print(f"OOF balanced accuracy tuned={score_proba(y, oof_proba, weights):.5f}"
          f"   weights={np.round(weights, 3).tolist()}")

    labels = to_submission_labels(test_proba, classes, weights)
    out = pd.DataFrame({D.ID: test_id, D.TARGET: labels})
    out = out.set_index(D.ID).reindex(sub[D.ID]).reset_index()  # order rows to match sample exactly

    # --- validate against sample_submission ---
    valid = set(classes)
    assert list(out.columns) == [D.ID, D.TARGET], out.columns
    assert len(out) == len(sub), f"row count {len(out)} != {len(sub)}"
    assert out[D.ID].tolist() == sub[D.ID].tolist(), "id mismatch vs sample_submission"
    assert out[D.TARGET].notna().all(), "missing predictions for some id"
    assert out[D.TARGET].isin(valid).all(), "unknown class label present"

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model}_submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}")
    print("predicted class distribution:\n" + out[D.TARGET].value_counts().to_string())
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
    p.add_argument("--model", default="blend")
    p.add_argument("--submit", action="store_true", help="upload via Kaggle CLI")
    p.add_argument("-m", "--message", default="")
    a = p.parse_args()
    path = build_submission(a.model)
    if a.submit:
        submit_to_kaggle(path, a.message or f"{a.model} submission")
