"""Build a validated AUC submission from saved bagged test predictions; optional upload.

Turns a model's ``(n_test,)`` test-probability vector into the ``id,Heart Disease``
submission (probability of `Presence`, since ROC AUC needs continuous scores, not the
string label), validates the result against ``sample_submission.csv``, and writes
``outputs/{model}_submission.csv``.

Usage:
    uv run python -m src.submit --model xgb
    uv run python -m src.submit --model blend --submit -m "blend, OOF AUC 0.95xx"
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

COMPETITION = "playground-series-s6e2"
ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def build_submission(model: str) -> Path:
    pred = np.load(ART / f"{model}_test.npy").astype(float)
    test_id = np.load(ART / "test_id.npy")
    sub = D.load_sample_submission()

    pred = np.clip(pred, 0.0, 1.0)
    out = pd.DataFrame({D.ID: test_id, D.TARGET: pred})
    # order rows to match sample_submission exactly
    out = out.set_index(D.ID).reindex(sub[D.ID]).reset_index()

    # --- validate against sample_submission ---
    assert list(out.columns) == [D.ID, D.TARGET], out.columns
    assert len(out) == len(sub), f"row count {len(out)} != {len(sub)}"
    assert out[D.ID].tolist() == sub[D.ID].tolist(), "id mismatch vs sample_submission"
    assert out[D.TARGET].notna().all(), "missing predictions for some id"
    assert np.isfinite(out[D.TARGET]).all(), "non-finite predictions"
    assert out[D.TARGET].between(0.0, 1.0).all(), "predictions out of [0,1] range"

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model}_submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}")
    print(f"P(Presence)  min={out[D.TARGET].min():.4f}  mean={out[D.TARGET].mean():.4f}  "
          f"max={out[D.TARGET].max():.4f}")
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
