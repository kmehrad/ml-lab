"""Build a validated submission from saved bagged test predictions; optional Kaggle upload.

Usage:
    uv run python -m src.submit --model lgbm
    uv run python -m src.submit --model lgbm --submit -m "lgbm baseline, OOF 0.96091"
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

COMPETITION = "energy-anomaly-detection"
ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def build_submission(model: str) -> Path:
    pred = np.load(ART / f"{model}_test.npy")
    row_id = np.load(ART / "test_row_id.npy")
    sub = D.load_sample_submission()

    df = pd.DataFrame({"row_id": row_id, "anomaly": pred}).set_index("row_id")
    out = sub[["row_id"]].copy()
    out["anomaly"] = out["row_id"].map(df["anomaly"])

    # validate against sample_submission
    assert len(out) == len(sub), f"row count {len(out)} != {len(sub)}"
    assert out["anomaly"].notna().all(), "missing predictions for some row_id"
    assert out["anomaly"].between(0, 1).all(), "predictions out of [0,1]"
    assert list(out.columns) == ["row_id", "anomaly"], out.columns

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model}_submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}  pred[min/mean/max]="
          f"{out.anomaly.min():.4f}/{out.anomaly.mean():.4f}/{out.anomaly.max():.4f}")
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
    p.add_argument("--model", default="lgbm")
    p.add_argument("--submit", action="store_true", help="upload via Kaggle CLI")
    p.add_argument("-m", "--message", default="")
    a = p.parse_args()
    path = build_submission(a.model)
    if a.submit:
        submit_to_kaggle(path, a.message or f"{a.model} submission")
