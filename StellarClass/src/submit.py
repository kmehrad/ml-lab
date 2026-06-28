"""Build a validated submission from saved test probabilities; optional Kaggle upload.

Usage:
    uv run python -m src.submit --model blend
    uv run python -m src.submit --model blend --submit -m "blend, OOF bacc 0.97xxx"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M

COMPETITION = "playground-series-s6e6"
ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def build_submission(model: str) -> Path:
    test_proba = np.load(ART / f"{model}_test.npy")
    test_id = np.load(ART / "test_id.npy")

    # apply the same balanced-accuracy multipliers chosen on OOF, if present
    metrics_path = ART / f"{model}_metrics.json"
    mult = None
    if metrics_path.exists():
        mult = np.array(json.load(open(metrics_path)).get("mult", np.ones(len(D.CLASSES))))
    labels = D.decode_target(M.predict_labels(test_proba, mult))

    sub = D.load_sample_submission()
    out = sub[[D.ID_COL]].copy()
    pred_map = dict(zip(test_id, labels))
    out[D.TARGET] = out[D.ID_COL].map(pred_map)

    assert len(out) == len(sub), f"row count {len(out)} != {len(sub)}"
    assert out[D.TARGET].notna().all(), "missing predictions for some id"
    assert set(out[D.TARGET].unique()) <= set(D.CLASSES), "unexpected labels"
    assert list(out.columns) == [D.ID_COL, D.TARGET], out.columns

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model}_submission.csv"
    out.to_csv(path, index=False)
    dist = out[D.TARGET].value_counts(normalize=True).round(4).to_dict()
    print(f"wrote {path}  rows={len(out)}  pred dist={dist}")
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
