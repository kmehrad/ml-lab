"""Build, validate, and (optionally) upload a NIPS4B submission.

Reads a fold-averaged test-probability artifact ``{name}_test.npy`` (1000×87),
expands it to the 87,000-row ``ID,Probability`` format
(``nips4b_birds_testfile{NNNN}.wav_classnumber_{C}``), validates it against the
organizer example submission, writes ``outputs/{name}_submission.csv``, and with
``--submit`` uploads it via the Kaggle CLI.

Usage:
    uv run python -m src.submit --name rf
    uv run python -m src.submit --name rf --submit -m "RF baseline agg-MFCC(86) OOF 0.930"
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src import data

SLUG = "multilabel-bird-species-classification-nips2013"
ART = data.ROOT / "experiments" / "artifacts"
OUT = data.ROOT / "outputs"


def build(name: str) -> Path:
    test = np.load(ART / f"{name}_test.npy")
    if test.shape != (data.N_TEST, data.N_CLASSES):
        raise ValueError(f"{name}_test.npy shape {test.shape} != {(data.N_TEST, data.N_CLASSES)}")

    # id -> prob, then align to the example submission's exact ID order
    prob = {
        data.submission_id(f, c): float(test[f - 1, c - 1])
        for f in range(1, data.N_TEST + 1)
        for c in range(1, data.N_CLASSES + 1)
    }
    sample = data.load_sample_submission()
    df = pd.DataFrame({"ID": sample["ID"], "Probability": sample["ID"].map(prob)})

    _validate(df, sample)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_submission.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({len(df)} rows, prob range [{df.Probability.min():.3f}, {df.Probability.max():.3f}])")
    return path


def _validate(df: pd.DataFrame, sample: pd.DataFrame) -> None:
    if len(df) != len(sample) != data.N_TEST * data.N_CLASSES:
        raise ValueError(f"row count {len(df)} != {data.N_TEST * data.N_CLASSES}")
    if set(df["ID"]) != set(sample["ID"]):
        raise ValueError("ID set does not match the example submission")
    p = df["Probability"].to_numpy()
    if np.isnan(p).any():
        raise ValueError("NaN probabilities (some test file × class pair unmapped)")
    if p.min() < 0 or p.max() > 1:
        raise ValueError(f"probabilities out of [0,1]: [{p.min()}, {p.max()}]")


def upload(path: Path, message: str) -> None:
    env = dict(os.environ)
    token = Path.home() / ".kaggle" / "access_token"
    if token.exists():
        env.setdefault("KAGGLE_API_TOKEN", token.read_text().strip())
    cmd = ["uv", "run", "kaggle", "competitions", "submit", "-c", SLUG, "-f", str(path), "-m", message]
    print("+", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True, cwd=data.ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="artifact prefix, e.g. 'rf'")
    ap.add_argument("--submit", action="store_true", help="upload to Kaggle after building")
    ap.add_argument("-m", "--message", default="", help="submission description/tag")
    args = ap.parse_args()

    path = build(args.name)
    if args.submit:
        msg = args.message or f"{args.name} submission"
        upload(path, msg)


if __name__ == "__main__":
    main()
