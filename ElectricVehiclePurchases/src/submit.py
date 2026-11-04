"""Validate sample-aligned probability CSV; upload only with --submit."""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import data

ART = data.ROOT / "experiments" / "artifacts"
OUT = data.ROOT / "outputs"
COMPETITION = "playground-series-s6e9"


def build_submission(model: str) -> Path:
    _, test, sample = data.load_all()
    predictions = np.load(ART / f"{model}_test.npy")
    ids = np.load(ART / "test_id.npy")
    if predictions.shape != (len(test),) or ids.shape != (len(test),):
        raise ValueError("prediction or ID shape differs from test")
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("predictions must be finite probabilities in [0, 1]")
    if len(set(ids)) != len(ids) or set(ids) != set(sample[data.ID]):
        raise ValueError("saved IDs do not match sample submission")
    frame = pd.DataFrame({data.ID: ids, data.TARGET: predictions})
    frame = frame.set_index(data.ID).loc[sample[data.ID]].reset_index()
    if frame.columns.tolist() != sample.columns.tolist() or not frame[data.ID].equals(sample[data.ID]):
        raise ValueError("submission schema/order mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model}_submission.csv"
    frame.to_csv(path, index=False)
    print(f"validated {path} ({len(frame):,} rows)")
    return path


def upload(path: Path, message: str) -> None:
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN"):
        token = Path.home() / ".kaggle" / "access_token"
        if not token.is_file():
            raise FileNotFoundError("~/.kaggle/access_token missing")
        env["KAGGLE_API_TOKEN"] = token.read_text().strip()
    subprocess.run(["kaggle", "competitions", "submit", "-c", COMPETITION,
                    "-f", str(path), "-m", message], check=True, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="blend")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("-m", "--message", default="")
    args = parser.parse_args()
    result_path = build_submission(args.model)
    if args.submit:
        upload(result_path, args.message or f"S6E9 {args.model}")
