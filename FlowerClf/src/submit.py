"""Build (and optionally upload) the Kaggle submission from saved test probs.

Reads ``{run}_test_probs.npy`` + ``{run}_test_ids.npy``, argmax → ``label``, validates
against ``data/raw/sample_submission.csv``, writes ``outputs/{run}_submission.csv``.

Uploading is gated behind ``--submit`` (only after the val macro-F1 has been approved).

Usage:
    uv run python -m src.submit --run tf_efficientnet_b3_ns_r224
    uv run python -m src.submit --run tf_efficientnet_b3_ns_r224 --submit -m "b3 @224 TTA"
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np
import pandas as pd

from src import data
from src.train import ARTIFACTS

COMPETITION = "tpu-getting-started"
OUTPUTS = data.ROOT / "outputs"


def build_submission(run: str) -> pd.DataFrame:
    probs = np.load(ARTIFACTS / f"{run}_test_probs.npy")
    ids = np.load(ARTIFACTS / f"{run}_test_ids.npy", allow_pickle=True).astype(str)
    if len(ids) != len(probs):
        raise ValueError(f"ids ({len(ids)}) and probs ({len(probs)}) length mismatch")
    sub = pd.DataFrame({"id": ids, "label": probs.argmax(1).astype(int)})

    sample_path = data.RAW_DIR / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if list(sample.columns) != ["id", "label"]:
            raise ValueError(f"sample_submission columns are {list(sample.columns)}")
        sample_ids = set(sample["id"].astype(str))
        if set(sub["id"]) != sample_ids:
            raise ValueError("submission ids do not match sample_submission ids")
        sub = sub.set_index("id").loc[sample["id"].astype(str)].reset_index()  # match order
    if not sub["label"].between(0, data.N_CLASSES - 1).all():
        raise ValueError("labels out of range [0, 103]")
    return sub


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/validate (and optionally upload) submission")
    ap.add_argument("--run", required=True)
    ap.add_argument("--submit", action="store_true", help="upload to Kaggle (after approval)")
    ap.add_argument("-m", "--message", default=None, help="submission message")
    args = ap.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    sub = build_submission(args.run)
    out = OUTPUTS / f"{args.run}_submission.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out}  ({len(sub)} rows)  label range "
          f"[{sub['label'].min()}, {sub['label'].max()}]  {sub['label'].nunique()} classes used")

    if args.submit:
        msg = args.message or f"{args.run}"
        cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(out), "-m", msg]
        print("uploading:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("(not uploaded — re-run with --submit after the val macro-F1 is approved)")


if __name__ == "__main__":
    main()
