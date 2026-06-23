"""Build a Kaggle submission from saved bagged test predictions.

``src.train`` saves bagged test probabilities per model (``{key}_test.npy``) and
``src.blend`` saves the blended test probabilities (``blend_test.npy``). This
module turns either into a validated submission CSV and optionally uploads it —
no model refitting needed.

Usage
-----
    python -m src.blend                                   # build the blend first
    python -m src.submit --model blend                    # outputs/blend_submission.csv
    python -m src.submit --model lgbm_aug
    python -m src.submit --model blend --submit -m "diversity blend"
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ID_COLUMN, load_raw
from .blend import BLEND_TEST

COMPETITION = "playground-series-s5e12"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _load_test_probs(model: str) -> np.ndarray:
    """Load bagged test probabilities for a model key or the blend."""
    path = BLEND_TEST if model == "blend" else ARTIFACTS_DIR / f"{model}_test.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run `python -m src.train` (and `src.blend` "
            f"for the blend) before submitting."
        )
    return np.load(path)


def make_submission(model: str = "blend") -> Path:
    """Write a validated probability submission for a model key or the blend."""
    proba = _load_test_probs(model)
    test_ids = load_raw("test")[ID_COLUMN].to_numpy()
    if len(proba) != len(test_ids):
        raise ValueError(f"prediction length {len(proba)} != test rows {len(test_ids)}")

    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    prob_col = [c for c in sample.columns if c != ID_COLUMN][0]
    submission = pd.DataFrame({ID_COLUMN: test_ids, prob_col: proba})
    _validate(submission, sample, prob_col)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{model}_submission.csv"
    submission.to_csv(path, index=False)
    print(f"Wrote {path} ({len(submission):,} rows)")
    print(f"Probability summary: min={proba.min():.4f} mean={proba.mean():.4f} max={proba.max():.4f}")
    return path


def _validate(submission: pd.DataFrame, sample: pd.DataFrame, prob_col: str) -> None:
    """Check the submission matches the sample submission's shape and schema."""
    if list(submission.columns) != list(sample.columns):
        raise ValueError(f"columns {list(submission.columns)} != {list(sample.columns)}")
    if len(submission) != len(sample):
        raise ValueError(f"row count {len(submission)} != {len(sample)}")
    if set(submission[ID_COLUMN]) != set(sample[ID_COLUMN]):
        raise ValueError("submission ids do not match the sample submission ids")
    values = submission[prob_col].to_numpy()
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("probabilities must be finite and within [0, 1]")


def submit_to_kaggle(path: Path, message: str) -> None:
    """Upload a submission file with the Kaggle CLI."""
    cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(path), "-m", message]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="blend", help="model key or 'blend'")
    parser.add_argument("--submit", action="store_true", help="upload via the Kaggle CLI")
    parser.add_argument("-m", "--message", default=None, help="submission message")
    args = parser.parse_args()

    path = make_submission(model=args.model)
    if args.submit:
        message = args.message or f"{args.model} (diversity blend; see experiments/README.md)"
        submit_to_kaggle(path, message)


if __name__ == "__main__":
    main()
