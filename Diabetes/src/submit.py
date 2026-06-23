"""Fit the chosen model(s) on all training data and build a Kaggle submission.

The submission is the positive-class probability (ROC-AUC metric). Pass a single
model name to submit that model, or ``blend`` to apply the rank-average weights
found by ``src.blend`` across its constituent models. The preprocessing pipeline
is fit on the full training set and applied to test. The output is validated
against ``sample_submission.csv`` before (optionally) uploading via Kaggle CLI.

Usage
-----
    python -m src.submit --model lgbm                 # write outputs/lgbm_submission.csv
    python -m src.submit --model blend
    python -m src.submit --model blend --use-original
    python -m src.submit --model blend --submit -m "Blend LGBM+XGB+Cat, CV 0.83x"
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import ID_COLUMN, RAW_FEATURES, TARGET_COLUMN, load_raw, split_features_target
from .preprocessing import build_preprocessor
from .train import MODEL_FAMILY, build_estimator, fit_predict_proba, _load_training_data
from .blend import WEIGHTS_JSON

COMPETITION = "playground-series-s5e12"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _predict_full(model: str, seed: int, use_original: bool) -> np.ndarray:
    """Train ``model`` on all training data and return test probabilities."""
    X, y = _load_training_data(seed, sample=None, use_original=use_original)
    X_test = load_raw("test")[list(RAW_FEATURES)]

    family = MODEL_FAMILY[model]
    pre = build_preprocessor(family).fit(X, y)
    X_model = pre.transform(X)
    X_test_model = pre.transform(X_test)
    estimator = build_estimator(model, seed)
    return fit_predict_proba(model, estimator, X_model, y, X_test_model)


def make_submission(model: str = "blend", seed: int = 42, use_original: bool = False) -> Path:
    """Build a submission for one model or the saved blend, write and validate it."""
    test_ids = load_raw("test")[ID_COLUMN].to_numpy()

    if model == "blend":
        if not WEIGHTS_JSON.exists():
            raise FileNotFoundError("blend_weights.json not found; run `python -m src.blend` first.")
        spec = json.loads(WEIGHTS_JSON.read_text())
        members, weights = spec["models"], np.asarray(spec["weights"])
        n = len(test_ids)
        proba = np.zeros(n, dtype="float64")
        for name, w in zip(members, weights):
            if w <= 0:  # contributes nothing; skip the (expensive) refit
                continue
            preds = _predict_full(name, seed, use_original)
            proba += w * (rankdata(preds) / n)  # rank-average to match the blender
            print(f"  blended {name!r} (weight {w:.3f})")
    else:
        if model not in MODEL_FAMILY:
            raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODEL_FAMILY)} or 'blend'")
        proba = _predict_full(model, seed, use_original)

    submission = pd.DataFrame({ID_COLUMN: test_ids, TARGET_COLUMN: proba})
    _validate(submission)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{model}_submission.csv"
    submission.to_csv(path, index=False)
    print(f"Wrote {path} ({len(submission):,} rows)")
    print(
        f"Probability summary: min={proba.min():.4f} mean={proba.mean():.4f} "
        f"max={proba.max():.4f}"
    )
    return path


def _validate(submission: pd.DataFrame) -> None:
    """Check the submission matches the sample submission's shape and schema."""
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    # Align the probability column name to whatever the sample uses.
    prob_col = [c for c in sample.columns if c != ID_COLUMN][0]
    if prob_col != TARGET_COLUMN:
        submission.rename(columns={TARGET_COLUMN: prob_col}, inplace=True)
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
    parser.add_argument("--model", default="blend", help="model name or 'blend'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-original", action="store_true", help="augment with CDC original data")
    parser.add_argument("--submit", action="store_true", help="upload via the Kaggle CLI")
    parser.add_argument("-m", "--message", default=None, help="submission message")
    args = parser.parse_args()

    path = make_submission(model=args.model, seed=args.seed, use_original=args.use_original)
    if args.submit:
        message = args.message or f"{args.model} probabilities (5-fold CV documented in README)"
        submit_to_kaggle(path, message)


if __name__ == "__main__":
    main()
