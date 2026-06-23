"""Fit the chosen model on all training data and build a Kaggle submission.

The default model is ``histgb`` — the best balanced-accuracy model in the 5-fold
comparison (see the README). The preprocessing pipeline is fit on the full
training set and applied to test, then the model is trained with balanced class
weights and used to predict test labels. The output is validated against
``sample_submission.csv`` before (optionally) uploading via the Kaggle CLI.

Usage
-----
    python -m src.submit                       # write outputs/histgb_submission.csv
    python -m src.submit --model xgb
    python -m src.submit --submit -m "HistGB, engineered features, 5-fold CV 0.9695"
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from .data import (
    CLASS_ORDER,
    ID_COLUMN,
    TARGET_COLUMN,
    decode_target,
    encode_target,
    load_raw,
    split_features_target,
)
from .preprocessing import build_preprocessor
from .train import MODEL_FAMILY, build_estimator, fit_predict

COMPETITION = "playground-series-s6e4"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def make_submission(model: str = "histgb", seed: int = 42) -> Path:
    """Train ``model`` on all train data, predict test, and write a CSV."""
    if model not in MODEL_FAMILY:
        raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODEL_FAMILY)}")

    train = load_raw("train")
    test = load_raw("test")
    X, y_labels = split_features_target(train)
    y = encode_target(y_labels).to_numpy()
    X_test = test[list(X.columns)]

    family = MODEL_FAMILY[model]
    pre = build_preprocessor(family).fit(X, y)
    X_model = pre.transform(X)
    X_test_model = pre.transform(X_test)

    estimator = build_estimator(model, seed)
    codes = fit_predict(model, estimator, X_model, y, X_test_model)
    labels = decode_target(codes)

    submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].to_numpy(), TARGET_COLUMN: labels})
    _validate(submission)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{model}_submission.csv"
    submission.to_csv(path, index=False)

    counts = submission[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER)
    print(f"Wrote {path} ({len(submission):,} rows)")
    print("Predicted class distribution:")
    for cls in CLASS_ORDER:
        print(f"  {cls:>6}: {int(counts[cls]):>7,} ({counts[cls] / len(submission) * 100:.2f}%)")
    return path


def _validate(submission: pd.DataFrame) -> None:
    """Check the submission matches the sample submission's shape and schema."""
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    if list(submission.columns) != list(sample.columns):
        raise ValueError(f"columns {list(submission.columns)} != {list(sample.columns)}")
    if len(submission) != len(sample):
        raise ValueError(f"row count {len(submission)} != {len(sample)}")
    if set(submission[ID_COLUMN]) != set(sample[ID_COLUMN]):
        raise ValueError("submission ids do not match the sample submission ids")
    unknown = set(submission[TARGET_COLUMN]) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"unexpected labels in submission: {unknown}")


def submit_to_kaggle(path: Path, message: str) -> None:
    """Upload a submission file with the Kaggle CLI."""
    cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(path), "-m", message]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="histgb", help="model name (default: histgb)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submit", action="store_true", help="upload via the Kaggle CLI")
    parser.add_argument("-m", "--message", default=None, help="submission message")
    args = parser.parse_args()

    path = make_submission(model=args.model, seed=args.seed)
    if args.submit:
        message = args.message or f"{args.model} on engineered features (5-fold CV documented in README)"
        submit_to_kaggle(path, message)


if __name__ == "__main__":
    main()
