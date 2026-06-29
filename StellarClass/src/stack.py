"""Stacked generalization over base-model OOF probabilities.

A meta-learner is trained on the base models' out-of-fold class probabilities using the **same CV
folds** (so meta-OOF rows are honest), then applied to the base models' fold-averaged test
probabilities. The balanced-accuracy decision correction (`src/metrics.py`) is applied to the meta
output. Compare against the simple mean-blend (`src/blend.py`) and keep whichever wins on OOF.

Usage:
    uv run python -m src.stack --models lgbm xgb cat nn
    uv run python -m src.stack --models lgbm_eo xgb_eo nn --meta logreg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import cv
from . import data as D
from . import metrics as M

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def _stack_features(models: list[str], which: str) -> np.ndarray:
    """Horizontally stack each model's (n, 3) probability arrays -> (n, 3*len(models))."""
    arrs = [np.load(ART / f"{m}_{which}.npy") for m in models]
    return np.hstack(arrs)


def _meta_model(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", multi_class="multinomial")
    raise ValueError(f"unknown meta model: {kind}")


def stack(models: list[str], meta: str = "logreg") -> dict:
    y = np.load(ART / "y_true.npy")
    Xoof = _stack_features(models, "oof")
    Xtest = _stack_features(models, "test")

    folds = cv.get_folds(y)
    meta_oof = np.zeros((len(y), len(D.CLASSES)))
    meta_test = np.zeros((len(Xtest), len(D.CLASSES)))

    for tr_idx, va_idx in folds:
        clf = _meta_model(meta)
        clf.fit(Xoof[tr_idx], y[tr_idx])
        meta_oof[va_idx] = clf.predict_proba(Xoof[va_idx])
        meta_test += clf.predict_proba(Xtest) / len(folds)

    raw = M.balanced_accuracy(y, meta_oof)
    mult, corrected = M.tune_multipliers(y, meta_oof)
    print(f"[stack {meta} {models}] OOF bal-acc raw={raw:.5f}  corrected={corrected:.5f}  "
          f"mult={np.round(mult,3).tolist()}")

    np.save(ART / "stack_oof.npy", meta_oof)
    np.save(ART / "stack_test.npy", meta_test)
    result = {
        "models": models, "meta": meta, "oof_bacc_raw": raw,
        "oof_bacc_corrected": corrected, "mult": mult.tolist(),
    }
    json.dump(result, open(ART / "stack_metrics.json", "w"), indent=2)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--meta", default="logreg", choices=["logreg"])
    a = p.parse_args()
    stack(a.models, a.meta)