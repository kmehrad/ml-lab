"""Leak-free OOF stacking meta-model over base learners' probability matrices.

Trains a multinomial `LogisticRegression` meta-model on the base learners' OOF probabilities using an
inner StratifiedKFold, so each row's meta-prediction comes from a meta-model that never saw that row —
the stacked OOF is honest. The same meta-model (refit on all OOF) is applied to the base learners'
fold-bagged test probabilities. Scored on tuned balanced accuracy (`metric.tune_weights`), and compared
against the equal-weight blend so we only adopt stacking if it beats averaging above the noise floor.

    uv run python -m src.stack --models xgb lgbm cat realmlp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .cv import folds
from .metric import score_proba, tune_weights
from .blend import combine

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def _design(mats: list[np.ndarray]) -> np.ndarray:
    """Stack base OOF/test probability matrices side by side as meta-features."""
    return np.hstack(mats)


def run_stack(models: list[str], tag: str = "stack", C: float = 1.0, n_splits: int = 5) -> dict:
    y = np.load(ART / "y.npy")
    oof_mats = [np.load(ART / f"{m}_oof.npy") for m in models]
    test_mats = [np.load(ART / f"{m}_test.npy") for m in models]
    Xoof, Xtest = _design(oof_mats), _design(test_mats)

    per_model = {m: score_proba(y, o, tune_weights(y, o)) for m, o in zip(models, oof_mats)}
    print("base OOF balanced accuracy (tuned):")
    for m, s in per_model.items():
        print(f"  {m:14s} {s:.5f}")

    # leak-free stacked OOF: meta-model trained out-of-fold
    stack_oof = np.zeros((len(y), oof_mats[0].shape[1]))
    for tr, va in folds(y, n_splits, seed=2024):
        meta = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
        meta.fit(Xoof[tr], y[tr])
        stack_oof[va] = meta.predict_proba(Xoof[va])
    meta_full = LogisticRegression(max_iter=2000, C=C, class_weight="balanced").fit(Xoof, y)
    stack_test = meta_full.predict_proba(Xtest)

    w = tune_weights(y, stack_oof)
    stack_bacc = score_proba(y, stack_oof, w)
    blend_bacc = score_proba(y, combine(oof_mats), tune_weights(y, combine(oof_mats)))
    best_base = max(per_model.values())
    print(f"\nstack OOF {stack_bacc:.5f}   blend OOF {blend_bacc:.5f}   best base {best_base:.5f}"
          f"   (stack Δ vs best base {stack_bacc - best_base:+.5f})")

    res = {"models": models, "tag": tag, "oof_bacc_tuned": float(stack_bacc),
           "blend_oof_bacc": float(blend_bacc), "best_base_oof_bacc": float(best_base),
           "delta_vs_best_base": float(stack_bacc - best_base),
           "decision_weights": [float(x) for x in w], "C": C}
    np.save(ART / f"{tag}_oof.npy", stack_oof)
    np.save(ART / f"{tag}_test.npy", stack_test)
    (ART / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {tag}_oof.npy / _test.npy / _metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="base artifact prefixes to stack")
    p.add_argument("--tag", default="stack")
    p.add_argument("--C", type=float, default=1.0, help="LogReg inverse-regularization")
    a = p.parse_args()
    run_stack(a.models, a.tag, a.C)
