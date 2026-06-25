"""Equal-weight rank-average blend over base learners' saved OOF/test predictions.

Rank-averaging is robust to per-model probability-scale differences (an AUC metric only
cares about ordering). Reports the blend OOF AUC and each member's OOF AUC, then saves
blend_oof.npy / blend_test.npy for submission.

Usage:
    uv run python -m src.blend --models lgbm xgb histgb cat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def _rank01(a: np.ndarray) -> np.ndarray:
    return (rankdata(a) - 1) / (len(a) - 1)


def blend(models: list[str]) -> dict:
    y = np.load(ART / "y.npy")
    oof = {m: np.load(ART / f"{m}_oof.npy") for m in models}
    test = {m: np.load(ART / f"{m}_test.npy") for m in models}

    print("Member OOF AUC:")
    for m in models:
        print(f"  {m:8s} {roc_auc_score(y, oof[m]):.5f}")

    blend_oof = np.mean([_rank01(oof[m]) for m in models], axis=0)
    blend_test = np.mean([_rank01(test[m]) for m in models], axis=0)
    auc = roc_auc_score(y, blend_oof)
    print(f"\nBLEND ({'+'.join(models)}) OOF AUC = {auc:.5f}")

    np.save(ART / "blend_oof.npy", blend_oof)
    np.save(ART / "blend_test.npy", blend_test)
    res = {"models": models, "blend_oof_auc": float(auc)}
    (ART / "blend_metrics.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["lgbm", "xgb", "histgb", "cat"])
    a = p.parse_args()
    blend(a.models)
