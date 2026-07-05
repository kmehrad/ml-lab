"""Pseudo-labeling experiment (Lever: semi-supervised).

Take the current best model's test predictions, keep the most-confident test rows as extra labelled
training data, and retrain with 5-fold CV to see if OOF balanced accuracy improves. Pseudo-labelled test
rows join **training folds only**; OOF is still scored purely on real synthetic validation rows, so the
comparison to the 0.94998 baseline stays honest.

Confidence = the max class probability after applying the decision weights (normalised). We keep rows
above a percentile so each class is represented (balanced-accuracy cares about minority recall).

    scripts/remote_run.sh run python scripts/pseudo_label.py --model xgb --trees 800 --keep 0.6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data as D
from src.cv import folds
from src.features import add_features, categorical_columns, feature_columns
from src.metric import score_proba, tune_weights, predict_codes
from src.train import build_estimator, _fit_predict

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
N = D.N_CLASSES


def main(model: str, trees: int, keep: float, base_tag: str):
    df = add_features(D.load_train(), ("base",))
    test = add_features(D.load_test(), ("base",))
    feats, cats = feature_columns(("base",)), categorical_columns(("base",))
    for c in cats:  # align categorical levels train/test
        levels = pd.unique(pd.concat([df[c].astype("object"), test[c].astype("object")], ignore_index=True))
        levels = [lv for lv in levels if pd.notna(lv)]
        dt = pd.CategoricalDtype(categories=levels)
        df[c] = df[c].astype("object").astype(dt); test[c] = test[c].astype("object").astype(dt)
    X, Xte = df[feats], test[feats]
    y = D.encode_target(df[D.TARGET])

    # Pseudo-labels from the saved best blend (equal-weight xgb_t800 + lgbm_t800).
    tp = (np.load(ART / "xgb_xgb_t800_test.npy") + np.load(ART / "lgbm_lgbm_t800_test.npy")) / 2
    op = (np.load(ART / "xgb_xgb_t800_oof.npy") + np.load(ART / "lgbm_lgbm_t800_oof.npy")) / 2
    w = tune_weights(y, op)
    pl = predict_codes(tp, w)                                   # pseudo class per test row
    conf = (tp * w).max(1) / (tp * w).sum(1)                    # normalised weighted confidence
    # keep the most-confident `keep` fraction *within each predicted class* (preserve minority coverage)
    mask = np.zeros(len(pl), bool)
    for c in range(N):
        idx = np.where(pl == c)[0]
        if len(idx) == 0:
            continue
        thr = np.quantile(conf[idx], 1 - keep)
        mask[idx[conf[idx] >= thr]] = True
    Xp, yp = Xte.iloc[mask].reset_index(drop=True), pl[mask]
    print(f"pseudo rows kept: {mask.sum():,}/{len(pl):,}  class dist {np.bincount(yp, minlength=N).tolist()}")

    oof = np.zeros((len(df), N))
    for k, (tr, va) in enumerate(folds(y, 5, seed=42)):
        est = build_estimator(model, device="cuda", seed=42, trees=trees)
        Xtr = pd.concat([X.iloc[tr], Xp], ignore_index=True)
        ytr = np.concatenate([y[tr], yp])
        va_p, _, bi = _fit_predict(model, est, Xtr, ytr, X.iloc[va], y[va], None, cats)
        oof[va] = va_p
        print(f"  fold {k}: bal_acc(raw)={score_proba(y[va], va_p):.5f} (best_iter={bi})")
    wf = tune_weights(y, oof)
    print(f"\npseudo-label OOF: raw {score_proba(y, oof):.5f}  tuned {score_proba(y, oof, wf):.5f}"
          f"   (baseline xgb_t800+lgbm_t800 blend 0.94998)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="xgb")
    p.add_argument("--trees", type=int, default=800)
    p.add_argument("--keep", type=float, default=0.6, help="fraction of test rows kept per class")
    p.add_argument("--base-tag", default="blend_t800")
    a = p.parse_args()
    main(a.model, a.trees, a.keep, a.base_tag)
