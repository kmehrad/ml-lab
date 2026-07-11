"""pytabkit TabM/FTT tabular learner -- the cross-family diversity base learner for the blend.

GBDTs already sit near the OOF ceiling for this dataset (see experiments/README.md); TabM/FTT
are a genuinely different model family (attention/ensemble-of-MLPs, not tree-based), so even a
solo-weaker model can still help an equal-weight blend by adding decorrelated errors -- unlike
the plain MLP tried in the sibling StudentHealthRisk project, which was too weak to help.
Intended to run on the remote GPU box (torch/pytabkit are the ``gpu`` extra). Saves OOF/test/
metrics in ``train.py``'s exact binary-classification contract (1-D proba-of-Presence vectors)
so ``blend.py --models lgbm xgb cat tabm`` works unchanged.

Usage:
    uv run python -m src.train_tabkit --variant tabm --sample 50000   # smoke test
    uv run python -m src.train_tabkit --variant tabm --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .cv import folds
from .features import add_features, categorical_columns, feature_columns
from .metric import roc_auc

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"


def build_estimator(variant: str, device: str, seed: int):
    if variant == "tabm":
        from pytabkit import TabM_D_Classifier
        return TabM_D_Classifier(device=device, random_state=seed, val_metric_name="1-auc_ovr")
    if variant == "ftt":
        from pytabkit import FTT_D_Classifier
        # FTT's skorch backend only implements val_metric_name in {"class_error", "cross_entropy"}
        # (unlike TabM's backend, which also supports "1-auc_ovr") -- confirmed by inspecting
        # pytabkit.models.nn_models.rtdl_resnet.create_classifier_skorch.
        return FTT_D_Classifier(device=device, random_state=seed, val_metric_name="cross_entropy")
    raise ValueError(f"unknown variant {variant}")


def run_cv(variant: str, sample: int | None = None, n_splits: int = 5,
           groups=("base",), tag: str = "", device: str = "cpu", seed: int = 42) -> dict:
    groups = tuple(groups)
    df = add_features(D.load_train(), groups)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    print(f"train rows={len(df):,}  variant={variant}  features={list(groups)}"
          + (f"  tag={tag}" if tag else ""))

    feats = feature_columns(groups)
    cats = categorical_columns(groups)

    test = Xte = test_id = None
    if not sample:
        test = add_features(D.load_test(), groups)
        # Align categorical levels across train/test as in train.py (not strictly required by
        # pytabkit, which reads categoricals by name, but keeps behavior consistent).
        for c in cats:
            levels = pd.unique(pd.concat([df[c].astype("object"), test[c].astype("object")],
                                         ignore_index=True))
            dtype = pd.CategoricalDtype(categories=levels)
            df[c] = df[c].astype("object").astype(dtype)
            test[c] = test[c].astype("object").astype(dtype)
        test_id = test[D.ID].to_numpy()
        Xte = test[feats]
        test_pred = np.zeros(len(test))

    X = df[feats]
    y = D.encode_target(df).to_numpy()

    oof = np.zeros(len(df))
    fold_scores = []
    t0 = time.time()
    for k, (tr, va) in enumerate(folds(y, n_splits, seed=seed)):
        est = build_estimator(variant, device=device, seed=seed)
        est.fit(X.iloc[tr], y[tr], X.iloc[va], y[va], cat_col_names=cats)
        va_pred = np.asarray(est.predict_proba(X.iloc[va])[:, 1], dtype=float)
        oof[va] = va_pred
        if Xte is not None:
            test_pred += np.asarray(est.predict_proba(Xte)[:, 1], dtype=float) / n_splits
        score = roc_auc(y[va], va_pred)
        fold_scores.append(score)
        print(f"  fold {k}: AUC={score:.5f}")

    oof_auc = roc_auc(y, oof)
    res = {
        "model": variant, "features": list(groups), "tag": tag, "oof_auc": float(oof_auc),
        "fold_mean": float(np.mean(fold_scores)), "fold_std": float(np.std(fold_scores)),
        "fold_scores": [float(s) for s in fold_scores], "n_rows": int(len(df)),
        "n_features": int(len(feats)), "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"\nOOF AUC = {oof_auc:.5f}   folds {res['fold_mean']:.5f} +/- {res['fold_std']:.5f}"
          f"   ({res['elapsed_s']}s)")

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        np.save(ART / f"{variant}{suffix}_oof.npy", oof)
        np.save(ART / f"{variant}{suffix}_test.npy", test_pred)
        np.save(ART / "y.npy", y)                       # shared, feature-independent
        np.save(ART / "test_id.npy", test_id)
        (ART / f"{variant}{suffix}_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> {ART}/{variant}{suffix}_oof.npy, {variant}{suffix}_test.npy, "
              f"{variant}{suffix}_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="tabm", choices=["tabm", "ftt"])
    p.add_argument("--sample", type=int, default=None, help="limit to N rows (smoke test; no test preds)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--features", nargs="+", default=["base"], help="feature groups: base interactions")
    p.add_argument("--tag", default="", help="artifact filename suffix (keeps experiments separate)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run_cv(a.variant, a.sample, a.folds, groups=a.features, tag=a.tag,
           device=a.device, seed=a.seed)
