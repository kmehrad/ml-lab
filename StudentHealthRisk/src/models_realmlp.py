"""RealMLP base learner via pytabkit — the strong, GBDT-decorrelated tabular NN.

The 0.951 LB cluster's edge is a modern tabular MLP (RealMLP / TabM) blended with the GBDTs. RealMLP
(`pytabkit.RealMLP_TD_Classifier`, "tuned defaults") does its own preprocessing (robust scaling, one-hot /
target encoding, numeric embeddings) and internal bagging, and rivals GBDTs on tabular data while making
*different* errors — so it lifts the blend even if its solo score only matches the GBDTs.

Runs on the box GPU (`device='cuda'`). Saves OOF/test/metrics in the standard artifact format so
blend/stack/submit consume it by name. Class imbalance is handled post-hoc by the balanced-accuracy
decision tuning in `metric.tune_weights` (same as the GBDTs), plus optional balanced sample weights.

    scripts/remote_run.sh run python -m src.models_realmlp --seeds 3 --tag realmlp
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
from .metric import score_proba, tune_weights

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
N = D.N_CLASSES


def build_realmlp(device: str, seed: int, n_cv: int, batch_size: int = 2048,
                  n_epochs: int | None = None, val_metric: str = "1-balanced_accuracy", n_ens: int = 8):
    from pytabkit import RealMLP_TD_Classifier
    # TD = "tuned defaults". Two levers that matter for our imbalanced balanced-accuracy target
    # (matching the strong public-notebook RealMLP): early-stop / select on **balanced_accuracy**
    # (not cross-entropy — the model is otherwise majority-biased, raw bal-acc ~0.87), and an
    # ensemble (`n_ens`, notebook used 8). Larger batch keeps the GPU busy on 690k rows.
    kw = dict(device=device, random_state=seed, n_cv=n_cv, val_metric_name=val_metric,
              n_ens=n_ens, batch_size=batch_size)
    if n_epochs is not None:
        kw["n_epochs"] = n_epochs
    return RealMLP_TD_Classifier(**kw)


def _oversample(tr: np.ndarray, y: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Upsample minority-class train indices toward the majority count (approximates class weighting,
    which pytabkit RealMLP doesn't expose). ratio=1.0 → fully balanced; 0.5 → halfway."""
    rng = np.random.default_rng(seed)
    cnt = np.bincount(y[tr], minlength=N)
    target = cnt.max()
    out = [tr]
    for c in range(N):
        idx = tr[y[tr] == c]
        need = int((target - len(idx)) * ratio)
        if need > 0 and len(idx) > 0:
            out.append(rng.choice(idx, size=need, replace=True))
    return np.concatenate(out)


def run_realmlp(sample: int | None = None, n_splits: int = 5, tag: str = "realmlp",
                seeds: int = 1, seed_base: int = 42, n_cv: int = 1, balanced: bool = True,
                batch_size: int = 2048, n_epochs: int | None = None,
                val_metric: str = "cross_entropy", n_ens: int = 8, oversample: float = 0.0) -> dict:
    # RealMLP forbids NaN in continuous columns, so add missing-indicator flags (missingness carries
    # a little signal — esp. bmi) then median-impute numerics; categoricals keep NaN as an "NA" level.
    groups = ("base", "missflag")
    df = add_features(D.load_train(), groups)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    test = add_features(D.load_test(), groups) if not sample else None
    feats, cats = feature_columns(groups), categorical_columns(groups)
    frames = [df] + ([test] if test is not None else [])
    for c in cats:  # align categorical levels (+ an explicit "NA" level for missing)
        lv = pd.unique(pd.concat([f[c].astype("object") for f in frames], ignore_index=True))
        lv = [x for x in lv if pd.notna(x)] + ["NA"]
        dt = pd.CategoricalDtype(categories=lv)
        for f in frames:
            f[c] = f[c].astype("object").fillna("NA").astype(dt)
    med = df[D.NUMERIC].median()
    for f in frames:
        f[D.NUMERIC] = f[D.NUMERIC].fillna(med)
    X = df[feats]
    y = D.encode_target(df[D.TARGET])
    Xte = test[feats] if test is not None else None
    test_id = test[D.ID].to_numpy() if test is not None else None

    if oversample > 0:
        print(f"oversampling minority classes toward balance (ratio={oversample})")
    seed_list = [seed_base + s for s in range(max(1, seeds))]
    n_avg = len(seed_list)
    oof = np.zeros((len(df), N))
    test_proba = np.zeros((len(Xte), N)) if Xte is not None else None
    fold_scores = []
    t0 = time.time()
    for seed in seed_list:
        for k, (tr, va) in enumerate(folds(y, n_splits, seed=seed)):
            est = build_realmlp("cuda", seed, n_cv, batch_size=batch_size, n_epochs=n_epochs,
                                val_metric=val_metric, n_ens=n_ens)
            tri = _oversample(tr, y, oversample, seed) if oversample > 0 else tr
            est.fit(X.iloc[tri], y[tri])
            oof[va] += est.predict_proba(X.iloc[va]) / n_avg
            if test_proba is not None:
                test_proba += est.predict_proba(Xte) / (n_splits * n_avg)
            s = score_proba(y[va], est.predict_proba(X.iloc[va]))
            fold_scores.append(s)
            print(f"  seed {seed} fold {k}: bal_acc(raw)={s:.5f}")

    raw = score_proba(y, oof); w = tune_weights(y, oof); tuned = score_proba(y, oof, w)
    res = {"model": "realmlp", "tag": tag, "oof_bacc_raw": float(raw), "oof_bacc_tuned": float(tuned),
           "decision_weights": [float(x) for x in w], "fold_mean_raw": float(np.mean(fold_scores)),
           "fold_std_raw": float(np.std(fold_scores)), "seeds": n_avg, "n_cv": n_cv,
           "elapsed_s": round(time.time() - t0, 1)}
    print(f"\nOOF balanced accuracy: raw {raw:.5f}  tuned {tuned:.5f}  ({res['elapsed_s']}s)")

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        suf = f"_{tag}" if tag else ""
        np.save(ART / f"realmlp{suf}_oof.npy", oof)
        np.save(ART / f"realmlp{suf}_test.npy", test_proba)
        np.save(ART / "y.npy", y); np.save(ART / "classes.npy", np.array(D.CLASSES, dtype=object))
        np.save(ART / "test_id.npy", test_id)
        (ART / f"realmlp{suf}_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> realmlp{suf}_oof.npy / _test.npy / _metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--tag", default="realmlp")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--n-cv", type=int, default=4, help="internal RealMLP bagging members")
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-epochs", type=int, default=None, help="override RealMLP epochs (default ~256)")
    p.add_argument("--val-metric", default="cross_entropy", help="RealMLP early-stop/select metric")
    p.add_argument("--n-ens", type=int, default=8, help="RealMLP ensemble members")
    p.add_argument("--oversample", type=float, default=0.0,
                   help="upsample minority classes toward balance (0=off, 1=full); approximates class weighting")
    a = p.parse_args()
    run_realmlp(a.sample, a.folds, tag=a.tag, seeds=a.seeds, seed_base=a.seed_base,
                n_cv=a.n_cv, batch_size=a.batch_size, n_epochs=a.n_epochs,
                val_metric=a.val_metric, n_ens=a.n_ens, oversample=a.oversample)
