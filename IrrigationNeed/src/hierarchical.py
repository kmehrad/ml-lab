"""Hierarchical two-binary-classifier decomposition + multi-FE ensembling.

Key insight (from the competition's winning write-up): models almost never
confuse ``Low`` and ``High``. So instead of one 3-way softmax, decompose into two
*easier* binary problems:

* **Model 1** — ``Low`` vs Rest on the whole dataset → ``p1 = P(Low)``.
* **Model 2** — ``Medium`` vs ``High``, trained on the rows Model 1 routes as Rest
  (option B in the write-up: matches the inference distribution), with OOF
  predictions generated for the whole dataset → ``p2 = P(High | not Low)``.

Recombine into class probabilities::

    P(Low)    = p1
    P(Medium) = (1 - p1) * (1 - p2)
    P(High)   = (1 - p1) * p2

Round 2: generate this OOF under several **feature-engineering variants** and
ensemble them (together with the cached multiclass models) via a stacked
logistic meta-learner, then optimize per-class weights for balanced accuracy.

    python -m src.hierarchical --submit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import (
    CATEGORICAL_FEATURES,
    CLASS_ORDER,
    encode_target,
    load_raw,
    split_features_target,
)
from .ensemble import (
    ARTIFACTS_DIR,
    COMPETITION,
    N_CLASSES,
    SEED,
    add_interactions,
    optimize_weights,
    stack,
    write_submission,
)
from .preprocessing import build_preprocessor
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# Feature-engineering variants -> (add_engineered, add_interactions).
FE_VARIANTS = {
    "raw": (False, False),
    "eng": (True, False),
    "eng_int": (True, True),
}
# Base learners used for the two binary stages (one hierarchical model per type).
HIER_MODELS = ("lgbm", "xgb")
# Multiclass OOF caches reused from src.ensemble (Round 1).
MULTICLASS = ("lgbm", "xgb", "histgb", "catboost", "ovr")
LOW, MEDIUM, HIGH = 0, 1, 2


def _binary(kind: str, seed: int):
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary", n_estimators=800, learning_rate=0.03, num_leaves=96,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    if kind == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic", n_estimators=800, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
            enable_categorical=True, random_state=seed, n_jobs=-1,
        )
    raise ValueError(kind)


def _transform(X_fold, pre, interactions):
    out = pre.transform(X_fold)
    return add_interactions(out) if interactions else out


def _fit_binary(model, Xtr, ytr, kind, cat_cols):
    if kind == "lgbm":
        model.fit(Xtr, ytr, categorical_feature=cat_cols)
    else:
        model.fit(Xtr, ytr)
    return model


def hierarchical_oof(X, y, X_test, fe_variant, kind, folds=5, seed=SEED):
    """Leakage-safe hierarchical OOF and fold-averaged test probabilities."""
    add_eng, interactions = FE_VARIANTS[fe_variant]
    oof = np.zeros((len(y), N_CLASSES))
    test = np.zeros((len(X_test), N_CLASSES))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        pre = build_preprocessor("tree", add_engineered=add_eng).fit(X.iloc[tr], y[tr])
        Xtr = _transform(X.iloc[tr], pre, interactions)
        Xva = _transform(X.iloc[va], pre, interactions)
        Xte = _transform(X_test, pre, interactions)
        cat = [c for c in CATEGORICAL_FEATURES if c in Xtr.columns]

        # Stage 1: Low vs Rest.
        m1 = _fit_binary(_binary(kind, seed), Xtr, (y[tr] == LOW).astype(int), kind, cat)
        p1_tr = m1.predict_proba(Xtr)[:, 1]
        p1_va = m1.predict_proba(Xva)[:, 1]
        p1_te = m1.predict_proba(Xte)[:, 1]

        # Stage 2: Medium vs High, trained on rows Model 1 routes as Rest (option B).
        rest = p1_tr < 0.5
        m2 = _fit_binary(_binary(kind, seed), Xtr[rest], (y[tr][rest] == HIGH).astype(int), kind, cat)
        p2_va = m2.predict_proba(Xva)[:, 1]
        p2_te = m2.predict_proba(Xte)[:, 1]

        oof[va, LOW] = p1_va
        oof[va, MEDIUM] = (1 - p1_va) * (1 - p2_va)
        oof[va, HIGH] = (1 - p1_va) * p2_va
        test[:, LOW] += p1_te / folds
        test[:, MEDIUM] += (1 - p1_te) * (1 - p2_te) / folds
        test[:, HIGH] += (1 - p1_te) * p2_te / folds
        print(f"  fold {fold}/{folds} [hier:{kind}:{fe_variant}] "
              f"bal_acc={balanced_accuracy_score(y[va], oof[va].argmax(1)):.4f}", flush=True)
    return oof, test


def _cache(name):
    return ARTIFACTS_DIR / f"oof_{name}.npy", ARTIFACTS_DIR / f"test_{name}.npy"


def run(folds=5, seed=SEED, refresh=False):
    train = load_raw("train")
    test = load_raw("test")
    X, yl = split_features_target(train)
    y = encode_target(yl).to_numpy()
    X_test = test[list(X.columns)]

    components = {}

    # Hierarchical models across FE variants and base learners.
    for fe in FE_VARIANTS:
        for kind in HIER_MODELS:
            name = f"hier_{kind}_{fe}"
            oof_p, test_p = _cache(name)
            if not refresh and oof_p.exists() and test_p.exists():
                components[name] = (np.load(oof_p), np.load(test_p))
                continue
            print(f"== computing {name} ==")
            o, t = hierarchical_oof(X, y, X_test, fe, kind, folds, seed)
            np.save(oof_p, o)
            np.save(test_p, t)
            components[name] = (o, t)

    # Reuse Round-1 multiclass OOF caches if present.
    for m in MULTICLASS:
        oof_p, test_p = _cache(m)
        if oof_p.exists() and test_p.exists():
            components[f"mc_{m}"] = (np.load(oof_p), np.load(test_p))

    print("\n== component balanced accuracy (argmax / tuned) ==")
    for name, (o, _) in components.items():
        _, tuned = optimize_weights(o, y)
        print(f"{name:>20}: argmax={balanced_accuracy_score(y, o.argmax(1)):.5f}  tuned={tuned:.5f}")

    names = list(components)
    oof = {n: components[n][0] for n in names}
    test_p = {n: components[n][1] for n in names}

    print("\n== blends ==")
    hier_names = [n for n in names if n.startswith("hier_")]
    results = {}

    def evaluate(label, blend_oof, blend_test):
        _, tuned = optimize_weights(blend_oof, y)
        print(f"{label:>28}: argmax={balanced_accuracy_score(y, blend_oof.argmax(1)):.5f}  tuned={tuned:.5f}")
        results[label] = (blend_oof, blend_test)

    evaluate("hier_only_mean", np.mean([oof[n] for n in hier_names], axis=0),
             np.mean([test_p[n] for n in hier_names], axis=0))
    evaluate("all_mean", np.mean([oof[n] for n in names], axis=0),
             np.mean([test_p[n] for n in names], axis=0))

    meta_oof_h, meta_test_h = stack(oof, test_p, y, hier_names, folds, seed)
    evaluate("stack_hier", meta_oof_h, meta_test_h)
    meta_oof, meta_test = stack(oof, test_p, y, names, folds, seed)
    evaluate("stack_all", meta_oof, meta_test)

    best = max(results, key=lambda k: optimize_weights(results[k][0], y)[1])
    best_oof, best_test = results[best]
    best_w, best_score = optimize_weights(best_oof, y)
    print(f"\nBEST: {best} tuned balanced acc = {best_score:.5f}  weights={np.round(best_w,3)}")
    return best, best_test, best_w, best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("-m", "--message", default=None)
    args = parser.parse_args()

    best, best_test, best_w, best_score = run(args.folds, args.seed, args.refresh)
    path = write_submission(best_test, best_w, f"hier_{best}")
    if args.submit:
        import subprocess

        msg = args.message or f"Hierarchical multi-FE ensemble ({best}), tuned bal_acc {best_score:.5f}"
        subprocess.run(["kaggle", "competitions", "submit", "-c", COMPETITION,
                        "-f", str(path), "-m", msg], check=True)


if __name__ == "__main__":
    main()
