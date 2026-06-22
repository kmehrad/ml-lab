"""Ensembling, stacking, and decision-threshold optimization.

Goal: push balanced accuracy past the ~0.969 single-model ceiling. The strategy:

1. Train diverse base learners on the *natural* class distribution so their
   ``predict_proba`` outputs are well calibrated (no class weighting).
2. Produce leakage-safe out-of-fold (OOF) probabilities and fold-averaged test
   probabilities in a single pass over the folds (the preprocessor is fit once
   per fold and shared across models).
3. Optimize a per-class weight vector on the OOF probabilities to maximize
   **balanced accuracy** directly — the Bayes rule for this metric reweights the
   posteriors by (roughly) the inverse class prior, so this is the key lever.
4. Blend the base models and stack a logistic meta-learner on their OOF
   probabilities, re-optimizing the weights each time.

Run ``python -m src.ensemble`` to compute everything and pick the best recipe;
add ``--submit`` to upload the winning submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from .data import (
    CATEGORICAL_FEATURES,
    CLASS_ORDER,
    ID_COLUMN,
    TARGET_COLUMN,
    decode_target,
    encode_target,
    load_raw,
    split_features_target,
)
from .preprocessing import build_preprocessor

N_CLASSES = len(CLASS_ORDER)
SEED = 42
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
COMPETITION = "playground-series-s6e4"

BASE_MODELS = ("lgbm", "xgb", "histgb", "catboost")


# --------------------------------------------------------------------------- #
# Extra interaction features (numeric; appended to the tree-encoded frame)
# --------------------------------------------------------------------------- #
def add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Append a handful of physically-motivated interaction terms.

    Operates on the tree-preprocessed frame (raw + engineered columns present).
    Trees ignore unhelpful columns, so this is low-risk extra signal aimed at
    the Medium<->High boundary.
    """
    f = frame
    moisture = f["Soil_Moisture"]
    temp = f["Temperature_C"]
    rain = f["Rainfall_mm"]
    wind = f["Wind_Speed_kmh"]
    et = f["et_demand"]
    out = f.copy()
    out["moisture_x_temp"] = moisture * temp
    out["wind_x_temp"] = wind * temp
    out["rain_per_temp"] = rain / (temp + 1.0)
    out["et_per_moisture"] = et / (moisture + 1.0)
    out["supply_minus_demand"] = (moisture + rain / 50.0) - et
    out["aridity_sq"] = f["aridity_index"] ** 2
    return out


# --------------------------------------------------------------------------- #
# Base model factory (natural distribution -> calibrated probabilities)
# --------------------------------------------------------------------------- #
def _build(name: str, seed: int):
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass", n_estimators=900, learning_rate=0.03,
            num_leaves=128, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob", n_estimators=900, learning_rate=0.03,
            max_depth=7, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=-1,
        )
    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=800, max_leaf_nodes=128, l2_regularization=1.0,
            categorical_features=list(CATEGORICAL_FEATURES), random_state=seed,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass", iterations=800, learning_rate=0.04, depth=7,
            random_seed=seed, thread_count=-1, allow_writing_files=False, verbose=False,
        )
    raise ValueError(f"unknown base model {name!r}")


def _fit_proba(name, X_tr, y_tr, X_eval_list, seed):
    """Fit one base model; return predict_proba for each frame in X_eval_list."""
    if name == "ovr":
        return _fit_ovr_proba(X_tr, y_tr, X_eval_list, seed)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_tr.columns]
    model = _build(name, seed)
    if name == "lgbm":
        model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    elif name == "catboost":
        X_tr = X_tr.assign(**{c: X_tr[c].astype(str) for c in cat_cols})
        X_eval_list = [X.assign(**{c: X[c].astype(str) for c in cat_cols}) for X in X_eval_list]
        model.fit(X_tr, y_tr, cat_features=cat_cols)
    else:  # xgb, histgb handle category dtype natively
        model.fit(X_tr, y_tr)
    return [model.predict_proba(X) for X in X_eval_list]


def _fit_ovr_proba(X_tr, y_tr, X_eval_list, seed):
    """One-vs-rest LightGBM: a dedicated binary model per class.

    Each class gets its own binary learner, so the rare ``High`` boundary is
    modelled directly instead of competing inside a single softmax. The three
    per-class scores are row-normalized into a probability vector. The winning
    write-up combined exactly this with multiclass models.
    """
    from lightgbm import LGBMClassifier

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_tr.columns]
    scores = [np.zeros((len(X), N_CLASSES)) for X in X_eval_list]
    for c in range(N_CLASSES):
        clf = LGBMClassifier(
            objective="binary", n_estimators=700, learning_rate=0.03, num_leaves=96,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        clf.fit(X_tr, (y_tr == c).astype(int), categorical_feature=cat_cols)
        for i, X in enumerate(X_eval_list):
            scores[i][:, c] = clf.predict_proba(X)[:, 1]
    return [s / s.sum(axis=1, keepdims=True) for s in scores]


# --------------------------------------------------------------------------- #
# OOF + test probability generation (single pass over folds)
# --------------------------------------------------------------------------- #
def compute_oof(models=BASE_MODELS, folds=5, seed=SEED, sample=None, refresh=False):
    """Return ``(oof[model], test[model], y)`` probability arrays, cached on disk."""
    train = load_raw("train")
    test = load_raw("test")
    if sample is not None:
        train = train.sample(sample, random_state=seed).reset_index(drop=True)
    X, y_labels = split_features_target(train)
    y = encode_target(y_labels).to_numpy()
    X_test = test[list(X.columns)]

    tag = f"_s{sample}" if sample else ""
    cached = {m: ARTIFACTS_DIR / f"oof_{m}{tag}.npy" for m in models}
    cached_te = {m: ARTIFACTS_DIR / f"test_{m}{tag}.npy" for m in models}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Only (re)compute models that are missing from the cache.
    need = [m for m in models if refresh or not cached[m].exists() or not cached_te[m].exists()]
    if need:
        print(f"computing OOF for: {need} (reusing cache for {[m for m in models if m not in need]})")
        oof_new = {m: np.zeros((len(y), N_CLASSES)) for m in need}
        test_new = {m: np.zeros((len(X_test), N_CLASSES)) for m in need}
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            pre = build_preprocessor("tree").fit(X.iloc[tr_idx], y[tr_idx])
            X_tr = add_interactions(pre.transform(X.iloc[tr_idx]))
            X_va = add_interactions(pre.transform(X.iloc[va_idx]))
            X_te = add_interactions(pre.transform(X_test))
            for m in need:
                va_proba, te_proba = _fit_proba(m, X_tr, y[tr_idx], [X_va, X_te], seed)
                oof_new[m][va_idx] = va_proba
                test_new[m] += te_proba / folds
                print(f"  fold {fold}/{folds} [{m}] "
                      f"bal_acc={balanced_accuracy_score(y[va_idx], va_proba.argmax(1)):.4f}",
                      flush=True)
        for m in need:
            np.save(cached[m], oof_new[m])
            np.save(cached_te[m], test_new[m])

    oof = {m: np.load(cached[m]) for m in models}
    test_p = {m: np.load(cached_te[m]) for m in models}
    np.save(ARTIFACTS_DIR / f"y_oof{tag}.npy", y)
    return oof, test_p, y


# --------------------------------------------------------------------------- #
# Decision-threshold (per-class weight) optimization for balanced accuracy
# --------------------------------------------------------------------------- #
def optimize_weights(proba, y, rounds=4, grid=41):
    """Coordinate-ascent search for class weights maximizing balanced accuracy.

    Prediction rule is ``argmax_c (w_c * proba_c)``. Weights are scale-invariant,
    so ``w`` is renormalized each step. Returns ``(weights, best_balanced_acc)``.
    """
    w = np.ones(N_CLASSES)

    def score(weights):
        return balanced_accuracy_score(y, (proba * weights).argmax(1))

    best = score(w)
    span = 1.0
    for _ in range(rounds):
        for c in range(N_CLASSES):
            candidates = np.geomspace(max(1e-3, w[c] / (1 + span)), w[c] * (1 + span), grid)
            for cand in candidates:
                trial = w.copy()
                trial[c] = cand
                trial /= trial.sum()
                s = score(trial)
                if s > best:
                    best, w = s, trial
        span *= 0.5  # refine
    return w, best


def _report(name, proba, y):
    raw = balanced_accuracy_score(y, proba.argmax(1))
    w, tuned = optimize_weights(proba, y)
    rec = recall_score(y, (proba * w).argmax(1), average=None, labels=range(N_CLASSES))
    pc = " ".join(f"{c}={r:.3f}" for c, r in zip(CLASS_ORDER, rec))
    print(f"{name:>22}: argmax={raw:.5f}  tuned={tuned:.5f}  weights={np.round(w,3)}  [{pc}]")
    return w, tuned


# --------------------------------------------------------------------------- #
# Stacking meta-learner (logistic regression on OOF probabilities)
# --------------------------------------------------------------------------- #
def stack(oof, test_p, y, models, folds=5, seed=SEED):
    """Cross-fitted logistic meta-learner over stacked base OOF probabilities."""
    from sklearn.linear_model import LogisticRegression

    Z = np.hstack([oof[m] for m in models])
    Z_test = np.hstack([test_p[m] for m in models])
    meta_oof = np.zeros((len(y), N_CLASSES))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(Z, y):
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(Z[tr_idx], y[tr_idx])
        meta_oof[va_idx] = lr.predict_proba(Z[va_idx])
    meta_full = LogisticRegression(max_iter=2000, C=1.0).fit(Z, y)
    return meta_oof, meta_full.predict_proba(Z_test)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(models=BASE_MODELS, folds=5, seed=SEED, sample=None, refresh=False):
    oof, test_p, y = compute_oof(models, folds, seed, sample, refresh)
    results = {}

    print("\n== individual base models ==")
    for m in models:
        results[m] = (_report(m, oof[m], y), test_p[m])

    print("\n== equal-weight blend ==")
    blend_oof = np.mean([oof[m] for m in models], axis=0)
    blend_test = np.mean([test_p[m] for m in models], axis=0)
    results["blend_mean"] = (_report("blend_mean", blend_oof, y), blend_test)

    print("\n== stacked logistic meta-learner ==")
    meta_oof, meta_test = stack(oof, test_p, y, models, folds, seed)
    results["stack_logit"] = (_report("stack_logit", meta_oof, y), meta_test)

    # Pick the recipe with the best tuned balanced accuracy.
    best_name = max(results, key=lambda k: results[k][0][1])
    (best_w, best_score), best_test = results[best_name]
    summary = {
        "best_recipe": best_name,
        "tuned_balanced_accuracy": round(float(best_score), 5),
        "weights": [round(float(x), 4) for x in best_w],
        "models": list(models),
        "folds": folds,
        "seed": seed,
        "sample": sample,
        "scores": {k: round(float(v[0][1]), 5) for k, v in results.items()},
    }
    (ARTIFACTS_DIR / f"ensemble_summary{'_s'+str(sample) if sample else ''}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nBEST: {best_name} tuned balanced acc = {best_score:.5f}")
    return summary, best_test, best_w, best_name


def write_submission(test_proba, weights, name):
    test = load_raw("test")
    codes = (test_proba * weights).argmax(1)
    submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].to_numpy(),
                               TARGET_COLUMN: decode_target(codes)})
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample)
    assert set(submission[ID_COLUMN]) == set(sample[ID_COLUMN])
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"ensemble_{name}_submission.csv"
    submission.to_csv(path, index=False)
    print(f"Wrote {path} ({len(submission):,} rows)")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(BASE_MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="recompute OOF even if cached")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("-m", "--message", default=None)
    args = parser.parse_args()

    summary, best_test, best_w, best_name = run(
        tuple(args.models), args.folds, args.seed, args.sample, args.refresh
    )
    path = write_submission(best_test, best_w, best_name)
    if args.submit:
        import subprocess

        msg = args.message or f"Ensemble {best_name}, tuned bal_acc {summary['tuned_balanced_accuracy']}"
        subprocess.run(["kaggle", "competitions", "submit", "-c", COMPETITION,
                        "-f", str(path), "-m", msg], check=True)


if __name__ == "__main__":
    main()
