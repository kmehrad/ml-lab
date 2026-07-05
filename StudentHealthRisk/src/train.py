"""Train a GBDT with StratifiedKFold CV; save OOF + bagged-test probability matrices.

Each base learner writes an OOF probability matrix ``(n_train, 3)`` and a fold-bagged test matrix
``(n_test, 3)`` to ``experiments/artifacts/``, plus shared ``y.npy`` (int class codes),
``classes.npy`` (class names), and ``test_id.npy`` — so blending and submission are decoupled from
training. Early stopping uses each library's multiclass log-loss as a proxy.

The competition metric is **balanced accuracy** (mean per-class recall), a *decision-sensitive*
metric: plain argmax chases the 86%-majority ``at-risk`` class and scores poorly. We therefore report
two OOF numbers — raw argmax and the metric after tuning per-class decision weights on OOF
(``metric.tune_weights``) — and it is the **tuned** score that matters. Weights are re-tuned on the
final (possibly blended) OOF at submit time, so only the proba matrices are persisted here.

Usage:
    uv run python -m src.train --model lgbm
    uv run python -m src.train --model lgbm --sample 50000   # smoke test on 50k rows
    uv run python -m src.train --model xgb --device cuda --tag xgb_gpu
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
N_CLASSES = D.N_CLASSES
_CAT_NA = "__NA__"  # CatBoost placeholder for missing categoricals


def build_estimator(model: str, device: str = "cpu", seed: int = 42,
                    depth: int | None = None, trees: int | None = None, lr: float | None = None):
    # class_weight="balanced" aligns the training objective with the balanced-accuracy metric.
    lr = lr or 0.03
    if model == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=trees or 2000, learning_rate=lr, num_leaves=2 ** (depth or 6) - 1,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_samples=100, n_jobs=-1, random_state=seed,
            objective="multiclass", num_class=N_CLASSES, class_weight="balanced", verbose=-1,
        )
    if model == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=trees or 2000, learning_rate=lr, max_depth=depth or 6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=5,
            tree_method="hist", device=device, enable_categorical=True,
            objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
            early_stopping_rounds=100, n_jobs=-1, random_state=seed,
        )
    if model == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=trees or 2000, learning_rate=lr, depth=depth or 6, l2_leaf_reg=3.0,
            loss_function="MultiClass", auto_class_weights="Balanced",
            early_stopping_rounds=100, random_seed=seed,
            task_type="GPU" if device == "cuda" else "CPU", thread_count=-1, verbose=False,
        )
    raise ValueError(f"unknown model {model}")


def _align_proba(est, proba: np.ndarray) -> np.ndarray:
    """Reorder predict_proba columns to canonical class codes 0..N-1 (defensive)."""
    classes = np.asarray(est.classes_).ravel().astype(int)
    if list(classes) == list(range(N_CLASSES)):
        return proba
    out = np.zeros_like(proba)
    out[:, classes] = proba
    return out


def _fit_predict(model, est, Xtr, ytr, Xva, yva, Xte, cats):
    """Fit with per-library early stopping; return (va_proba, test_proba_or_None, best_iter)."""
    if model == "lgbm":
        import lightgbm as lgb
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], categorical_feature=cats,
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        bi = est.best_iteration_
    elif model == "xgb":
        # XGBoost has no class_weight param; pass balanced per-sample weights to match lgbm/cat.
        counts = np.bincount(ytr, minlength=N_CLASSES).astype(np.float64)
        cw = len(ytr) / (N_CLASSES * np.clip(counts, 1, None))
        est.fit(Xtr, ytr, sample_weight=cw[ytr], eval_set=[(Xva, yva)], verbose=False)
        bi = est.best_iteration
    elif model == "cat":
        # CatBoost wants categoricals as plain strings with no NaN; fill and cast.
        Xtr, Xva = Xtr.copy(), Xva.copy()
        for c in cats:
            Xtr[c] = Xtr[c].astype("object").fillna(_CAT_NA).astype(str)
            Xva[c] = Xva[c].astype("object").fillna(_CAT_NA).astype(str)
        est.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=cats, verbose=False)
        bi = est.get_best_iteration()
        if Xte is not None:
            Xte = Xte.copy()
            for c in cats:
                Xte[c] = Xte[c].astype("object").fillna(_CAT_NA).astype(str)
    else:
        raise ValueError(model)
    va_proba = _align_proba(est, est.predict_proba(Xva))
    te_proba = _align_proba(est, est.predict_proba(Xte)) if Xte is not None else None
    return va_proba, te_proba, bi


def _fit_predict_numeric(model, est, Xtr, ytr, Xva, yva, Xte):
    """Early-stopped fit on dense numeric arrays (target-encoding path — no categorical dtype)."""
    if model == "xgb":
        counts = np.bincount(ytr, minlength=N_CLASSES).astype(np.float64)
        cw = len(ytr) / (N_CLASSES * np.clip(counts, 1, None))
        est.fit(Xtr, ytr, sample_weight=cw[ytr], eval_set=[(Xva, yva)], verbose=False)
        bi = est.best_iteration
    elif model == "lgbm":
        import lightgbm as lgb
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        bi = est.best_iteration_
    else:
        raise ValueError(f"TE path supports lgbm/xgb, not {model}")
    va_proba = _align_proba(est, est.predict_proba(Xva))
    te_proba = _align_proba(est, est.predict_proba(Xte)) if Xte is not None else None
    return va_proba, te_proba, bi


def run_cv(model: str, sample: int | None = None, n_splits: int = 5,
           groups=("base",), tag: str = "", device: str = "cpu",
           seeds: int = 1, seed_base: int = 42, augment: bool = False, te_order: int = 0, te_m: float = 20.0,
           depth: int | None = None, trees: int | None = None, lr: float | None = None) -> dict:
    groups = tuple(groups)
    df = add_features(D.load_train(), groups)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    print(f"train rows={len(df):,}  model={model}  features={list(groups)}"
          + (f"  augment" if augment else "") + (f"  tag={tag}" if tag else ""))

    feats = feature_columns(groups)
    cats = categorical_columns(groups)

    # Original real seed rows appended to each TRAIN fold only (never validation/test).
    orig = add_features(D.load_original(), groups) if augment else None

    # Test set (fold-bagged). Skipped for smoke tests.
    test = Xte = test_id = test_proba = None
    if not sample:
        test = add_features(D.load_test(), groups)
        # Align categorical levels across train/test (+ original, if augmenting) so codes match.
        for c in cats:
            frames = [df[c].astype("object"), test[c].astype("object")]
            if orig is not None:
                frames.append(orig[c].astype("object"))
            levels = pd.unique(pd.concat(frames, ignore_index=True))
            levels = [lv for lv in levels if pd.notna(lv)]   # NaN stays a NaN value, not a category
            dtype = pd.CategoricalDtype(categories=levels)
            df[c] = df[c].astype("object").astype(dtype)
            test[c] = test[c].astype("object").astype(dtype)
            if orig is not None:
                orig[c] = orig[c].astype("object").astype(dtype)
        test_id = test[D.ID].to_numpy()
        Xte = test[feats]
        test_proba = np.zeros((len(test), N_CLASSES))

    X = df[feats]
    y = D.encode_target(df[D.TARGET])
    Xorig = orig[feats] if orig is not None else None
    yorig = D.encode_target(orig[D.TARGET]) if orig is not None else None
    if augment:
        print(f"augmenting each train fold with {len(orig):,} original real rows")

    # Target-encoding path: dense numerics + categorical codes + per-fold k-fold-OOF combo TE.
    keys_tr = keys_te = base_dense = base_dense_test = None
    if te_order:
        from .target_encoding import combos, build_keys
        combo_list = combos(te_order)
        keys_tr = build_keys(df, combo_list)
        keys_te = build_keys(test, combo_list) if test is not None else None
        base_dense = np.column_stack(
            [df[D.NUMERIC].to_numpy(np.float32)]
            + [df[c].cat.codes.to_numpy().astype(np.float32).reshape(-1, 1) for c in D.CATEGORICAL])
        if test is not None:
            base_dense_test = np.column_stack(
                [test[D.NUMERIC].to_numpy(np.float32)]
                + [test[c].cat.codes.to_numpy().astype(np.float32).reshape(-1, 1) for c in D.CATEGORICAL])
        print(f"target encoding: order<={te_order} -> {len(combo_list)} combos x {N_CLASSES} "
              f"= {len(combo_list) * N_CLASSES} TE features (m={te_m})")

    # Seed-averaging (RepeatedKFold): average probabilities across independent (split + estimator)
    # seeds before scoring — a cheap, robust lift when the decision is threshold-sensitive.
    seed_list = [seed_base + s for s in range(max(1, seeds))]
    n_avg = len(seed_list)
    oof = np.zeros((len(df), N_CLASSES))
    fold_scores = []
    t0 = time.time()
    for seed in seed_list:
        oof_s = np.zeros((len(df), N_CLASSES))
        for k, (tr, va) in enumerate(folds(y, n_splits, seed=seed)):
            est = build_estimator(model, device=device, seed=seed, depth=depth, trees=trees, lr=lr)
            if te_order:  # dense TE path (leak-safe per-fold combo target encoding)
                from .target_encoding import build_te_features
                te_tr, te_va, te_te = build_te_features(
                    keys_tr.iloc[tr], y[tr], keys_tr.iloc[va], keys_te, m=te_m, seed=seed)
                Xtr_d = np.hstack([base_dense[tr], te_tr])
                Xva_d = np.hstack([base_dense[va], te_va])
                Xte_d = np.hstack([base_dense_test, te_te]) if base_dense_test is not None else None
                va_proba, te_proba, bi = _fit_predict_numeric(
                    model, est, Xtr_d, y[tr], Xva_d, y[va], Xte_d)
            else:
                Xtr, ytr = X.iloc[tr], y[tr]
                if augment:  # original rows join TRAIN only — validation stays purely synthetic
                    Xtr = pd.concat([Xtr, Xorig], ignore_index=True)
                    ytr = np.concatenate([ytr, yorig])
                va_proba, te_proba, bi = _fit_predict(
                    model, est, Xtr, ytr, X.iloc[va], y[va], Xte, cats)
            oof_s[va] = va_proba
            if Xte is not None:
                test_proba += te_proba / (n_splits * n_avg)
            score = score_proba(y[va], va_proba)  # raw-argmax balanced accuracy for the fold
            fold_scores.append(score)
            pre = f"seed {seed} " if n_avg > 1 else ""
            print(f"  {pre}fold {k}: bal_acc(raw)={score:.5f}  (best_iter={bi})")
        oof += oof_s / n_avg

    raw_bacc = score_proba(y, oof)
    weights = tune_weights(y, oof)
    tuned_bacc = score_proba(y, oof, weights)
    res = {
        "model": model, "features": list(groups), "tag": tag, "augment": bool(augment),
        "depth": depth, "trees": trees, "lr": lr,
        "oof_bacc_raw": float(raw_bacc), "oof_bacc_tuned": float(tuned_bacc),
        "decision_weights": [float(w) for w in weights],
        "fold_mean_raw": float(np.mean(fold_scores)), "fold_std_raw": float(np.std(fold_scores)),
        "fold_scores_raw": [float(s) for s in fold_scores],
        "n_rows": int(len(df)), "n_features": int(len(feats)),
        "elapsed_s": round(time.time() - t0, 1), "seeds": n_avg,
    }
    print(f"\nOOF balanced accuracy: raw-argmax {raw_bacc:.5f}   tuned {tuned_bacc:.5f}"
          f"   (weights {np.round(weights, 3).tolist()})")
    print(f"raw folds {res['fold_mean_raw']:.5f} +/- {res['fold_std_raw']:.5f}"
          f"   ({res['elapsed_s']}s)")

    if not sample:
        ART.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        np.save(ART / f"{model}{suffix}_oof.npy", oof)
        np.save(ART / f"{model}{suffix}_test.npy", test_proba)
        np.save(ART / "y.npy", y)                       # shared, feature-independent
        np.save(ART / "classes.npy", np.array(D.CLASSES, dtype=object))
        np.save(ART / "test_id.npy", test_id)
        (ART / f"{model}{suffix}_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> {ART}/{model}{suffix}_oof.npy, {model}{suffix}_test.npy, "
              f"{model}{suffix}_metrics.json")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lgbm", choices=["lgbm", "xgb", "cat"])
    p.add_argument("--sample", type=int, default=None, help="limit to N rows (smoke test; no test preds)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--features", nargs="+", default=["base"], help="feature groups: base ratio missflag")
    p.add_argument("--tag", default="", help="artifact filename suffix (keeps experiments separate)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="xgb/cat compute device")
    p.add_argument("--seeds", type=int, default=1, help="seed-average N runs before scoring")
    p.add_argument("--seed-base", type=int, default=42, help="first seed; run uses seed_base..+N-1")
    p.add_argument("--depth", type=int, default=None, help="tree depth override")
    p.add_argument("--trees", type=int, default=None, help="n_estimators override")
    p.add_argument("--lr", type=float, default=None, help="learning-rate override (default 0.03)")
    p.add_argument("--augment", action="store_true", help="append original real rows to each train fold")
    p.add_argument("--te-order", type=int, default=0,
                   help="k-fold OOF target-encode feature combos up to this order (0=off, lgbm/xgb only)")
    p.add_argument("--te-m", type=float, default=20.0, help="TE Bayesian smoothing weight")
    a = p.parse_args()
    run_cv(a.model, a.sample, a.folds, groups=a.features, tag=a.tag, device=a.device,
           seeds=a.seeds, seed_base=a.seed_base, augment=a.augment, te_order=a.te_order, te_m=a.te_m,
           depth=a.depth, trees=a.trees, lr=a.lr)
