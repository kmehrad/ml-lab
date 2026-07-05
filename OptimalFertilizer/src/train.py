"""Train a GBDT with StratifiedKFold CV; save OOF + bagged-test probability matrices.

Each base learner writes an OOF probability matrix ``(n_train, 7)`` and a fold-bagged test
matrix ``(n_test, 7)`` to ``experiments/artifacts/``, plus shared ``y.npy`` (int class
codes), ``classes.npy`` (class names), and ``test_id.npy`` — so blending and submission are
decoupled from training. OOF is scored with **MAP@3** (the competition metric); early
stopping uses each library's multiclass log-loss as a proxy.

Usage:
    uv run python -m src.train --model lgbm
    uv run python -m src.train --model lgbm --sample 50000   # smoke test on 50k rows
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
from .metric import mapk3
from .target_encoding import COMBO_FEATURES, build_te_features, combos

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
N_CLASSES = D.N_CLASSES


def build_estimator(model: str, device: str = "cpu", seed: int = 42,
                    depth: int | None = None, trees: int | None = None):
    if model == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=trees or 2000, learning_rate=0.03, num_leaves=2 ** (depth or 6) - 1,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_samples=100, n_jobs=-1, random_state=seed,
            objective="multiclass", num_class=N_CLASSES, verbose=-1,
        )
    if model == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=trees or 2000, learning_rate=0.03, max_depth=depth or 6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=5,
            tree_method="hist", device=device, enable_categorical=True,
            objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
            early_stopping_rounds=100, n_jobs=-1, random_state=seed,
        )
    if model == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
            loss_function="MultiClass", early_stopping_rounds=100, random_seed=seed,
            task_type="GPU" if device == "cuda" else "CPU",
            thread_count=-1, verbose=False,
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
    """Fit with per-library early stopping; return (oof_va_proba, test_proba_or_None, best_iter)."""
    if model == "lgbm":
        import lightgbm as lgb
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                categorical_feature=cats,
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        bi = est.best_iteration_
    elif model == "xgb":
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        bi = est.best_iteration
    elif model == "cat":
        # CatBoost wants categoricals as plain strings, not the pandas `category` dtype.
        Xtr, Xva = Xtr.copy(), Xva.copy()
        for c in cats:
            Xtr[c] = Xtr[c].astype(str)
            Xva[c] = Xva[c].astype(str)
        est.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=cats, verbose=False)
        bi = est.get_best_iteration()
        if Xte is not None:
            Xte = Xte.copy()
            for c in cats:
                Xte[c] = Xte[c].astype(str)
    else:
        raise ValueError(model)
    va_proba = _align_proba(est, est.predict_proba(Xva))
    te_proba = _align_proba(est, est.predict_proba(Xte)) if Xte is not None else None
    return va_proba, te_proba, bi


def _to_numpy(a):
    """cupy/numpy array -> numpy (no-op for numpy). Keeps _align_proba/mapk3 on the host."""
    if a is None:
        return None
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _fit_predict_numeric(model, est, Xtr, ytr, Xva, yva, Xte):
    """Early-stopped fit on dense numeric arrays (TE path — no categorical dtype).

    ``Xtr/Xva/Xte`` may be numpy (CPU backend) or cupy (GPU backend); xgboost accepts either
    when ``device`` matches. Probabilities are pulled back to the host for scoring/saving.
    """
    if model == "xgb":
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        bi = est.best_iteration
    elif model == "lgbm":
        import lightgbm as lgb
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        bi = est.best_iteration_
    else:
        raise ValueError(f"TE path supports lgbm/xgb, not {model}")
    va_proba = _align_proba(est, _to_numpy(est.predict_proba(Xva)))
    te_proba = _align_proba(est, _to_numpy(est.predict_proba(Xte))) if Xte is not None else None
    return va_proba, te_proba, bi


def run_cv(model: str, sample: int | None = None, n_splits: int = 5,
           groups=("base",), tag: str = "", augment: bool = False,
           device: str = "cpu", te_order: int = 0, te_m: float = 10.0,
           te_base: bool = True, backend: str = "cpu", te_original: bool = True,
           te_cats: bool = False, seeds: int = 1, seed_base: int = 42,
           depth: int | None = None, trees: int | None = None) -> dict:
    groups = tuple(groups)
    if backend == "gpu" and device == "cpu":
        device = "cuda"                      # GPU TE backend implies GPU training
    df = add_features(D.load_train(), groups)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    print(f"train rows={len(df):,}  model={model}  features={list(groups)}"
          + (f"  augment={augment}" if augment else "") + (f"  tag={tag}" if tag else ""))

    feats = feature_columns(groups)
    cats = categorical_columns(groups)

    # Original UCI seed rows to append to each TRAINING fold only (never validation/test).
    Xorig = yorig = None
    if augment:
        orig = add_features(D.load_original(), groups)
        Xorig, yorig = orig[feats], D.encode_target(orig[D.TARGET])
        print(f"augmenting each train fold with {len(orig)} original UCI rows")

    # Test set (fold-bagged). Skipped for smoke tests.
    test = Xte = test_id = None
    if not sample:
        test = add_features(D.load_test(), groups)
        # Align categorical levels across train/test so category codes match at predict time.
        for c in cats:
            levels = pd.unique(pd.concat([df[c].astype("object"), test[c].astype("object")],
                                         ignore_index=True))
            dtype = pd.CategoricalDtype(categories=levels)
            df[c] = df[c].astype("object").astype(dtype)
            test[c] = test[c].astype("object").astype(dtype)
        test_id = test[D.ID].to_numpy()
        Xte = test[feats]
        test_proba = np.zeros((len(test), N_CLASSES))

    X = df[feats]
    y = D.encode_target(df[D.TARGET])
    # Match original-row categoricals to df's (aligned) dtypes so category codes line up.
    if augment:
        for c in cats:
            Xorig[c] = Xorig[c].astype("object").astype(df[c].dtype)

    combo_list = combos(te_order) if te_order else []
    _build_te = build_te_features
    xp = np
    if te_order:
        if backend == "gpu":
            from . import gpu_te
            import cupy as cp
            _build_te = gpu_te.build_te_features
            xp = cp
        base_te = df[COMBO_FEATURES]
        base_te_test = test[COMBO_FEATURES] if test is not None else None

        def _raw_base(frame):
            """Raw passthrough features hstacked with the TE columns: numerics (+ categorical
            codes on the gpu path, matching the winner keeping raw features alongside TE)."""
            arr = frame[list(D.NUMERIC)].to_numpy(np.float32)
            if backend == "gpu" and te_cats:
                codes = np.column_stack([frame[c].cat.codes.to_numpy()
                                         for c in D.CATEGORICAL]).astype(np.float32)
                arr = np.hstack([arr, codes])
            return xp.asarray(arr)

        base_num = _raw_base(df) if te_base else None
        base_num_test = _raw_base(test) if (te_base and test is not None) else None
        per_combo = N_CLASSES * (2 if (backend == "gpu" and te_original) else 1)
        print(f"target encoding [{backend}]: order<={te_order} -> {len(combo_list)} combos "
              f"x {per_combo} = {len(combo_list) * per_combo} TE features  (m={te_m}, "
              f"base_numerics={te_base}"
              + (f", original_cols={te_original}" if backend == 'gpu' else "") + ")")

    # Seed-averaging (RepeatedKFold): MAP@3 is seed-sensitive; averaging probabilities across
    # independent (split seed + estimator seed) runs before top-3 is a cheap, robust lift.
    seed_list = [seed_base + s for s in range(max(1, seeds))]
    n_avg = len(seed_list)
    oof = np.zeros((len(df), N_CLASSES))
    fold_scores, seed_scores = [], []
    t0 = time.time()
    for seed in seed_list:
        oof_s = np.zeros((len(df), N_CLASSES))
        for k, (tr, va) in enumerate(folds(y, n_splits, seed=seed)):
            est = build_estimator(model, device=device, seed=seed, depth=depth, trees=trees)
            if te_order:
                te_kw = {"use_original": te_original} if backend == "gpu" else {}
                Xtr_te, Xva_te, Xte_te, _ = _build_te(
                    base_te.iloc[tr], y[tr], base_te.iloc[va], base_te_test, combo_list, te_m, **te_kw)
                if te_base:
                    Xtr_te = xp.hstack([base_num[tr], Xtr_te])
                    Xva_te = xp.hstack([base_num[va], Xva_te])
                    Xte_te = xp.hstack([base_num_test, Xte_te]) if Xte_te is not None else None
                va_proba, te_proba, bi = _fit_predict_numeric(
                    model, est, Xtr_te, y[tr], Xva_te, y[va], Xte_te)
            else:
                Xtr, ytr = X.iloc[tr], y[tr]
                if augment:
                    Xtr = pd.concat([Xtr, Xorig], ignore_index=True)
                    ytr = np.concatenate([ytr, yorig])
                va_proba, te_proba, bi = _fit_predict(
                    model, est, Xtr, ytr, X.iloc[va], y[va], Xte, cats)
            oof_s[va] = va_proba
            if Xte is not None:
                test_proba += te_proba / (n_splits * n_avg)
            score = mapk3(y[va], va_proba)
            fold_scores.append(score)
            pre = f"seed {seed} " if n_avg > 1 else ""
            print(f"  {pre}fold {k}: MAP@3={score:.5f}  (best_iter={bi})")
        oof += oof_s / n_avg
        if n_avg > 1:
            seed_scores.append(float(mapk3(y, oof_s)))
            print(f"  seed {seed} OOF MAP@3={seed_scores[-1]:.5f}   "
                  f"running-avg={mapk3(y, oof * n_avg / (seed_list.index(seed) + 1)):.5f}")

    oof_map3 = mapk3(y, oof)
    if te_order:
        per_combo = N_CLASSES * (2 if (backend == "gpu" and te_original) else 1)
        n_features = len(D.NUMERIC) * te_base + len(combo_list) * per_combo
    else:
        n_features = len(feats)
    res = {
        "model": model, "features": list(groups), "tag": tag, "oof_map3": float(oof_map3),
        "fold_mean": float(np.mean(fold_scores)), "fold_std": float(np.std(fold_scores)),
        "fold_scores": [float(s) for s in fold_scores], "n_rows": int(len(df)),
        "n_features": int(n_features), "elapsed_s": round(time.time() - t0, 1),
        "seeds": n_avg, "seed_scores": seed_scores,
    }
    if te_order:
        res["te"] = {"order": te_order, "n_combos": len(combo_list), "m": te_m,
                     "base_numerics": te_base, "backend": backend,
                     "original_cols": bool(backend == "gpu" and te_original)}
    print(f"\nOOF MAP@3 = {oof_map3:.5f}   folds {res['fold_mean']:.5f} +/- {res['fold_std']:.5f}"
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
    p.add_argument("--features", nargs="+", default=["base"],
                   help="feature groups: base soilcross npk env")
    p.add_argument("--tag", default="", help="artifact filename suffix (keeps experiments separate)")
    p.add_argument("--augment", action="store_true", help="append original UCI rows to each train fold")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="xgb/cat compute device")
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"],
                   help="TE compute backend: gpu = cuDF (adds original-data-as-columns), implies device=cuda")
    p.add_argument("--te-order", type=int, default=0,
                   help="target-encode all feature combos up to this order (0=off, 2=pairs, 3=triples, 4=quads)")
    p.add_argument("--te-m", type=float, default=10.0, help="TE Bayesian smoothing weight")
    p.add_argument("--no-te-base", action="store_true", help="drop raw numerics in the TE path")
    p.add_argument("--no-te-original", action="store_true",
                   help="gpu backend: skip the original-data-as-columns TE (synthetic TE only)")
    p.add_argument("--te-cats", action="store_true",
                   help="gpu backend: add raw categorical codes to the base passthrough (sweep: no help)")
    p.add_argument("--seeds", type=int, default=1,
                   help="seed-average N runs (varying fold split + estimator seed) before scoring")
    p.add_argument("--seed-base", type=int, default=42, help="first seed; run uses seed_base..+N-1")
    p.add_argument("--depth", type=int, default=None, help="tree depth override (xgb best=4 on TE)")
    p.add_argument("--trees", type=int, default=None, help="n_estimators override")
    a = p.parse_args()
    run_cv(a.model, a.sample, a.folds, groups=a.features, tag=a.tag, augment=a.augment,
           device=a.device, te_order=a.te_order, te_m=a.te_m, te_base=not a.no_te_base,
           backend=a.backend, te_original=not a.no_te_original, te_cats=a.te_cats,
           seeds=a.seeds, seed_base=a.seed_base, depth=a.depth, trees=a.trees)
