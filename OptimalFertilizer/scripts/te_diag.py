"""Rescue hypothesis: bin high-cardinality numerics so combos form multi-row buckets.

Reports feature cardinalities, then tests binned-numeric TE (order 3) vs numerics-only on a full
fold. If binning makes LOO signal survive, valid MAP@3 should beat the ~0.33 numerics-only ref.
"""
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb

from src import data as D
from src.cv import folds
from src import gpu_te
from src.gpu_te import combos, COMBO_FEATURES
from src.metric import mapk3

df = D.load_train().reset_index(drop=True)
y = D.encode_target(df[D.TARGET])
print("cardinalities:", {c: int(df[c].nunique()) for c in COMBO_FEATURES})

tr, va = folds(y, 5)[0]
ytr, yva = y[tr], y[va]
num = cp.asarray(df[list(D.NUMERIC)].to_numpy(np.float32))
num_tr, num_va = num[tr], num[va]

base = {"device": "cuda", "objective": "multi:softprob", "num_class": 7, "tree_method": "hist",
        "eta": 0.03, "eval_metric": "mlogloss", "subsample": 0.8}


def fit(Xt, Xv, params):
    dtr = xgb.QuantileDMatrix(Xt, label=ytr)
    dva = xgb.QuantileDMatrix(Xv, label=yva, ref=dtr)
    bst = xgb.train({**base, **params}, dtr, num_boost_round=3000,
                    evals=[(dva, "v")], early_stopping_rounds=100, verbose_eval=False)
    p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    return mapk3(yva, p), bst.best_iteration


s, bi = fit(num_tr, num_va, {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 5})
print(f"numerics-only  valid MAP@3={s:.4f} best_iter={bi}")

# bin the 6 numerics to q quantile levels; categoricals kept as-is
for q in (8, 16):
    dfb = df.copy()
    for c in D.NUMERIC:
        dfb[c] = pd.qcut(df[c], q=q, labels=False, duplicates="drop").astype("int32")
    print(f"  binned card (q={q}):", {c: int(dfb[c].nunique()) for c in D.NUMERIC})
    for order in (2, 3):
        Xtr, Xva, _, _ = gpu_te.build_te_features(
            dfb[COMBO_FEATURES].iloc[tr], ytr, dfb[COMBO_FEATURES].iloc[va], None,
            combos(order), m=20.0, use_original=True)
        Xt = cp.hstack([num_tr, Xtr]); Xv = cp.hstack([num_va, Xva])
        s, bi = fit(Xt, Xv, {"max_depth": 5, "colsample_bytree": 0.5, "min_child_weight": 50,
                             "reg_lambda": 10})
        print(f"  binned q{q} TE order{order}  valid MAP@3={s:.4f} best_iter={bi} ncols={Xt.shape[1]}")
