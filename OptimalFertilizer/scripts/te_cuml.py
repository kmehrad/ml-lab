"""Single-fold config sweep on full data: pick base features (num vs num+cat) and depth.

Builds the cuML order-2 TE once, then fits XGB variants. Cheap decision before full 5-fold runs.
"""
import sys
import numpy as np
import cupy as cp
import cudf
import xgboost as xgb
from cuml.preprocessing import TargetEncoder

from src import data as D
from src.cv import folds
from src.gpu_te import combos, COMBO_FEATURES
from src.metric import mapk3

ORDER = int(sys.argv[1]) if len(sys.argv) > 1 else 2

df = D.load_train().reset_index(drop=True)
for c in D.CATEGORICAL:
    df[c] = df[c].astype("category")
y = D.encode_target(df[D.TARGET])
tr, va = folds(y, 5)[0]
ytr, yva = y[tr], y[va]
gtr = cudf.from_pandas(df[COMBO_FEATURES].iloc[tr].reset_index(drop=True))
gva = cudf.from_pandas(df[COMBO_FEATURES].iloc[va].reset_index(drop=True))
orig = D.load_original()
gor = cudf.from_pandas(orig[COMBO_FEATURES].reset_index(drop=True))
yor = cudf.Series(D.encode_target(orig[D.TARGET]))
ytr_g = cudf.Series(ytr)

num = cp.asarray(df[list(D.NUMERIC)].to_numpy(np.float32))
cat = cp.asarray(np.column_stack([df[c].cat.codes.to_numpy() for c in D.CATEGORICAL]).astype(np.float32))
num_tr, num_va = num[tr], num[va]
cat_tr, cat_va = cat[tr], cat[va]

tr_p, va_p = [], []
for combo in combos(ORDER):
    cols = list(combo)
    for j in range(7):
        te = TargetEncoder(smooth=0, output_type="cupy")
        tr_p.append(te.fit_transform(gtr[cols], (ytr_g == j).astype("float32")).reshape(-1, 1))
        va_p.append(te.transform(gva[cols]).reshape(-1, 1))
        te2 = TargetEncoder(smooth=0, output_type="cupy")
        te2.fit_transform(gor[cols], (yor == j).astype("float32"))
        tr_p.append(te2.transform(gtr[cols]).reshape(-1, 1))
        va_p.append(te2.transform(gva[cols]).reshape(-1, 1))
TE_tr, TE_va = cp.hstack(tr_p), cp.hstack(va_p)
print(f"order={ORDER} TE cols={TE_tr.shape[1]}")

base = {"device": "cuda", "objective": "multi:softprob", "num_class": 7, "tree_method": "hist",
        "eta": 0.03, "eval_metric": "mlogloss", "subsample": 0.8, "colsample_bytree": 0.5}


def fit(Xt, Xv, depth):
    dtr = xgb.QuantileDMatrix(Xt, label=ytr)
    dva = xgb.QuantileDMatrix(Xv, label=yva, ref=dtr)
    bst = xgb.train({**base, "max_depth": depth}, dtr, num_boost_round=4000,
                    evals=[(dva, "v")], early_stopping_rounds=100, verbose_eval=False)
    p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    return mapk3(yva, p), bst.best_iteration


for tag, Xt, Xv in [("num+TE", cp.hstack([num_tr, TE_tr]), cp.hstack([num_va, TE_va])),
                    ("num+cat+TE", cp.hstack([num_tr, cat_tr, TE_tr]),
                     cp.hstack([num_va, cat_va, TE_va]))]:
    for depth in (4, 6, 8):
        s, bi = fit(Xt, Xv, depth)
        print(f"{tag:12s} depth{depth}  valid MAP@3={s:.4f}  best_iter={bi}")
