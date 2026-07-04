"""Faithful reproduction using cuML's own TargetEncoder (the winner's exact tool).

Removes my hand-rolled LOO as a variable. Builds 7-binary TE for every combo up to `ORDER`,
measures max valid corr, and fits a regularized XGB on numerics+TE vs numerics-only (one full fold).
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
SMOOTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

df = D.load_train().reset_index(drop=True)
y = D.encode_target(df[D.TARGET])
tr, va = folds(y, 5)[0]
ytr, yva = y[tr], y[va]
gtr = cudf.from_pandas(df[COMBO_FEATURES].iloc[tr].reset_index(drop=True))
gva = cudf.from_pandas(df[COMBO_FEATURES].iloc[va].reset_index(drop=True))
ytr_g = cudf.Series(ytr)
num = cp.asarray(df[list(D.NUMERIC)].to_numpy(np.float32))
num_tr, num_va = num[tr], num[va]

cl = combos(ORDER)
print(f"cuML TargetEncoder  order<={ORDER}  combos={len(cl)}  smooth={SMOOTH}  n_folds=4")

cols_tr, cols_va, best_corr = [], [], 0.0
for combo in cl:
    sub_tr = gtr[list(combo)]
    sub_va = gva[list(combo)]
    for j in range(D.N_CLASSES):
        te = TargetEncoder(smooth=SMOOTH, output_type="cupy")
        yj = (ytr_g == j).astype("float32")
        e_tr = te.fit_transform(sub_tr, yj).reshape(-1, 1)
        e_va = te.transform(sub_va).reshape(-1, 1)
        cols_tr.append(e_tr); cols_va.append(e_va)
        c = abs(np.corrcoef(cp.asnumpy(e_va).ravel(), (yva == j).astype(float))[0, 1])
        best_corr = max(best_corr, c if np.isfinite(c) else 0.0)

Xtr = cp.hstack([num_tr] + cols_tr)
Xva = cp.hstack([num_va] + cols_va)
print(f"max valid corr (cuML) = {best_corr:.3f}   Xtr={Xtr.shape}")

base = {"device": "cuda", "objective": "multi:softprob", "num_class": 7, "tree_method": "hist",
        "eta": 0.03, "eval_metric": "mlogloss", "subsample": 0.8}


def fit(Xt, Xv, params):
    dtr = xgb.QuantileDMatrix(Xt, label=ytr)
    dva = xgb.QuantileDMatrix(Xv, label=yva, ref=dtr)
    bst = xgb.train({**base, **params}, dtr, num_boost_round=3000,
                    evals=[(dva, "v")], early_stopping_rounds=100, verbose_eval=False)
    p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    return mapk3(yva, p), bst.best_iteration


s0, b0 = fit(num_tr, num_va, {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 5})
print(f"numerics-only        valid MAP@3={s0:.4f} best_iter={b0}")
for dep, cs, mcw in [(6, 0.5, 20), (4, 0.4, 50)]:
    s, bi = fit(Xtr, Xva, {"max_depth": dep, "colsample_bytree": cs, "min_child_weight": mcw})
    print(f"num+cuML-TE depth{dep} cs{cs} mcw{mcw}  valid MAP@3={s:.4f} best_iter={bi}")
