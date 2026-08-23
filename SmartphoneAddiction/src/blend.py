from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize
from scipy.special import expit
from .artifacts import save_run
from .metric import metrics


def cross_fitted_stack(oof_matrix, y, folds, test_matrix):
    x, y, folds, xt = map(np.asarray, (oof_matrix, y, folds, test_matrix))
    if x.ndim != 2 or xt.ndim != 2 or x.shape[1] != xt.shape[1] or len(y) != len(x):
        raise ValueError("stack inputs do not align")
    out, test = np.zeros(len(y)), np.zeros(len(xt))
    def fit_nonnegative(a, b):
        def loss(v):
            q=np.clip(expit(v[0]+a@v[1:]),1e-9,1-1e-9)
            return -np.mean(b*np.log(q)+(1-b)*np.log(1-q))+1e-6*np.dot(v[1:],v[1:])
        return minimize(loss,np.r_[0.,np.ones(a.shape[1])],bounds=[(None,None)]+[(0,None)]*a.shape[1],method="L-BFGS-B").x
    for fold in np.unique(folds):
        tr, va = folds != fold, folds == fold
        coef=fit_nonnegative(x[tr],y[tr]); out[va]=expit(coef[0]+x[va]@coef[1:]); test += expit(coef[0]+xt@coef[1:]) / len(np.unique(folds))
    return out, test


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--method", choices=["mean","rank","stack"], default="mean"); ap.add_argument("--tag", default="blend")
    a=ap.parse_args(); p=Path("experiments/artifacts")
    o=np.column_stack([np.load(p/f"{t}_oof.npy") for t in a.tags]); te=np.column_stack([np.load(p/f"{t}_test.npy") for t in a.tags]); y=np.load(p/"y.npy")
    if a.method == "mean": pred, test=o.mean(1), te.mean(1)
    elif a.method == "rank": pred=np.column_stack([rankdata(z)/len(z) for z in o.T]).mean(1); test=np.column_stack([rankdata(z)/len(z) for z in te.T]).mean(1)
    else: pred,test=cross_fitted_stack(o,y,np.load(p/"folds.npy"),te)
    meta={**metrics(y,pred),"method":a.method,"members":a.tags}; save_run(p,a.tag,pred,test,meta); print(json.dumps(meta,indent=2))
if __name__ == "__main__": main()
