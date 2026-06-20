"""Train the staged ensemble and persist artifacts.

Run from the project root:

    python -m src.train

Stages: ElasticNet baseline -> LightGBM -> XGBoost -> NNLS blend. Tunes each model
with RandomizedSearchCV on shared folds, evaluates an honest out-of-fold blend, then
refits every model on the full training data and saves models + blend weights.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config, models
from .data import basic_clean, drop_outliers, load_raw
from .features import make_linear_matrix, make_tree_matrix

# How many RandomizedSearchCV samples per model.
N_ITER = {"enet": 40, "lgbm": 50, "xgb": 50}


def _banner(text: str) -> None:
    print("\n" + "=" * 64 + f"\n{text}\n" + "=" * 64)


def main() -> None:
    # 1. Data -> matrices
    train_raw, test_raw = load_raw()
    train = drop_outliers(basic_clean(train_raw))
    test = basic_clean(test_raw)

    Xtr_tree, _, y = make_tree_matrix(train, test)
    Xtr_lin, _, y_lin = make_linear_matrix(train, test, scale=False)
    assert np.allclose(y.values, y_lin.values), "target mismatch between matrices"
    print(f"tree matrix {Xtr_tree.shape} | linear matrix {Xtr_lin.shape} | y {y.shape}")

    builders = {
        "enet": (models.enet_model, Xtr_lin),
        "lgbm": (models.lgbm_model, Xtr_tree),
        "xgb": (models.xgb_model, Xtr_tree),
    }

    # 2. Tune each model (staged)
    fitted, cv_rmse, best_params = {}, {}, {}
    for name in models.MODEL_ORDER:
        build, X = builders[name]
        _banner(f"Tuning {name}  ({N_ITER[name]} iters, {config.N_FOLDS}-fold CV)")
        est, space = build()
        best, params, score = models.tune(est, space, X, y, n_iter=N_ITER[name])
        fitted[name], cv_rmse[name], best_params[name] = best, score, params
        print(f"{name} CV RMSE: {score:.5f}")

    # 3. Ensemble via out-of-fold predictions on the shared folds
    _banner("Ensemble (NNLS blend of OOF predictions)")
    oof_cols = [models.oof_predict(fitted[n], builders[n][1], y)
                for n in models.MODEL_ORDER]
    oof = np.column_stack(oof_cols)
    weights, _, blend_rmse = models.nnls_blend(oof, y)

    print("\nCV RMSE scoreboard")
    for name in models.MODEL_ORDER:
        print(f"  {name:5s} {cv_rmse[name]:.5f}")
    print(f"  {'blend':5s} {blend_rmse:.5f}")
    print("\nblend weights: " + ", ".join(
        f"{n}={w:.3f}" for n, w in zip(models.MODEL_ORDER, weights)))
    best_single = min(cv_rmse.values())
    print(f"blend improves over best single model: "
          f"{blend_rmse <= best_single}  ({best_single - blend_rmse:+.5f})")

    # 4. Refit on full training data and persist
    _banner("Refitting on full data + saving artifacts")
    for name in models.MODEL_ORDER:
        X = builders[name][1]
        fitted[name].fit(X, y)
        joblib.dump(fitted[name], config.MODELS / f"{name}.joblib")
        print(f"saved {config.MODELS / f'{name}.joblib'}")
    joblib.dump(
        {"weights": weights, "order": models.MODEL_ORDER,
         "cv_rmse": cv_rmse, "blend_rmse": blend_rmse},
        config.MODELS / "blend.joblib",
    )
    print(f"saved {config.MODELS / 'blend.joblib'}")

    _write_summary(cv_rmse, blend_rmse, weights, best_params)
    _plot_importance(fitted["lgbm"], Xtr_tree.columns)
    print("\nTraining complete.")


def _write_summary(cv_rmse, blend_rmse, weights, best_params) -> None:
    path = config.OUTPUTS / "cv_summary.txt"
    lines = ["House Prices — CV summary", "=" * 32, "", "CV RMSE (log target):"]
    for name in models.MODEL_ORDER:
        lines.append(f"  {name:5s} {cv_rmse[name]:.5f}")
    lines.append(f"  {'blend':5s} {blend_rmse:.5f}")
    lines.append("")
    lines.append("Blend weights: " + ", ".join(
        f"{n}={w:.3f}" for n, w in zip(models.MODEL_ORDER, weights)))
    lines.append("")
    lines.append("Best params:")
    for name in models.MODEL_ORDER:
        lines.append(f"  {name}: {best_params[name]}")
    path.write_text("\n".join(lines) + "\n")
    print(f"saved {path}")


def _plot_importance(lgbm, columns, top: int = 25) -> None:
    imp = pd.Series(lgbm.feature_importances_, index=columns).sort_values()
    imp = imp.tail(top)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(imp) + 1))
    ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set_title("LightGBM feature importance (gain split count)")
    ax.set_xlabel("importance")
    path = config.OUTPUTS / "feature_importance.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
