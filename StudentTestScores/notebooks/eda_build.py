"""Generate ``notebooks/01_eda.ipynb`` programmatically (nbformat), then execute
it in-place with ``jupyter nbconvert --to notebook --execute``.

Edit *this* file, not the notebook JSON. Figures are written to
``reports/figures/``; written conclusions live in ``reports/EDA_FINDINGS.md``.

Usage:
    uv run python notebooks/eda_build.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

Note: ``src/`` schema constants are imported, but the notebook loads the raw CSVs
directly so it stays runnable independently of the caching loaders.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

cells = [
    md(r"""
# Predicting Student Test Scores (Kaggle PS S6E1) — EDA

**Task:** regression — predict the continuous `exam_score` (range ~19.6–100) from 11 features
(4 numeric, 7 low-cardinality categorical). **Metric: RMSE** (lower is better); the naive floor is
predicting the mean (RMSE = target std ≈ 18.9).

This notebook answers: **target distribution · numeric & categorical distributions · feature↔target
signal (correlations, per-category means, mutual information, single-feature baseline RMSE) ·
duplicates · train/test drift**. Figures → `reports/figures/`; written conclusions →
`reports/EDA_FINDINGS.md`.
"""),
    code(r"""
%matplotlib inline
import pathlib, sys
_root = pathlib.Path.cwd()
if not (_root / "data").exists():   # notebook runs from notebooks/
    _root = _root.parent
sys.path.insert(0, str(_root))      # make `src` importable regardless of cwd

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

sns.set_theme(style="whitegrid")
RAW = _root / "data" / "raw"
FIG = _root / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

TARGET = "exam_score"
NUMERIC = ["age", "study_hours", "class_attendance", "sleep_hours"]
CATEGORICAL = ["gender", "course", "internet_access", "sleep_quality",
               "study_method", "facility_rating", "exam_difficulty"]

train = pd.read_csv(RAW / "train.csv")
test = pd.read_csv(RAW / "test.csv")
print("train", train.shape, " test", test.shape)
print("missing cells — train:", int(train.isna().sum().sum()), " test:", int(test.isna().sum().sum()))
print("duplicate feature rows in train:",
      int(train.drop(columns=['id', TARGET]).duplicated().sum()))
train.head()
"""),
    md("## Target distribution"),
    code(r"""
s = train[TARGET]
print(s.describe())
print(f"skew={s.skew():.3f}  kurtosis={s.kurt():.3f}  naive-mean RMSE (=std) = {s.std():.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(s, bins=60, kde=True, ax=ax[0]); ax[0].set_title("exam_score distribution")
sns.boxplot(x=s, ax=ax[1]); ax[1].set_title("exam_score boxplot")
fig.tight_layout(); fig.savefig(FIG / "target_distribution.png", dpi=110); plt.show()
"""),
    md("## Numeric feature distributions"),
    code(r"""
fig, axes = plt.subplots(1, len(NUMERIC), figsize=(4 * len(NUMERIC), 3.5))
for ax, c in zip(axes, NUMERIC):
    sns.histplot(train[c], bins=40, ax=ax); ax.set_title(c)
fig.tight_layout(); fig.savefig(FIG / "numeric_distributions.png", dpi=110); plt.show()
print(train[NUMERIC].describe().round(3))
"""),
    md("## Numeric ↔ target correlation"),
    code(r"""
corr = train[NUMERIC + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Pearson correlation (numeric + target)")
fig.tight_layout(); fig.savefig(FIG / "numeric_correlation.png", dpi=110); plt.show()
print("Pearson r with exam_score:")
print(corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False).round(4))
"""),
    md("## Categorical group-means of exam_score"),
    code(r"""
fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.ravel()
spread = {}
for ax, c in zip(axes, CATEGORICAL):
    g = train.groupby(c, observed=True)[TARGET].mean().sort_values()
    spread[c] = float(g.max() - g.min())
    sns.barplot(x=g.values, y=g.index, ax=ax, orient="h")
    ax.set_title(f"{c}  (spread {spread[c]:.2f})"); ax.set_xlabel("mean exam_score")
for ax in axes[len(CATEGORICAL):]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(FIG / "per_category_means.png", dpi=110); plt.show()
print("mean-exam_score spread by categorical (max level - min level):")
for c, v in sorted(spread.items(), key=lambda kv: -kv[1]):
    print(f"  {c:18s} {v:.3f}")
"""),
    md("## Mutual information with the target"),
    code(r"""
X = train[NUMERIC + CATEGORICAL].copy()
cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL]
for c in CATEGORICAL:
    X[c] = X[c].astype("category").cat.codes
mi = mutual_info_regression(X, train[TARGET], discrete_features=cat_idx, random_state=42)
mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=mi.values, y=mi.index, ax=ax); ax.set_title("Mutual information with exam_score")
fig.tight_layout(); fig.savefig(FIG / "mutual_information.png", dpi=110); plt.show()
print(mi.round(4))
"""),
    md("## Single-feature baseline RMSE (signal strength)"),
    code(r"""
# For each feature, predict the mean exam_score within bins (10 quantile bins for numeric,
# each level for categorical) — an out-of-model measure of how much each feature alone explains.
from src.metric import rmse
mean_pred = np.full(len(train), train[TARGET].mean())
print(f"predict-global-mean RMSE = {rmse(train[TARGET], mean_pred):.4f}")
rows = []
for c in NUMERIC:
    b = pd.qcut(train[c], 10, duplicates="drop")
    pred = train.groupby(b, observed=True)[TARGET].transform("mean")
    rows.append((c, rmse(train[TARGET], pred)))
for c in CATEGORICAL:
    pred = train.groupby(c, observed=True)[TARGET].transform("mean")
    rows.append((c, rmse(train[TARGET], pred)))
sf = pd.DataFrame(rows, columns=["feature", "binned_mean_RMSE"]).sort_values("binned_mean_RMSE")
print(sf.to_string(index=False))
"""),
    md("## Train/test drift"),
    code(r"""
fig, axes = plt.subplots(1, len(NUMERIC), figsize=(4 * len(NUMERIC), 3.5))
for ax, c in zip(axes, NUMERIC):
    sns.kdeplot(train[c], ax=ax, label="train"); sns.kdeplot(test[c], ax=ax, label="test")
    ax.set_title(c); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "train_test_drift.png", dpi=110); plt.show()
print("numeric mean train vs test:")
print(pd.DataFrame({"train": train[NUMERIC].mean(), "test": test[NUMERIC].mean()}).round(3))
for c in CATEGORICAL:
    tr = train[c].value_counts(normalize=True)
    te = test[c].value_counts(normalize=True)
    print(f"  {c}: max |train-test| level share diff = {(tr - te).abs().max():.4f}")
"""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
out = Path(__file__).resolve().parent / "01_eda.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
