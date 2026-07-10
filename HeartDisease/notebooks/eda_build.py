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
# Predicting Heart Disease (Kaggle PS S6E2) — EDA

**Task:** binary classification — predict `Heart Disease` (`Presence`/`Absence`) from 13
features on the classic UCI Statlog Heart schema (5 numeric, 8 low-cardinality
categorical/ordinal-int). **Metric: ROC AUC** (higher is better; confirmed via the Kaggle
API `evaluationMetric` field, not visible on the JS-rendered web page).

This notebook answers: **target balance · numeric & categorical distributions · feature↔target
signal (correlations, per-category positive rates, mutual information, single-feature baseline
AUC) · duplicates · train/test drift**. Figures → `reports/figures/`; written conclusions →
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

sns.set_theme(style="whitegrid")
RAW = _root / "data" / "raw"
FIG = _root / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

TARGET = "Heart Disease"
NUMERIC = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
CATEGORICAL = ["Sex", "Chest pain type", "FBS over 120", "EKG results", "Exercise angina",
               "Slope of ST", "Number of vessels fluro", "Thallium"]

train = pd.read_csv(RAW / "train.csv")
test = pd.read_csv(RAW / "test.csv")
train["y"] = (train[TARGET] == "Presence").astype(int)
print("train", train.shape, " test", test.shape)
print("missing cells — train:", int(train.isna().sum().sum()), " test:", int(test.isna().sum().sum()))
print("duplicate feature rows in train:",
      int(train.drop(columns=['id', TARGET, 'y']).duplicated().sum()))
print("duplicate feature rows in test:", int(test.drop(columns=['id']).duplicated().sum()))
train.head()
"""),
    md("## Target balance"),
    code(r"""
counts = train[TARGET].value_counts()
print(counts)
print(f"positive rate (Presence) = {train['y'].mean():.4f}")

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(x=counts.index, y=counts.values, ax=ax)
ax.set_title("Heart Disease class balance")
fig.tight_layout(); fig.savefig(FIG / "target_balance.png", dpi=110); plt.show()
"""),
    md("## Numeric feature distributions (by class)"),
    code(r"""
fig, axes = plt.subplots(1, len(NUMERIC), figsize=(4 * len(NUMERIC), 3.5))
for ax, c in zip(axes, NUMERIC):
    sns.kdeplot(data=train, x=c, hue=TARGET, ax=ax, common_norm=False)
    ax.set_title(c)
fig.tight_layout(); fig.savefig(FIG / "numeric_distributions.png", dpi=110); plt.show()
print(train[NUMERIC].describe().round(3))
"""),
    md("## Numeric ↔ target correlation"),
    code(r"""
corr = train[NUMERIC + ["y"]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Pearson correlation (numeric + y)")
fig.tight_layout(); fig.savefig(FIG / "numeric_correlation.png", dpi=110); plt.show()
print("Pearson r with y (Presence=1):")
print(corr["y"].drop("y").sort_values(key=np.abs, ascending=False).round(4))
"""),
    md("## Categorical feature ↔ target (positive rate by level)"),
    code(r"""
fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.ravel()
spread = {}
corr_cat = {}
for ax, c in zip(axes, CATEGORICAL):
    g = train.groupby(c, observed=True)["y"].mean().sort_values()
    spread[c] = float(g.max() - g.min())
    corr_cat[c] = float(train[[c, "y"]].corr().iloc[0, 1])
    sns.barplot(x=g.values, y=g.index.astype(str), ax=ax, orient="h")
    ax.set_title(f"{c}  (spread {spread[c]:.2f})"); ax.set_xlabel("P(Presence)")
for ax in axes[len(CATEGORICAL):]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(FIG / "per_category_rates.png", dpi=110); plt.show()
print("Pearson r with y, and positive-rate spread by categorical (max level - min level):")
for c in sorted(CATEGORICAL, key=lambda c: -abs(corr_cat[c])):
    print(f"  {c:24s} r={corr_cat[c]:+.3f}  spread={spread[c]:.3f}")
"""),
    md("## Mutual information with the target"),
    code(r"""
X = train[NUMERIC + CATEGORICAL].copy()
cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL]
mi = mutual_info_classif(X, train["y"], discrete_features=cat_idx, random_state=42)
mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=mi.values, y=mi.index, ax=ax); ax.set_title("Mutual information with Heart Disease")
fig.tight_layout(); fig.savefig(FIG / "mutual_information.png", dpi=110); plt.show()
print(mi.round(4))
"""),
    md("## Single-feature baseline AUC (signal strength)"),
    code(r"""
# For each feature, score AUC using the per-bin/per-level positive rate as the predicted
# probability (10 quantile bins for numeric, each level for categorical) — an out-of-model
# measure of how much each feature alone explains, on the actual competition metric.
rows = []
for c in NUMERIC:
    b = pd.qcut(train[c], 10, duplicates="drop")
    pred = train.groupby(b, observed=True)["y"].transform("mean")
    rows.append((c, roc_auc_score(train["y"], pred)))
for c in CATEGORICAL:
    pred = train.groupby(c, observed=True)["y"].transform("mean")
    rows.append((c, roc_auc_score(train["y"], pred)))
sf = pd.DataFrame(rows, columns=["feature", "binned_mean_AUC"]).sort_values(
    "binned_mean_AUC", ascending=False)
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
