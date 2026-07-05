"""Generate ``notebooks/01_eda.ipynb`` programmatically (nbformat), then execute it in-place with
``jupyter nbconvert --to notebook --execute``.

Edit *this* file, not the notebook JSON. Figures are written to ``reports/figures/``; written
conclusions live in ``reports/EDA_FINDINGS.md``.

Usage:
    uv run python notebooks/eda_build.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

The notebook loads the raw CSVs directly (kept self-contained) and defines a small inline
balanced-accuracy helper so it does not depend on src/.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

cells = [
    md(r"""
# Predicting Student Health Risk (Kaggle PS S6E7) — EDA

**Task:** 3-class classification of `health_condition` (`at-risk`, `fit`, `unhealthy`) from 7 numeric
behavioural + 6 categorical lifestyle features. **Metric: Balanced Accuracy Score** (mean of per-class
recall). Because the target is heavily imbalanced (`at-risk` ~86%), the naive "always at-risk"
prediction scores only **1/3 = 0.333** — every class matters equally, so minority-class recall is
where the score is won.

This notebook answers: **class balance · numeric & categorical distributions · missingness (is it
informative?) · feature↔target signal (per-class means, mutual information, single-feature balanced
accuracy) · duplicates & label-conflict ceiling · train/test drift · correlation**. Figures →
`reports/figures/`; written conclusions → `reports/EDA_FINDINGS.md`.
"""),
    code(r"""
%matplotlib inline
import pathlib
_root = pathlib.Path.cwd()
if not (_root / "data").exists():   # notebook runs from notebooks/
    _root = _root.parent

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score

sns.set_theme(style="whitegrid")
RAW = _root / "data" / "raw"
FIG = _root / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

TARGET = "health_condition"
CATEGORICAL = ["diet_type", "stress_level", "sleep_quality",
               "physical_activity_level", "smoking_alcohol", "gender"]
NUMERIC = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
           "step_count", "exercise_duration", "water_intake"]

train = pd.read_csv(RAW / "train.csv")
test = pd.read_csv(RAW / "test.csv")
sample = pd.read_csv(RAW / "sample_submission.csv")
print("train:", train.shape, "| test:", test.shape, "| sample_submission:", sample.shape)
CLASSES = sorted(train[TARGET].unique())
print(f"{len(CLASSES)} classes:", CLASSES)
y = train[TARGET].values
code_of = {c: i for i, c in enumerate(CLASSES)}
yc = np.array([code_of[v] for v in y])
prior = train[TARGET].value_counts(normalize=True).reindex(CLASSES).values
print("naive 'always-majority' balanced accuracy = 1/n_classes =", round(1/len(CLASSES), 4))
"""),
    md("## 1. Class balance\nHeavy imbalance is the defining feature of this task under a per-class-recall metric."),
    code(r"""
counts = train[TARGET].value_counts().reindex(CLASSES)
print((counts / len(train) * 100).round(2).astype(str).add(" %").to_string())
print(f"\nmax/min class ratio: {counts.max()/counts.min():.1f}")
fig, ax = plt.subplots(figsize=(7, 4))
counts.sort_values(ascending=False).plot.bar(ax=ax, color="teal")
ax.set(title="health_condition class balance (train)", xlabel="class", ylabel="rows")
fig.tight_layout(); fig.savefig(FIG / "class_balance.png", dpi=110); plt.show()
"""),
    md("## 2. Numeric feature distributions\nRanges, shape, and how they differ by class (violin by target)."),
    code(r"""
print(train[NUMERIC].describe().round(2).to_string())
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), NUMERIC):
    ax.hist(train[col].dropna(), bins=40, color="steelblue")
    ax.set(title=col, ylabel="rows")
axes.ravel()[-1].axis("off")
fig.suptitle("Numeric feature distributions (train)", y=1.02)
fig.tight_layout(); fig.savefig(FIG / "numeric_distributions.png", dpi=110); plt.show()
"""),
    md("## 3. Categorical cardinality & frequency\nAll six categoricals are 3-level. Check level frequency."),
    code(r"""
for col in CATEGORICAL:
    vc = train[col].value_counts(dropna=False)
    print(f"{col}: {train[col].nunique()} levels (+NaN {train[col].isna().mean()*100:.1f}%)")
    print((vc / len(train) * 100).round(2).astype(str).add(" %").to_string(), "\n")
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.ravel(), CATEGORICAL):
    train[col].value_counts().plot.bar(ax=ax, color="indianred", title=col)
    ax.set_ylabel("rows")
fig.tight_layout(); fig.savefig(FIG / "categorical_frequency.png", dpi=110); plt.show()
"""),
    md(r"""
## 4. Missingness — is it informative?

~65% of rows have at least one missing field. Two questions: (a) how much is missing per column, and
(b) does the *presence* of a missing value shift the class distribution? If missingness correlates with
the target, explicit `_isna` indicator features (the `missflag` group) should help.
"""),
    code(r"""
miss = train[NUMERIC + CATEGORICAL].isna().mean().sort_values(ascending=False)
print("per-column missing rate:\n", (miss*100).round(2).astype(str).add(" %").to_string())
print(f"\nrows with >=1 missing: {train[NUMERIC+CATEGORICAL].isna().any(axis=1).mean()*100:.1f}%")

# class balance among rows where a column is missing vs present -> lift on the minority classes
rows = []
base = train[TARGET].value_counts(normalize=True).reindex(CLASSES)
for c in NUMERIC + CATEGORICAL:
    m = train[c].isna()
    if m.sum() == 0: continue
    p_missing = train.loc[m, TARGET].value_counts(normalize=True).reindex(CLASSES).fillna(0)
    rows.append([c] + [round(p_missing[k] - base[k], 4) for k in CLASSES])
delta = pd.DataFrame(rows, columns=["column"] + [f"d_{k}" for k in CLASSES]) \
    .set_index("column")
print("\nP(class | column missing) - P(class)   (biggest abs shift = most informative missingness):")
print(delta.reindex(delta.abs().max(axis=1).sort_values(ascending=False).index).round(4).to_string())
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(delta, annot=True, fmt="+.3f", cmap="coolwarm", center=0, ax=ax)
ax.set(title="Class-share shift when a column is missing")
fig.tight_layout(); fig.savefig(FIG / "missingness_signal.png", dpi=110); plt.show()
"""),
    md(r"""
## 5. Feature ↔ target signal

Three complementary views on how well single features separate the classes under **balanced accuracy**:
1. **Per-class numeric means** (z-scored across classes) — a heatmap of which features move with class.
2. **Mutual information** `I(feature; target)` (subsampled).
3. **Single-feature balanced accuracy** — bin the feature, estimate `P(class | bin)`, predict the
   prior-corrected argmax (`argmax_c P(c|bin)/P(c)` — the balanced-accuracy-optimal rule), and score.
   The lift over the 0.333 floor is the honest single-feature signal.
"""),
    code(r"""
# 5.1 per-class z-scored numeric means
means = train.groupby(TARGET)[NUMERIC].mean()
z = (means - means.mean(0)) / means.std(0)
fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(z.loc[CLASSES], annot=True, fmt="+.2f", cmap="coolwarm", center=0, ax=ax,
            cbar_kws={"label": "z-scored class-mean"})
ax.set(title="Per-class numeric feature means (z-scored across classes)", xlabel="feature", ylabel=TARGET)
fig.tight_layout(); fig.savefig(FIG / "per_class_means.png", dpi=110); plt.show()
print("range of z-scored class means per feature (max-min):")
print((z.max() - z.min()).round(2).sort_values(ascending=False).to_string())
"""),
    code(r"""
# 5.2 mutual information (subsample; median-impute numerics, code categoricals, NaN as its own code)
rng = np.random.default_rng(42)
sub = rng.choice(len(train), size=min(150_000, len(train)), replace=False)
Xmi = train.iloc[sub]
feat_cols = NUMERIC + CATEGORICAL
Xenc = pd.DataFrame(index=Xmi.index); discrete = []
for c in feat_cols:
    if c in CATEGORICAL:
        Xenc[c] = Xmi[c].astype("category").cat.codes; discrete.append(True)   # NaN -> -1
    else:
        Xenc[c] = Xmi[c].fillna(Xmi[c].median()); discrete.append(False)
mi = mutual_info_classif(Xenc.values, Xmi[TARGET].values, discrete_features=discrete, random_state=42)
mi_s = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
print("mutual information I(feature; target) [nats]:\n", mi_s.round(5).to_string())
fig, ax = plt.subplots(figsize=(9, 4)); mi_s.plot.bar(ax=ax, color="slateblue")
ax.set(title="Mutual information with target", ylabel="MI (nats)")
fig.tight_layout(); fig.savefig(FIG / "mutual_information.png", dpi=110); plt.show()
"""),
    code(r"""
# 5.3 single-feature balanced accuracy (prior-corrected argmax)
def single_feature_bacc(col, bins=None):
    s = train[col]
    if bins:
        key = pd.cut(s, bins=bins, labels=False, include_lowest=True)
    else:
        key = s.astype("category").cat.codes    # NaN -> -1 (its own bin)
    key = key.fillna(-1)
    ct = pd.crosstab(key, train[TARGET]).reindex(columns=CLASSES).fillna(0)
    P = (ct.values + 1e-9); P = P / P.sum(1, keepdims=True)
    corrected = P / prior                        # balanced-accuracy-optimal correction
    pred_code = corrected.argmax(1)
    lut = {k: pred_code[i] for i, k in enumerate(ct.index)}
    pred = np.array([lut.get(k, np.argmax(1/prior)) for k in key.values])
    return balanced_accuracy_score(yc, pred)

rows = [("(naive floor)", 1/len(CLASSES))]
for c in CATEGORICAL:
    rows.append((c, single_feature_bacc(c)))
for c in NUMERIC:
    rows.append((c + " (10 bins)", single_feature_bacc(c, bins=10)))
res = pd.DataFrame(rows, columns=["feature", "train bal_acc"]).sort_values("train bal_acc", ascending=False)
res["lift vs floor"] = res["train bal_acc"] - 1/len(CLASSES)
print(res.to_string(index=False))
print("\nNOTE: in-sample (train) balanced accuracy — an upper bound; OOF will be a touch lower.")
"""),
    md("## 6. Duplicates & label-conflict ceiling\nExact-duplicate feature rows, and feature vectors mapping to >1 class (irreducible ambiguity)."),
    code(r"""
feat_cols = NUMERIC + CATEGORICAL
dup = train.duplicated(subset=feat_cols).sum()
print(f"exact-duplicate feature rows: {dup} ({dup/len(train)*100:.2f}%)")
grp = train.groupby(feat_cols, dropna=False, observed=True)[TARGET].nunique()
ambig_rows = int(train.groupby(feat_cols, dropna=False, observed=True)[TARGET]
                 .transform("nunique").gt(1).sum())
print(f"distinct feature vectors: {grp.shape[0]:,}")
print(f"vectors mapping to >1 class: {(grp>1).sum():,} groups, {ambig_rows:,} rows "
      f"({ambig_rows/len(train)*100:.2f}%)")
"""),
    md("## 7. Train vs test drift\nTest labels are hidden; compare feature distributions (KS numeric, share deltas categorical, missing-rate deltas)."),
    code(r"""
ks = {c: stats.ks_2samp(train[c].dropna(), test[c].dropna()).pvalue for c in NUMERIC}
print("numeric KS train-vs-test p-values:")
for c, p in ks.items(): print(f"  {c:20s} p={p:.3f}")
print("features with KS p<0.01:", sum(p < 0.01 for p in ks.values()), "/", len(ks))
for c in CATEGORICAL:
    d = (train[c].value_counts(normalize=True) - test[c].value_counts(normalize=True)).abs().max()
    print(f"{c}: max |share delta| = {d*100:.3f} pp")
mr = pd.DataFrame({"train_miss": train[feat_cols].isna().mean(),
                   "test_miss": test[feat_cols].isna().mean()})
mr["delta_pp"] = (mr["test_miss"] - mr["train_miss"]) * 100
print("\nmissing-rate drift (pp):\n", mr["delta_pp"].round(3).to_string())

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), NUMERIC):
    ax.hist(train[col].dropna(), bins=30, alpha=0.5, density=True, label="train")
    ax.hist(test[col].dropna(), bins=30, alpha=0.5, density=True, label="test")
    ax.set(title=f"{col} (KS p={ks[col]:.2f})"); ax.legend(fontsize=7)
axes.ravel()[-1].axis("off")
fig.suptitle("Train vs test numeric distributions", y=1.02)
fig.tight_layout(); fig.savefig(FIG / "train_test_drift.png", dpi=110); plt.show()
"""),
    md("## 8. Correlation\nLinear correlation among numeric features (redundancy is worth knowing even though trees use interactions)."),
    code(r"""
corr = train[NUMERIC].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt="+.2f", cmap="coolwarm", center=0, square=True, ax=ax)
ax.set(title="Numeric feature correlation (train)")
fig.tight_layout(); fig.savefig(FIG / "numeric_correlation.png", dpi=110); plt.show()
print("max |off-diagonal| correlation:",
      round(corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max(), 3))
"""),
    md("## Summary\nKey numbers and modeling implications are written up in **`reports/EDA_FINDINGS.md`**."),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("01_eda.ipynb")
nbf.write(nb, out)
print("wrote", out)
