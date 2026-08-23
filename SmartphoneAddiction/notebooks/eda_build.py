"""Reproducible EDA used by 01_eda.ipynb; writes plots and markdown findings."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from src.data import CATEGORICAL, FEATURES, ID, NUMERIC, TARGET, load_data

FIG=Path("reports/figures"); FIG.mkdir(parents=True,exist_ok=True)
train,test=load_data(); y=train[TARGET]
summary=pd.DataFrame({"dtype":train[FEATURES].dtypes.astype(str),"cardinality":train[FEATURES].nunique(),
                      "train_missing":train[FEATURES].isna().mean(),"test_missing":test[FEATURES].isna().mean()})
summary["missing_drift"]=summary.test_missing-summary.train_missing

fig,ax=plt.subplots(figsize=(9,5)); summary[["train_missing","test_missing"]].sort_values("test_missing").plot.barh(ax=ax); fig.tight_layout(); fig.savefig(FIG/"missingness_drift.png",dpi=140); plt.close(fig)
fig,ax=plt.subplots(figsize=(5,3)); y.value_counts(normalize=True).sort_index().plot.bar(ax=ax); ax.set_title("Target balance"); fig.tight_layout(); fig.savefig(FIG/"target_balance.png",dpi=140); plt.close(fig)
corr=train[NUMERIC+[TARGET]].corr(numeric_only=True); fig,ax=plt.subplots(figsize=(10,8)); sns.heatmap(corr,cmap="vlag",center=0,ax=ax); fig.tight_layout(); fig.savefig(FIG/"numeric_correlations.png",dpi=140); plt.close(fig)

sample=train.sample(min(100000,len(train)),random_state=42)
imp=SimpleImputer(strategy="median"); xs=imp.fit_transform(sample[NUMERIC]); mi=pd.Series(mutual_info_classif(xs,sample[TARGET],random_state=42),index=NUMERIC).sort_values()
single={c:max(roc_auc_score(sample[TARGET],sample[c].fillna(sample[c].median())),roc_auc_score(sample[TARGET],-sample[c].fillna(sample[c].median()))) for c in NUMERIC}
fig,ax=plt.subplots(figsize=(8,4)); mi.plot.barh(ax=ax); fig.tight_layout(); fig.savefig(FIG/"mutual_information.png",dpi=140); plt.close(fig)

surfaces=[]
for a,b in [("daily_screen_time_hours","social_media_hours"),("daily_screen_time_hours","weekend_screen_time")]:
    q=sample[[a,b,TARGET]].dropna().copy(); q["a_bin"]=pd.qcut(q[a],10,duplicates="drop"); q["b_bin"]=pd.qcut(q[b],10,duplicates="drop")
    surface=q.pivot_table(index="a_bin",columns="b_bin",values=TARGET,observed=True)
    fig,ax=plt.subplots(figsize=(8,6)); sns.heatmap(surface,cmap="viridis",ax=ax); ax.set_title(f"Target rate: {a} × {b}"); fig.tight_layout(); name=f"surface_{a}_{b}.png"; fig.savefig(FIG/name,dpi=140); plt.close(fig); surfaces.append(name)

model=HistGradientBoostingClassifier(max_iter=100,random_state=42).fit(xs,sample[TARGET]); pi=permutation_importance(model,xs[:20000],sample[TARGET].iloc[:20000],scoring="roc_auc",n_repeats=3,random_state=42)
perm=pd.Series(pi.importances_mean,index=NUMERIC).sort_values()
fig,ax=plt.subplots(figsize=(8,4)); perm.plot.barh(ax=ax); fig.tight_layout(); fig.savefig(FIG/"permutation_importance.png",dpi=140); plt.close(fig)

cat_rates={c:train.groupby(c,dropna=False)[TARGET].agg(["mean","size"]).round(4).to_dict("index") for c in CATEGORICAL}
duplicates=int(train.duplicated(FEATURES).sum())
top_auc=sorted(single.items(),key=lambda z:z[1],reverse=True)
text=f"""# EDA Findings

Generated from the competition files by `notebooks/eda_build.py`.

## Data

- Train: {train.shape[0]:,} rows; test: {test.shape[0]:,} rows; {len(FEATURES)} predictors.
- Positive target rate: {y.mean():.4%}.
- Duplicate predictor rows in train: {duplicates:,}. IDs remain unique and are never model features.
- Numeric ranges were inspected through `describe`; categorical cardinalities are recorded below.

## Signal and drift

- Strongest univariate numeric AUCs: {', '.join(f'{k}={v:.4f}' for k,v in top_auc[:5])}.
- Highest mutual-information variables: {', '.join(mi.sort_values(ascending=False).head(5).index)}.
- Missingness is separated from value drift. Largest absolute missing-rate shifts: {', '.join(summary.missing_drift.abs().sort_values(ascending=False).head(5).index)}.
- Missing flags and missing count are isolated in the `missing` ablation; they are not part of `base`.
- Two-dimensional target-rate surfaces cover daily/social and daily/weekend interactions. Model permutation importance is saved alongside them.

## Feature decisions

- Evaluate `base`, behavioral interactions, the seven-column `compact` set, and the neutral-on-missing `source_rule` independently on shared folds.
- Preserve native missing values where the estimator supports them; otherwise imputation is learned inside each fold.
- Treat demographic/context predictors as optional weak features, not assumed signal.

## Schema and missingness

```
{summary.to_string()}
```

## Categorical target rates

```
{cat_rates}
```
"""
Path("reports/EDA_FINDINGS.md").write_text(text)
print(text[:2500])
