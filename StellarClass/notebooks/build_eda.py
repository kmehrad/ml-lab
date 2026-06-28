"""Generate notebooks/01_eda.ipynb programmatically, then execute it.

Edit this file (not the notebook JSON) to revise the EDA, then run:
    uv run python notebooks/build_eda.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parent / "01_eda.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(src: str):
    return nbf.v4.new_code_cell(src)


cells = [
    md("# Stellar Class (S6E6) — EDA\n\n"
       "3-class classification (GALAXY / QSO / STAR), metric = **Balanced Accuracy**. "
       "Figures saved to `reports/figures/`; written conclusions in `reports/EDA_FINDINGS.md`."),
    code(
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
        "from pathlib import Path\n"
        "import sys; sys.path.insert(0, '..')\n"
        "from src import data as D, features as F\n"
        "FIG = Path('../reports/figures'); FIG.mkdir(parents=True, exist_ok=True)\n"
        "sns.set_theme(style='whitegrid')\n"
        "tr = D.load_train(); te = D.load_test()\n"
        "print('train', tr.shape, 'test', te.shape)\n"
        "tr.head()"
    ),
    md("## 1. Target distribution"),
    code(
        "vc = tr[D.TARGET].value_counts()[D.CLASSES]\n"
        "print(vc); print((vc/len(tr)).round(4))\n"
        "ax = vc.plot.bar(color=['#4C72B0','#DD8452','#55A868']); ax.set_title('Class distribution')\n"
        "plt.tight_layout(); plt.savefig(FIG/'class_distribution.png', dpi=110); plt.show()"
    ),
    md("## 2. Missing values & sentinels (data is clean)"),
    code(
        "print('train NaNs:', int(tr.isna().sum().sum()), ' test NaNs:', int(te.isna().sum().sum()))\n"
        "print(tr[D.NUMERIC_COLS].describe().T[['min','max']])"
    ),
    md("## 3. Numeric features by class (redshift is the key separator)"),
    code(
        "fig, axes = plt.subplots(2, 4, figsize=(18, 8))\n"
        "for ax, c in zip(axes.ravel(), D.NUMERIC_COLS):\n"
        "    for cls in D.CLASSES:\n"
        "        s = tr.loc[tr[D.TARGET]==cls, c]\n"
        "        clip = s.clip(s.quantile(.01), s.quantile(.99))\n"
        "        ax.hist(clip, bins=60, alpha=.5, density=True, label=cls)\n"
        "    ax.set_title(c); ax.legend(fontsize=7)\n"
        "plt.tight_layout(); plt.savefig(FIG/'numeric_by_class.png', dpi=110); plt.show()"
    ),
    code(
        "# redshift on log1p scale, by class\n"
        "fig, ax = plt.subplots(figsize=(9,5))\n"
        "for cls in D.CLASSES:\n"
        "    ax.hist(np.log1p(tr.loc[tr[D.TARGET]==cls,'redshift'].clip(lower=-0.999)),\n"
        "            bins=80, alpha=.5, density=True, label=cls)\n"
        "ax.set_title('log1p(redshift) by class'); ax.legend()\n"
        "plt.tight_layout(); plt.savefig(FIG/'redshift_by_class.png', dpi=110); plt.show()\n"
        "print(tr.groupby(D.TARGET)['redshift'].describe().round(3))"
    ),
    md("## 4. Categorical features vs class"),
    code(
        "for c in D.CATEGORICAL_COLS:\n"
        "    ct = pd.crosstab(tr[c], tr[D.TARGET], normalize='index')[D.CLASSES]\n"
        "    print('\\n==', c, '==\\n', ct.round(3))\n"
        "    ct.plot.bar(stacked=True, figsize=(7,4), title=f'{c} -> class')\n"
        "    plt.tight_layout(); plt.savefig(FIG/f'cat_{c}.png', dpi=110); plt.show()"
    ),
    md("## 5. Engineered color-index separation"),
    code(
        "Xtr = F.add_features(tr)\n"
        "fig, axes = plt.subplots(1, 4, figsize=(20,4))\n"
        "for ax, c in zip(axes, F.COLOR_COLS):\n"
        "    for cls in D.CLASSES:\n"
        "        s = Xtr.loc[tr[D.TARGET]==cls, c]\n"
        "        ax.hist(s.clip(s.quantile(.01), s.quantile(.99)), bins=60, alpha=.5, density=True, label=cls)\n"
        "    ax.set_title(c); ax.legend(fontsize=7)\n"
        "plt.tight_layout(); plt.savefig(FIG/'colors_by_class.png', dpi=110); plt.show()"
    ),
    md("## 6. Correlation of numeric features"),
    code(
        "corr = Xtr[F.NUMERIC_FEATURES].corr()\n"
        "plt.figure(figsize=(11,9)); sns.heatmap(corr, cmap='coolwarm', center=0, square=True)\n"
        "plt.title('Numeric feature correlation'); plt.tight_layout()\n"
        "plt.savefig(FIG/'correlation.png', dpi=110); plt.show()"
    ),
    md("## 7. Train vs test distribution sanity (no drift expected)"),
    code(
        "fig, axes = plt.subplots(2,4, figsize=(18,8))\n"
        "for ax, c in zip(axes.ravel(), D.NUMERIC_COLS):\n"
        "    ax.hist(tr[c].clip(tr[c].quantile(.01),tr[c].quantile(.99)), bins=60, alpha=.5, density=True, label='train')\n"
        "    ax.hist(te[c].clip(te[c].quantile(.01),te[c].quantile(.99)), bins=60, alpha=.5, density=True, label='test')\n"
        "    ax.set_title(c); ax.legend(fontsize=7)\n"
        "plt.tight_layout(); plt.savefig(FIG/'train_vs_test.png', dpi=110); plt.show()"
    ),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nbf.write(nb, NB)
print(f"wrote {NB}")
