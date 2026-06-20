"""Exploratory data analysis for the Ames House Prices dataset.

Run as a module from the project root:

    python -m src.eda

Prints a structured report to the console and saves figures to ``outputs/``.
Analysis is done on the *raw* data (before cleaning) so the missing-value
structure stays visible. No modeling here — this is the understand-the-data step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive: just write PNGs
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    _HAS_SNS = True
except ImportError:  # seaborn is optional; fall back to plain matplotlib
    _HAS_SNS = False

from . import config
from .data import load_raw

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _feature_dtypes(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric vs categorical (pandas-3 safe)."""
    feats = train.drop(columns=[config.ID_COL, config.TARGET])
    numeric = feats.select_dtypes(include="number").columns.tolist()
    categorical = feats.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


# --- Report sections -------------------------------------------------------

def section_overview(train: pd.DataFrame, test: pd.DataFrame) -> None:
    _header("1. OVERVIEW")
    numeric, categorical = _feature_dtypes(train)
    print(f"train : {train.shape[0]:>5} rows x {train.shape[1]} cols "
          f"(incl. {config.ID_COL} + {config.TARGET})")
    print(f"test  : {test.shape[0]:>5} rows x {test.shape[1]} cols "
          f"(incl. {config.ID_COL}, no target)")
    print(f"\nfeatures: {len(numeric) + len(categorical)} total")
    print(f"  numeric     : {len(numeric)}")
    print(f"  categorical : {len(categorical)}")
    print(f"\n{config.ID_COL} range  train: {train[config.ID_COL].min()}–"
          f"{train[config.ID_COL].max()}   test: "
          f"{test[config.ID_COL].min()}–{test[config.ID_COL].max()}")


def section_target(train: pd.DataFrame) -> None:
    _header(f"2. TARGET — {config.TARGET}")
    s = train[config.TARGET]
    print(s.describe().round(1).to_string())
    print(f"\nskew     : {s.skew():.3f}")
    print(f"kurtosis : {s.kurtosis():.3f}")
    print(f"log1p skew: {np.log1p(s).skew():.3f}   "
          "(near 0 → log-target modeling is appropriate)")


def section_missing(train: pd.DataFrame, test: pd.DataFrame) -> None:
    _header("3. MISSING VALUES")
    combined = pd.concat(
        [train.drop(columns=[config.TARGET]), test], ignore_index=True
    )
    none_set = set(config.NA_MEANS_NONE)
    zero_set = set(config.NA_MEANS_ZERO)

    def tag(col: str) -> str:
        if col in none_set:
            return "semantic (None)"
        if col in zero_set:
            return "semantic (0)"
        return "impute"

    miss = combined.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    table = pd.DataFrame({
        "train": train.isna().sum().reindex(miss.index).fillna(0).astype(int),
        "test": test.isna().sum().reindex(miss.index).fillna(0).astype(int),
        "combined_%": (miss / len(combined) * 100).round(1),
        "kind": [tag(c) for c in miss.index],
    })
    print(table.to_string())
    print(f"\ncolumns with any missing: {len(miss)}")
    counts = table["kind"].value_counts()
    for kind, n in counts.items():
        print(f"  {kind:16s}: {n}")
    print("\n→ 'semantic' columns are handled by config NA_MEANS_NONE/ZERO; "
          "only 'impute' rows are genuinely missing-at-random.")


def section_numeric(train: pd.DataFrame) -> None:
    _header("4. NUMERIC FEATURES")
    numeric, _ = _feature_dtypes(train)
    num = train[numeric + [config.TARGET]]
    corr = (num.corr(numeric_only=True)[config.TARGET]
            .drop(config.TARGET)
            .sort_values(key=abs, ascending=False))
    print("Correlation with target (top 15):")
    print(corr.head(15).round(3).to_string())
    print("\nWeakest 5:")
    print(corr.tail(5).round(3).to_string())

    skew = (train[numeric].apply(lambda x: x.skew())
            .sort_values(key=abs, ascending=False))
    skewed = skew[skew.abs() > 0.75]
    print(f"\nHighly skewed numerics (|skew| > 0.75): {len(skewed)} "
          "→ log-transform candidates for the linear model")
    print(skewed.round(2).to_string())


def section_categorical(train: pd.DataFrame) -> None:
    _header("5. CATEGORICAL FEATURES")
    _, categorical = _feature_dtypes(train)
    card = train[categorical].nunique().sort_values(ascending=False)
    print("Cardinality (n unique levels), top 15:")
    print(card.head(15).to_string())

    print("\nMedian SalePrice by level (high-signal columns):")
    for col in ["OverallQual", "Neighborhood", "ExterQual", "KitchenQual"]:
        if col not in train.columns:
            continue
        g = (train.groupby(col)[config.TARGET].median()
             .sort_values(ascending=False))
        print(f"\n  {col}:")
        print(g.round(0).head(6).to_string().replace("\n", "\n    "))


def section_outliers(train: pd.DataFrame) -> None:
    _header("6. OUTLIERS — GrLivArea anomaly (De Cock, 2011)")
    big = (train.loc[train["GrLivArea"] > 4000,
                     [config.ID_COL, "GrLivArea", "OverallQual", config.TARGET]]
           .sort_values("GrLivArea"))
    print(big.to_string(index=False))
    flagged = big.loc[big[config.TARGET] < 300000, config.ID_COL].tolist()
    print(f"\n→ Ids {flagged} are huge, top-quality homes that sold cheap — "
          "the documented outliers to drop from TRAIN only.")


# --- Figures ---------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    path = config.OUTPUTS / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def make_figures(train: pd.DataFrame, test: pd.DataFrame) -> None:
    _header("FIGURES → outputs/")

    # Target distribution: raw vs log1p
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(train[config.TARGET], bins=50, color="#4C72B0")
    axes[0].set_title("SalePrice (raw)"); axes[0].set_xlabel("SalePrice")
    axes[1].hist(np.log1p(train[config.TARGET]), bins=50, color="#55A868")
    axes[1].set_title("log1p(SalePrice)"); axes[1].set_xlabel("log1p(SalePrice)")
    fig.suptitle("Target distribution")
    _save(fig, "eda_target_distribution.png")

    # Missingness bar (combined %)
    combined = pd.concat(
        [train.drop(columns=[config.TARGET]), test], ignore_index=True)
    miss = (combined.isna().mean() * 100)
    miss = miss[miss > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(miss))))
    ax.barh(miss.index[::-1], miss.values[::-1], color="#C44E52")
    ax.set_xlabel("% missing (train + test)")
    ax.set_title("Columns with missing values")
    _save(fig, "eda_missingness.png")

    # Correlation heatmap: target + top numeric features
    numeric, _ = _feature_dtypes(train)
    num = train[numeric + [config.TARGET]]
    corr_t = (num.corr(numeric_only=True)[config.TARGET]
              .drop(config.TARGET).sort_values(key=abs, ascending=False))
    top = corr_t.head(12).index.tolist()
    cmat = train[[config.TARGET] + top].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    if _HAS_SNS:
        sns.heatmap(cmat, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, cbar_kws={"shrink": 0.8})
    else:
        im = ax.imshow(cmat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cmat))); ax.set_xticklabels(cmat.columns, rotation=90)
        ax.set_yticks(range(len(cmat))); ax.set_yticklabels(cmat.columns)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation: SalePrice + top numeric features")
    _save(fig, "eda_corr_top.png")

    # GrLivArea scatter with outliers highlighted
    fig, ax = plt.subplots(figsize=(7, 5))
    out = (train["GrLivArea"] > 4000) & (train[config.TARGET] < 300000)
    ax.scatter(train.loc[~out, "GrLivArea"], train.loc[~out, config.TARGET],
               s=12, alpha=0.5, label="normal")
    ax.scatter(train.loc[out, "GrLivArea"], train.loc[out, config.TARGET],
               s=60, color="red", marker="x", label="outlier (drop)")
    ax.set_xlabel("GrLivArea"); ax.set_ylabel("SalePrice")
    ax.set_title("GrLivArea vs SalePrice"); ax.legend()
    _save(fig, "eda_grlivarea_scatter.png")

    # Boxplot: SalePrice by OverallQual
    fig, ax = plt.subplots(figsize=(8, 5))
    order = sorted(train["OverallQual"].unique())
    data = [train.loc[train["OverallQual"] == q, config.TARGET] for q in order]
    ax.boxplot(data, tick_labels=order)
    ax.set_xlabel("OverallQual"); ax.set_ylabel("SalePrice")
    ax.set_title("SalePrice by OverallQual")
    _save(fig, "eda_saleprice_by_overallqual.png")

    # Boxplot: SalePrice by Neighborhood (sorted by median)
    med = train.groupby("Neighborhood")[config.TARGET].median().sort_values()
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [train.loc[train["Neighborhood"] == n, config.TARGET] for n in med.index]
    ax.boxplot(data, tick_labels=med.index)
    ax.set_ylabel("SalePrice"); ax.set_title("SalePrice by Neighborhood (sorted)")
    plt.setp(ax.get_xticklabels(), rotation=90)
    _save(fig, "eda_saleprice_by_neighborhood.png")


def main() -> None:
    train, test = load_raw()
    section_overview(train, test)
    section_target(train)
    section_missing(train, test)
    section_numeric(train)
    section_categorical(train)
    section_outliers(train)
    make_figures(train, test)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
