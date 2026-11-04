"""Data profile written as Markdown; notebook wraps these checks interactively."""
import pandas as pd

from . import data
from .features import feature_columns, numeric_and_categorical


def main() -> None:
    train, test, _ = data.load_all()
    y = data.encode_target(train[data.TARGET])
    cols = feature_columns(train)
    nums, cats = numeric_and_categorical(train[cols])
    miss_train = train[cols].isna().mean().sort_values(ascending=False)
    miss_test = test[cols].isna().mean().sort_values(ascending=False)
    lines = ["# EDA findings", "", f"Train rows: {len(train):,}; test rows: {len(test):,}.",
             f"Features: {len(cols)} ({len(nums)} numeric, {len(cats)} categorical).",
             f"Positive class share: {y.mean():.4%}.",
             f"Duplicate feature rows in train: {train[cols].duplicated().sum():,}.", "",
             "## Missingness", "", "| Feature | Train | Test |", "|---|---:|---:|"]
    for col in cols:
        lines.append(f"| {col} | {miss_train[col]:.2%} | {miss_test[col]:.2%} |")
    lines += ["", "## Categorical cardinality and target rate", ""]
    for col in cats:
        rates = pd.DataFrame({col: train[col].fillna("__NA__"), "target": y}).groupby(col)["target"].agg(["count", "mean"])
        lines.append(f"### {col} ({len(rates)} levels)")
        lines.append("")
        lines.append(rates.sort_values("count", ascending=False).head(15).to_markdown())
        lines.append("")
    lines += ["## Train/test numeric shifts", "", "| Feature | Train mean | Test mean |", "|---|---:|---:|"]
    for col in nums:
        lines.append(f"| {col} | {train[col].mean():.4g} | {test[col].mean():.4g} |")
    lines += ["", "Review these shifts and class rates before creating features; target-derived encoders must fit within folds."]
    path = data.ROOT / "reports" / "EDA_FINDINGS.md"
    path.write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
