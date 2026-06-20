"""Central configuration: paths, column names, and constants.

Importing this module also resolves all paths relative to the project root,
so scripts work regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
OUTPUTS = ROOT / "outputs"

TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"

for _d in (DATA_PROCESSED, MODELS, SUBMISSIONS, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Problem definition ----------------------------------------------------
ID_COL = "Id"
TARGET = "SalePrice"

# Competition metric is RMSE on log(SalePrice); we train on log1p(target)
# and invert with expm1 at prediction time.
RANDOM_STATE = 42
N_FOLDS = 5

# Categorical columns where a missing value semantically means "None"
# (feature is absent), per the Ames data dictionary — not missing-at-random.
NA_MEANS_NONE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
    "MasVnrType",
]

# Numeric columns where a missing value means zero (e.g. no garage -> 0 cars).
NA_MEANS_ZERO = [
    "GarageYrBlt", "GarageArea", "GarageCars", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
]
