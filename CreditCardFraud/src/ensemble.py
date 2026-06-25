"""Rank-average blend over selected models' saved predictions.

The supervised models share saved OOF (``{key}_oof.npy``) and bagged test
(``{key}_test.npy``) arrays. This module **rank-averages** a chosen set of members
— converting each member's scores to ranks before averaging, which is robust to
the very different score calibrations of GBMs vs. logistic regression — and scores
the blend with AUPRC on both OOF and the held-out test set.

Equal-weight rank averaging is the default because OOF-optimised weighting tends to
overfit with so few positives. The blended test array is saved as ``blend_test.npy``
and the chosen members to ``blend_weights.json`` for ``src.evaluate``.

Usage
-----
    python -m src.ensemble                       # auto-select GBM members
    python -m src.ensemble --members lgbm xgb catboost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
BLEND_TEST = ARTIFACTS_DIR / "blend_test.npy"
BLEND_OOF = ARTIFACTS_DIR / "blend_oof.npy"
BLEND_WEIGHTS = ARTIFACTS_DIR / "blend_weights.json"

# Sensible default pool: the supervised GBMs (no reweighting) plus logreg.
DEFAULT_MEMBERS = ("lgbm", "xgb", "catboost", "histgb", "logreg")


def _rank_average(arrays: list[np.ndarray]) -> np.ndarray:
    """Equal-weight average of per-array ranks, normalised to ``[0, 1]``."""
    stacked = np.vstack([rankdata(a) / len(a) for a in arrays])
    return stacked.mean(axis=0)


def build_blend(members: list[str]) -> dict:
    """Rank-average the given members' OOF + test arrays and score the blend."""
    available = [m for m in members if (ARTIFACTS_DIR / f"{m}_oof.npy").exists()]
    if not available:
        raise FileNotFoundError(
            f"No OOF arrays found for {members}. Run `python -m src.train` first."
        )
    y_oof = np.load(ARTIFACTS_DIR / "y_oof.npy")
    y_test = np.load(ARTIFACTS_DIR / "y_test.npy")

    oof_blend = _rank_average([np.load(ARTIFACTS_DIR / f"{m}_oof.npy") for m in available])
    test_blend = _rank_average([np.load(ARTIFACTS_DIR / f"{m}_test.npy") for m in available])

    np.save(BLEND_OOF, oof_blend)
    np.save(BLEND_TEST, test_blend)
    result = {
        "members": available,
        "oof_ap": round(float(average_precision_score(y_oof, oof_blend)), 5),
        "oof_roc_auc": round(float(roc_auc_score(y_oof, oof_blend)), 5),
        "test_ap": round(float(average_precision_score(y_test, test_blend)), 5),
        "test_roc_auc": round(float(roc_auc_score(y_test, test_blend)), 5),
    }
    BLEND_WEIGHTS.write_text(json.dumps(result, indent=2))
    print(
        f"blend({'+'.join(available)}): oof_ap={result['oof_ap']:.5f}  "
        f"test_ap={result['test_ap']:.5f}  test_auc={result['test_roc_auc']:.5f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", default=list(DEFAULT_MEMBERS))
    args = parser.parse_args()
    build_blend(args.members)


if __name__ == "__main__":
    main()
