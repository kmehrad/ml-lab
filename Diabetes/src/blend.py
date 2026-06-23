"""Diversity blend over saved out-of-fold + bagged-test predictions.

Each ``src.train`` run saves ``{key}_oof.npy`` and ``{key}_test.npy`` (``key`` is
the model name, plus an ``_aug`` suffix for original-augmented runs). The winning
leaderboard recipe (see ``experiments/README.md``) is an **equal-weight
rank-average** of the regularised GBMs across the base and augmented pools —
local OOF is *not* predictive of the leaderboard, and an equal-weight diverse
blend generalised better than any OOF-optimal weighting.

This module rank-averages the selected members' OOF (for a sanity AUC) and their
bagged test predictions, then saves the blended test prediction (``blend_test.npy``)
and the member list (``blend_weights.json``) for :mod:`src.submit`.

Usage
-----
    python -m src.blend                                   # default GBM pool
    python -m src.blend --members lgbm lgbm2 xgb histgb lgbm_aug lgbm2_aug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
WEIGHTS_JSON = ARTIFACTS_DIR / "blend_weights.json"
BLEND_TEST = ARTIFACTS_DIR / "blend_test.npy"

# The winning pool: regularised GBMs, base and original-augmented.
DEFAULT_MEMBERS = (
    "lgbm", "lgbm2", "xgb", "histgb",
    "lgbm_aug", "lgbm2_aug", "xgb_aug", "histgb_aug",
)


def _available(members: list[str]) -> list[str]:
    """Keep members that have both OOF and test arrays saved."""
    keep = []
    for m in members:
        if (ARTIFACTS_DIR / f"{m}_oof.npy").exists() and (ARTIFACTS_DIR / f"{m}_test.npy").exists():
            keep.append(m)
        else:
            print(f"  skipping {m!r}: predictions not found")
    return keep


def find_blend(members: list[str]) -> dict:
    """Equal-weight rank-average blend; report OOF AUC, save test blend + members."""
    members = _available(members)
    if not members:
        raise RuntimeError("No usable predictions found; run src.train first.")
    y = np.load(ARTIFACTS_DIR / "y_true.npy")
    n_test = len(np.load(ARTIFACTS_DIR / f"{members[0]}_test.npy"))

    singles, oof_ranks, test_ranks = {}, [], []
    for m in members:
        oof = np.load(ARTIFACTS_DIR / f"{m}_oof.npy")
        if np.isnan(oof).any():
            print(f"  skipping {m!r}: OOF has NaN (incomplete CV)")
            continue
        singles[m] = round(float(roc_auc_score(y, oof)), 5)
        oof_ranks.append(rankdata(oof) / len(oof))
        test_ranks.append(rankdata(np.load(ARTIFACTS_DIR / f"{m}_test.npy")) / n_test)

    used = list(singles)
    oof_blend = np.mean(oof_ranks, axis=0)
    test_blend = np.mean(test_ranks, axis=0)
    blended_auc = round(float(roc_auc_score(y, oof_blend)), 5)

    result = {
        "members": used,
        "weights": "equal rank-average",
        "single_auc": singles,
        "blended_auc": blended_auc,
        "best_single_auc": max(singles.values()),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_JSON.write_text(json.dumps(result, indent=2))
    np.save(BLEND_TEST, test_blend)

    print(f"Blend of {len(used)} members (equal rank-average):")
    for m in used:
        print(f"  {m:>12}: {singles[m]:.5f}")
    print(f"Blended OOF AUC: {blended_auc:.5f}  (best single {result['best_single_auc']:.5f})")
    print("NOTE: OOF is not predictive of the leaderboard; the equal-weight diverse "
          "blend generalised best. See experiments/README.md.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", default=list(DEFAULT_MEMBERS),
                        help="prediction keys to blend (need saved *_oof and *_test)")
    args = parser.parse_args()
    find_blend(args.members)


if __name__ == "__main__":
    main()
