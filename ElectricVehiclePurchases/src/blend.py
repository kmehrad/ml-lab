"""Equal-weight OOF/test probability blend, scored without leaderboard fitting."""
import argparse
import json

import numpy as np

from . import data
from .metric import auc

ART = data.ROOT / "experiments" / "artifacts"


def blend(models: list[str]) -> dict:
    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("choose at least two distinct model artifacts")
    y = np.load(ART / "y.npy")
    test_id = np.load(ART / "test_id.npy")
    oofs = [np.load(ART / f"{name}_oof.npy") for name in models]
    tests = [np.load(ART / f"{name}_test.npy") for name in models]
    if any(p.shape != y.shape for p in oofs) or any(p.shape != test_id.shape for p in tests):
        raise ValueError("model prediction shapes do not align")
    oof, test = np.mean(oofs, axis=0), np.mean(tests, axis=0)
    score = auc(y, oof)
    np.save(ART / "blend_oof.npy", oof)
    np.save(ART / "blend_test.npy", test)
    result = {"models": models, "weights": [1 / len(models)] * len(models), "oof_auc": score}
    (ART / "blend_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    blend(parser.parse_args().models)
