"""Blend base-model probabilities and tune the balanced-accuracy decision on OOF.

Usage:
    uv run python -m src.blend                       # blend all available base models
    uv run python -m src.blend --models lgbm xgb cat # explicit subset

Reads {model}_oof.npy / {model}_test.npy from experiments/artifacts/, equal-weight averages the
class probabilities, then grid-searches per-class multipliers to maximize OOF balanced accuracy.
Writes blend_oof.npy, blend_test.npy, blend_metrics.json (incl. the chosen multipliers).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M

ART = Path(__file__).resolve().parent.parent / "experiments" / "artifacts"
BASE_MODELS = ["lgbm", "xgb", "cat"]


def available_models(requested: list[str] | None) -> list[str]:
    models = requested or BASE_MODELS
    found = [m for m in models if (ART / f"{m}_oof.npy").exists()]
    missing = set(models) - set(found)
    if missing:
        print(f"warning: skipping models with no artifacts: {sorted(missing)}")
    if not found:
        raise SystemExit("no model artifacts found — run src.train first")
    return found


def blend(models: list[str]) -> dict:
    y = np.load(ART / "y_true.npy")
    oof = np.mean([np.load(ART / f"{m}_oof.npy") for m in models], axis=0)
    test = np.mean([np.load(ART / f"{m}_test.npy") for m in models], axis=0)

    raw = M.balanced_accuracy(y, oof)
    mult, corrected = M.tune_multipliers(y, oof)
    print(f"[blend {models}] OOF bal-acc raw={raw:.5f}  corrected={corrected:.5f}  "
          f"mult(GALAXY,QSO,STAR)={np.round(mult,3).tolist()}")

    np.save(ART / "blend_oof.npy", oof)
    np.save(ART / "blend_test.npy", test)
    result = {
        "models": models, "oof_bacc_raw": raw, "oof_bacc_corrected": corrected,
        "mult": mult.tolist(),
    }
    json.dump(result, open(ART / "blend_metrics.json", "w"), indent=2)

    # per-model comparison for the log
    print("per-model OOF bal-acc (corrected):")
    for m in models:
        o = np.load(ART / f"{m}_oof.npy")
        _, c = M.tune_multipliers(y, o)
        print(f"  {m}: {c:.5f}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    a = p.parse_args()
    blend(available_models(a.models))
