from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def save_shared(path, y, folds, test_id):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    np.save(path / "y.npy", np.asarray(y))
    np.save(path / "folds.npy", np.asarray(folds))
    np.save(path / "test_id.npy", np.asarray(test_id))


def save_run(path, tag, oof, test, metadata):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    oof, test = np.asarray(oof, float), np.asarray(test, float)
    if oof.ndim != 1 or test.ndim != 1 or not np.isfinite(oof).all() or not np.isfinite(test).all():
        raise ValueError("artifacts must be finite 1-D arrays")
    np.save(path / f"{tag}_oof.npy", oof)
    np.save(path / f"{tag}_test.npy", test)
    (path / f"{tag}_metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
