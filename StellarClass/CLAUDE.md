# CLAUDE.md

Guidance for Claude Code when working in this project.

## Project — StellarClass (Kaggle S6E6)
[`playground-series-s6e6`](https://www.kaggle.com/competitions/playground-series-s6e6): 3-class
classification of celestial objects (**GALAXY / STAR / QSO**) from SDSS photometric/spectroscopic
features. **Metric: Balanced Accuracy** (hard labels — each class weighted equally).

**Most important consequence of the metric:** optimize macro-recall, not plain accuracy. The target
is imbalanced, so beyond `class_weight="balanced"` during training, the biggest lever is a
**per-class prior/probability correction on OOF** (grid-search per-class multipliers, then argmax) to
maximize OOF balanced accuracy. Tune this on OOF, not on the leaderboard.

## Environment & commands
Uses **uv** (not pip/`requirements.txt`). Run everything from this dir, prefix Python with `uv run`:

```bash
uv sync
uv run pytest
uv run python -m src.train --model lgbm        # also xgb, cat
uv run python -m src.blend
uv run python -m src.submit --model blend
```

Kaggle auth: token at `~/.kaggle/access_token`, passed via env var:
```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
uv run kaggle competitions download -c playground-series-s6e6 -p data/raw
```

## Architecture (prediction-file pipeline)
Models communicate through saved arrays in `experiments/artifacts/`: `{model}_oof.npy` (n×3 class
probs) + fold-averaged `{model}_test.npy`. Training and submission are decoupled; any subset blends.
- `src/data.py` — schema constants + loaders (raise on mismatch); label encoder.
- `src/features.py` — stateless, leakage-safe features (colors `u-g/g-r/r-i/i-z`, `redshift` log1p,
  sentinel→NaN + flags); applied identically to train/test.
- `src/cv.py` — `StratifiedKFold(5, shuffle, seed=42)` on encoded `class`, shared by all models.
- `src/train.py` — GBDT CV (LightGBM/XGBoost/CatBoost), multiclass, balanced weighting; saves OOF +
  bagged test probs; reports balanced accuracy (raw-argmax and prior-corrected).
- `src/dl/train_nn.py` — GPU MLP member on remote RTX 3090 Ti (Mac has no CUDA), same folds, same
  `.npy` format. See [[flowerclf-remote-gpu-box]].
- `src/blend.py` — equal-weight probability blend + per-class decision correction for balanced acc.
- `src/submit.py` — build/validate submission vs `sample_submission.csv`; optional upload.

Git-ignored: `data/`, `models/`, `outputs/`, `experiments/artifacts/`, `reports/figures/`, `.venv/`,
kaggle token.

## Conventions & rules
- **Show CV/OOF balanced accuracy and get explicit approval before any Kaggle submission.** Log every
  run in `experiments/README.md`. Submit only on above-noise gains.
- Commit incrementally after each meaningful step; stage only this project's files (monorepo has
  unrelated siblings).
- The living plan is at `~/.claude/plans/i-want-to-solve-snuggly-clock.md`.
