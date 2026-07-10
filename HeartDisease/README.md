# HeartDisease — Kaggle Playground Series S6E2

[Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2) —
binary classification (`Heart Disease`: `Presence`/`Absence`) on the classic UCI Statlog
Heart features. Metric: **ROC AUC**.

See `CLAUDE.md` for schema, architecture, and conventions. See `experiments/README.md` for
the run log and `reports/EDA_FINDINGS.md` for EDA conclusions.

```bash
uv sync
uv run pytest
uv run python -m src.train --model lgbm
uv run python -m src.blend
uv run python -m src.submit --model blend
```
