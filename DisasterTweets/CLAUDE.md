# CLAUDE.md — DisasterTweets

Kaggle `nlp-getting-started`: binary text classification, metric **F1**.

## Commands

```bash
uv sync                                          # create env from pyproject.toml
uv run python -m pytest                          # schema + feature tests
uv run python -m src.train --model tfidf         # classic TF-IDF baseline
uv run python -m src.transformer --model roberta # transformer fine-tune (cuda→mps→cpu)
uv run python -m src.blend                       # rank-average ensemble
uv run python -m src.submit --model blend        # build submission (add --submit to upload)
```

Data download: `export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"` then
`uv run kaggle competitions download -c nlp-getting-started -p data/raw && unzip -o data/raw/*.zip -d data/raw`.

## Architecture

- **Prediction-file based.** Each model writes `experiments/artifacts/{model}_oof.npy`,
  `{model}_test.npy`, `{model}_metrics.json`. Models are not pickled to git.
- **Shared CV.** `StratifiedKFold(5, shuffle=True, random_state=42)` on `target`, identical
  folds across classic + transformer models so OOF/test arrays line up for blending.
- **F1 metric.** Score F1 on hard labels; tune the probability threshold on OOF, store it in
  the metrics JSON, and apply it when building submissions.
- `src/data.py` pins the schema and raises loudly on column mismatch.

## Workflow rules (repo memories)

- Work on a feature branch; commit after each meaningful step.
- Show CV/OOF F1 and **wait for approval before any Kaggle submission**.
- Only adopt/submit a change if OOF F1 beats the current best by more than the fold std.
