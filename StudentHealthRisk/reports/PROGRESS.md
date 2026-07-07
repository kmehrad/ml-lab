# Progress report — StudentHealthRisk (Kaggle PS S6E7)

_Handoff for resuming later. Branch: `experiment/shr-improve-0.95`. Competition open until 2026-07-31._

## TL;DR
- **Best submission: diverse ensemble `xgb + TabM + FTT` (⅓ each) — public LB 0.94981**, OOF 0.94994.
- Metric: **Balanced Accuracy Score** (mean per-class recall). Target `health_condition`, 3 classes,
  86/8/6 imbalanced. 690k train / 296k test. Missing values throughout.
- The label is near-deterministic in a few raw features (0 label conflicts) → GBDTs saturate ~0.9495.
  **Cross-family model diversity (GBDT + strong tabular NNs) is the only thing that improved the LB.**
- Top LB cluster ≈ **0.951** (~+0.0013 away). Closing it = more/stronger diverse learners + stacking.

## LB history (what transferred, what didn't)
| Submission | OOF (tuned) | Public LB | Verdict |
|---|---|---|---|
| 3-GBDT blend (lgbm+xgb+cat) | 0.94979 | 0.94953 | first real submission |
| xgb+lgbm @800 trees (metric-aligned tree count) | 0.94998 | 0.94938 | **regressed** — OOF gain was noise |
| **xgb + TabM + FTT (diverse hillclimb)** | 0.94994 | **0.94981** | **new best; +0.00028, transferred** |

**Key lesson:** at this plateau the OOF↔LB coupling is ~±0.0002, so *within-GBDT-family* OOF moves don't
transfer (tree count, target encoding, augmentation, pseudo-labeling all failed/regressed). *Genuine
cross-family diversity* (independent NN families) does.

## What was tried
**Improvement levers — all negative** (exp-009..021 in `experiments/README.md`): original-data
augmentation, more-trees/lower-LR, decision-rule refinement, combination target encoding, diverse
GBDT ensembling, pseudo-labeling.

**NN-diversity phase — the breakthrough** (exp-022..029):
- **RealMLP 0.94785** — too weak; ensembles ignore it. Rejected.
- **TabM 0.94905**, **FTT 0.94911** — near-GBDT strength, different families, decorrelated (each fixes
  ~4–5% of xgb's errors) → hillclimb selects them. **This is the win.**
- Sub-lessons: metric-aligned early stopping (`1-balanced_accuracy`) and minority oversampling both
  *hurt* the tuned score (they trade the probability calibration our post-hoc decision tuning relies on) —
  use plain `cross_entropy` + decision tuning. RealMLP's tiny default batch = ~3 h/run; always set
  `--batch-size 2048` (→ ~12 min).

## How to resume
```bash
cd StudentHealthRisk && uv sync                      # local (CPU: metric/blend/stack/submit)
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
# GPU box (RealMLP/TabM/FTT need CUDA): scripts/remote_run.sh uses ~/.ssh/dragon
scripts/remote_run.sh push
scripts/remote_run.sh run "~/.local/bin/uv sync --extra gpu"     # installs torch/pytabkit/skorch/probmetrics/numba
# train a diverse NN (arch: realmlp|tabm|ftt|tabr|mlp_plr)
scripts/remote_run.sh run "python -m src.models_realmlp --arch tabm --folds 5 --n-cv 1 --batch-size 2048 --tag tabm"
scripts/remote_run.sh pull
# ensemble + submit (gated on approval)
uv run python -m src.hillclimb --models xgb lgbm cat tabm ftt --steps 40 --tag ens_full
uv run python -m src.submit --model ens_full            # add --submit to upload
```
Artifacts (`experiments/artifacts/*_oof.npy` / `*_test.npy`) are git-ignored — regenerate on the box,
or they persist there (`~/StudentHealthRisk/experiments/artifacts/`). `y.npy`/`classes.npy`/`test_id.npy`
are shared. Best ensemble artifact: `ens_full` (also `outputs/ens_full_submission.csv`).

## Next ideas to push toward 0.951 (ranked, diminishing returns per model)
1. **Fix TabR / MLP_PLR** — both fail with a pytabkit "0 features" error (likely an input-format quirk for
   those architectures); each is another decorrelated family for the ensemble.
2. **Seed-average TabM & FTT** (`--seeds 3`) — cheap variance reduction on the two learners that matter.
3. **LogReg stacker over the full zoo** (`src/stack.py --models xgb lgbm cat tabm ftt ...`) — a learned
   meta-model may beat equal-weight hillclimb once there are ≥4 strong diverse bases.
4. **Stronger TabM/FTT** — bump `--n-cv`/internal ensembling, or tune epochs/width toward the public
   notebook's config (they reached ~0.95 solo).
5. Only submit OOF gains that clear the noise floor, and **always LB-confirm** (the tree-count regression
   is the cautionary tale).

## Governance / conventions
Gate every change on OOF balanced accuracy (tuned decision); treat sub-±0.0015 OOF moves as ties;
**show CV/OOF and get approval before any Kaggle upload**; commit per step (this project's files only);
log every run in `experiments/README.md`. Plan file: `~/.claude/plans/i-want-to-build-cozy-seahorse.md`.
