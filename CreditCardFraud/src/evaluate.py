"""Final held-out evaluation and operating-threshold selection.

This replaces a Kaggle ``submit.py``: there is no leaderboard, so the deliverable
is a report on the **fixed held-out test set** (``src.data.make_holdout_split``).
Given a model key (or ``blend``), it loads that model's bagged test predictions
and reports the threshold-free summaries (**AUPRC** primary, ROC-AUC secondary)
plus the confusion matrix / precision / recall / F1 at a chosen operating point.

Two threshold policies are offered, both fit on **OOF** predictions (never the test
set, to avoid optimistic threshold selection) and then applied to the test set:

* ``max-f1``          — the threshold maximising F1 on the OOF precision-recall curve.
* ``precision-floor`` — the lowest threshold whose OOF precision is >= ``--min-precision``
  (maximises recall subject to a precision budget — the realistic fraud-ops framing).

Outputs a metrics row to ``outputs/{model}_eval.json`` and a precision-recall curve
figure to ``reports/figures/{model}_pr_curve.png``.

Usage
-----
    python -m src.evaluate --model lgbm
    python -m src.evaluate --model blend --policy precision-floor --min-precision 0.9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "artifacts"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def _load_preds(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(oof, test)`` prediction arrays for a model key or the blend."""
    if model == "blend":
        oof = ARTIFACTS_DIR / "blend_oof.npy"
        test = ARTIFACTS_DIR / "blend_test.npy"
    else:
        oof = ARTIFACTS_DIR / f"{model}_oof.npy"
        test = ARTIFACTS_DIR / f"{model}_test.npy"
    if not test.exists():
        raise FileNotFoundError(f"{test.name} not found. Train/blend before evaluating.")
    oof_arr = np.load(oof) if oof.exists() else None
    return oof_arr, np.load(test)


def select_threshold(
    y_oof: np.ndarray, oof_pred: np.ndarray, policy: str, min_precision: float
) -> float:
    """Pick an operating threshold from the OOF precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_oof, oof_pred)
    # precision/recall have one more element than thresholds; align to thresholds.
    precision, recall = precision[:-1], recall[:-1]
    if policy == "max-f1":
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        return float(thresholds[int(np.argmax(f1))])
    if policy == "precision-floor":
        ok = precision >= min_precision
        if not ok.any():
            raise ValueError(
                f"No OOF threshold reaches precision >= {min_precision}; "
                f"max achievable is {precision.max():.3f}."
            )
        # Among thresholds meeting the floor, take the one with highest recall.
        idx = np.where(ok)[0]
        return float(thresholds[idx[np.argmax(recall[idx])]])
    raise ValueError(f"Unknown policy {policy!r}")


def evaluate(model: str, policy: str = "max-f1", min_precision: float = 0.9) -> dict:
    """Score ``model`` on the held-out test set and report an operating point."""
    oof_pred, test_pred = _load_preds(model)
    y_test = np.load(ARTIFACTS_DIR / "y_test.npy")
    y_oof = np.load(ARTIFACTS_DIR / "y_oof.npy")
    if oof_pred is None:
        # Unsupervised detectors have no OOF; fall back to test-derived threshold.
        oof_pred, y_oof = test_pred, y_test

    threshold = select_threshold(y_oof, oof_pred, policy, min_precision)
    pred_label = (test_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_label).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result = {
        "model": model,
        "policy": policy,
        "threshold": round(threshold, 6),
        "test_ap": round(float(average_precision_score(y_test, test_pred)), 5),
        "test_roc_auc": round(float(roc_auc_score(y_test, test_pred)), 5),
        "precision": round(float(precision), 5),
        "recall": round(float(recall), 5),
        "f1": round(float(f1), 5),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_test)),
        "n_fraud_test": int(y_test.sum()),
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / f"{model}_eval.json").write_text(json.dumps(result, indent=2))
    _plot_pr_curve(model, y_test, test_pred, result["test_ap"])

    print(
        f"{model}: test AUPRC={result['test_ap']:.5f}  ROC-AUC={result['test_roc_auc']:.5f}\n"
        f"  @thr={threshold:.4g} ({policy}): "
        f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}  "
        f"(TP={tp} FP={fp} FN={fn} of {int(y_test.sum())} frauds)"
    )
    return result


def _plot_pr_curve(model: str, y_test: np.ndarray, test_pred: np.ndarray, ap: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test, test_pred)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AUPRC={ap:.4f}")
    ax.axhline(float(y_test.mean()), ls="--", c="grey", label=f"baseline={y_test.mean():.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model} — held-out PR curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{model}_pr_curve.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="blend", help="model key or 'blend'")
    parser.add_argument("--policy", default="max-f1", choices=("max-f1", "precision-floor"))
    parser.add_argument("--min-precision", type=float, default=0.9)
    args = parser.parse_args()
    evaluate(args.model, policy=args.policy, min_precision=args.min_precision)


if __name__ == "__main__":
    main()
