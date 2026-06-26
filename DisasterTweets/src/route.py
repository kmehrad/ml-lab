"""Disagreement-gated LLM ensemble — the LLM only judges the *ambiguous* tweets.

Idea: the two trained models (RoBERTa, TF-IDF) agree on the easy/non-ambiguous tweets (~88%);
take that shared label with **no LLM call**. They disagree on the ambiguous ~12%; resolve those
with a 3-way **majority vote** that brings in the LLM as the deciding vote. Under majority vote
the LLM can only change rows where RoBERTa and TF-IDF disagree, so we query it on exactly those
rows — cutting LLM calls ~8x versus scoring the whole set.

Usage:
    uv run python -m src.route --llm qwen3 --shots 0 --split val      # estimate F1 (cached)
    uv run python -m src.route --llm qwen3 --shots 0 --split test     # build submission

On ``val`` the ambiguous rows are a subset of the cached 400-tweet slice, so it costs 0 calls.
On ``test`` the LLM is queried (only on disagreements) and a submission CSV is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src import data, llm_eval

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "artifacts"
OUT = ROOT / "outputs"
VAL_N = 400


def _thr(model: str) -> float:
    return json.load(open(ART / f"{model}_metrics.json"))["threshold"]


def _load(split: str, n: int):
    """Return (df[id,keyword,text], dec_roberta, dec_tfidf, y_or_None)."""
    thr_r, thr_t = _thr("roberta"), _thr("tfidf_lr")
    if split == "val":
        tr = data.load_train()
        y_all = tr[data.TARGET].to_numpy()
        idx = llm_eval.val_indices(y_all, n)
        p_r = np.load(ART / "roberta_oof.npy")[idx]
        p_t = np.load(ART / "tfidf_lr_oof.npy")[idx]
        df = tr.iloc[idx][[data.ID_COL, data.KEYWORD_COL, data.TEXT_COL]].reset_index(drop=True)
        y = y_all[idx]
        exclude = set(idx)
    else:
        df = data.load_test()
        p_r = np.load(ART / "roberta_test.npy")
        p_t = np.load(ART / "tfidf_lr_test.npy")
        y = None
        exclude = set()
    dec_r = (p_r >= thr_r).astype(int)
    dec_t = (p_t >= thr_t).astype(int)
    return df, dec_r, dec_t, y, exclude


def route(llm_key: str, shots: int, split: str, n: int = VAL_N, workers: int = 4) -> dict:
    df, dec_r, dec_t, y, exclude = _load(split, n)
    ids = df[data.ID_COL].to_numpy()
    amb = dec_r != dec_t  # the only rows where an LLM tiebreaker can change the vote

    # query the LLM on ambiguous rows only (val: all cached -> 0 calls)
    from groq import Groq
    client = Groq()
    model = llm_eval.MODELS.get(llm_key, llm_key)
    fewshot = llm_eval._build_fewshot(data.load_train(), exclude, shots)
    sub = df[amb]
    preds, fails = llm_eval.predict_frame(
        client, model, fewshot, sub, f"{llm_key}_{split}_s{shots}", shots, workers)

    # LLM label per row (default to RoBERTa where missing/failed -> neutral in the vote)
    llm = np.array([preds.get(int(i), dec_r[k]) for k, i in enumerate(ids)])
    vote = ((dec_r + dec_t + llm) >= 2).astype(int)
    routed = dec_r.copy()
    routed[amb] = vote[amb]

    info = {"llm": model, "shots": shots, "split": split, "n": len(ids),
            "ambiguous": int(amb.sum()), "ambiguous_pct": round(float(amb.mean()), 4),
            "llm_calls_made": int(len(sub) - (len(sub) - len(preds)) - fails),
            "failures": int(fails)}
    if y is not None:
        info.update({
            "f1_routed": round(f1_score(y, routed), 5),
            "f1_roberta_alone": round(f1_score(y, dec_r), 5),
            "precision": round(precision_score(y, routed), 5),
            "recall": round(recall_score(y, routed), 5),
        })
        delta = info["f1_routed"] - info["f1_roberta_alone"]
        print(f"[route/{split}] LLM={llm_key} s{shots}  F1 routed={info['f1_routed']} "
              f"vs RoBERTa-alone {info['f1_roberta_alone']} (delta {delta:+.5f})  "
              f"ambiguous {amb.sum()}/{len(ids)} ({amb.mean():.0%})")
    else:
        OUT.mkdir(exist_ok=True)
        import pandas as pd
        path = OUT / f"route_{llm_key}_s{shots}_submission.csv"
        pd.DataFrame({data.ID_COL: ids, data.TARGET: routed}).to_csv(path, index=False)
        info["submission"] = str(path)
        print(f"[route/{split}] wrote {path}  (pos rate {routed.mean():.3f}, "
              f"LLM queried {amb.sum()}/{len(ids)} ambiguous rows, {fails} failures)")
    with open(ART / f"route_{llm_key}_s{shots}_{split}.json", "w") as fh:
        json.dump(info, fh, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="qwen3", help="llama4|qwen3|... (Groq model)")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=VAL_N)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    route(args.llm, args.shots, args.split, args.n, args.workers)


if __name__ == "__main__":
    main()
