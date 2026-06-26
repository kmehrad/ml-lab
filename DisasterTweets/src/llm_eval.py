"""Zero-/few-shot disaster-tweet classification via the Groq API.

Estimates F1 on a fixed, labeled validation slice of ``train`` BEFORE spending calls on
the unlabeled test set. Same models can later score the full test set (``--split test``).

Usage:
    uv run python -m src.llm_eval --model llama4 --n 400              # zero-shot val estimate
    uv run python -m src.llm_eval --model qwen3 --n 400 --shots 8     # few-shot
    uv run python -m src.llm_eval --model llama4 --split test         # full test predictions

Models (Groq):  llama4 -> meta-llama/llama-4-scout-17b-16e-instruct,
                qwen3  -> qwen/qwen3-32b,  llama33 -> llama-3.3-70b-versatile,
                gptoss -> openai/gpt-oss-120b   (or pass a full Groq model id).

Responses are cached at experiments/artifacts/llm_cache/{key}_{split}.jsonl keyed by tweet
id, so reruns are free. Predictions/metrics are saved like the other models so the LLM can
be blended (``{key}_oof.npy`` is only a partial-coverage val slice, so it is saved separately
as ``{key}_val.npy`` with its ids).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

from src import data

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "artifacts"
CACHE = ART / "llm_cache"
CACHE.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")

MODELS = {
    "llama4": "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen3": "qwen/qwen3-32b",
    "llama33": "llama-3.3-70b-versatile",
    "gptoss": "openai/gpt-oss-120b",
}

VAL_SEED = 42

SYSTEM = (
    "You are an expert content classifier for emergency and disaster monitoring. "
    "Given a tweet, decide whether it refers to a REAL disaster or emergency — an actual "
    "ongoing or reported event such as a natural disaster, fire, explosion, accident, crash, "
    "attack, outbreak, or similar — as opposed to a metaphorical, figurative, joking, "
    "headline-of-an-unrelated-topic, or otherwise non-disaster use of those words. "
    "Reply with exactly one character: 1 if it is about a real disaster, 0 if not. No other text."
)


def _user_msg(keyword, text: str) -> str:
    kw = "" if (keyword is None or (isinstance(keyword, float))) else str(keyword).replace("%20", " ")
    head = f"Keyword: {kw}\n" if kw else ""
    return f"{head}Tweet: {text}\nAnswer (1 or 0):"


def val_indices(y: np.ndarray, n: int) -> np.ndarray:
    """Deterministic stratified validation subset of size n."""
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=VAL_SEED)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return np.sort(idx)


def _retry_after(err: str, attempt: int) -> float:
    """Honor Groq's 'try again in X' hint when present, else exponential backoff."""
    m = re.search(r"try again in ([\d.]+)(ms|s)", err)
    if m:
        val = float(m.group(1))
        secs = val / 1000 if m.group(2) == "ms" else val
        return min(secs + 0.3, 30.0)
    return min(2 ** attempt, 30.0)


def _parse(content: str) -> int | None:
    if not content:
        return None
    s = content.strip()
    m = re.search(r'"?label"?\s*[:=]\s*([01])', s)
    if m:
        return int(m.group(1))
    digits = re.findall(r"\b([01])\b", s)
    if digits:
        return int(digits[-1])
    low = s.lower()
    if "not" in low or "non-disaster" in low or "no disaster" in low:
        return 0
    if "disaster" in low or "real" in low:
        return 1
    return None


def _build_fewshot(tr, exclude: set[int], shots: int) -> list[dict]:
    """K balanced few-shot examples drawn from train, excluding the val rows."""
    if shots <= 0:
        return []
    pool = tr[~tr.index.isin(exclude)]
    per = shots // 2
    ex = []
    for label, grp in [(1, pool[pool[data.TARGET] == 1]), (0, pool[pool[data.TARGET] == 0])]:
        for _, row in grp.sample(n=per, random_state=VAL_SEED).iterrows():
            ex.append((row[data.KEYWORD_COL], row[data.TEXT_COL], label))
    msgs = []
    for kw, txt, label in ex:
        msgs.append({"role": "user", "content": _user_msg(kw, txt)})
        msgs.append({"role": "assistant", "content": str(label)})
    return msgs


def _load_cache(path: Path) -> dict[int, int]:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[int(r["id"])] = r["pred"]
    return cache


def predict_frame(client, model, fewshot, df, cache_tag, shots, workers,
                  qwen_nothink: bool = True) -> tuple[dict[int, int], int]:
    """Classify the rows of ``df`` (cols id/keyword/text); return ({id: label}, n_failures).

    Cache is per ``cache_tag`` and keyed by tweet id, so any previously-scored ids are reused.
    Shared by :func:`classify` (whole split) and :mod:`src.route` (just the ambiguous subset).
    """
    cache_path = CACHE / f"{cache_tag}.jsonl"
    cache = _load_cache(cache_path)
    sys_msg = SYSTEM + (" /no_think" if (qwen_nothink and "qwen" in model) else "")
    base = [{"role": "system", "content": sys_msg}] + fewshot
    max_tok = 1024 if ("qwen" in model or "oss" in model) else 8

    def call(row_id, keyword, text):
        if row_id in cache:
            return row_id, cache[row_id], "cached"
        msgs = base + [{"role": "user", "content": _user_msg(keyword, text)}]
        for attempt in range(8):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=msgs, temperature=0.0,
                    max_completion_tokens=max_tok,
                )
                pred = _parse(resp.choices[0].message.content)
                if pred is None and attempt < 2:  # unparseable -> retry a couple times
                    time.sleep(1.0)
                    continue
                return row_id, (0 if pred is None else pred), "ok"
            except Exception as e:  # rate limit / transient
                wait = _retry_after(str(e), attempt)
                if attempt == 7:
                    print(f"  id {row_id}: failed after retries ({e})")
                    return row_id, None, "fail"  # None -> not cached, excluded
                time.sleep(wait)

    records = list(zip(df[data.ID_COL], df[data.KEYWORD_COL], df[data.TEXT_COL]))
    preds: dict[int, int] = {}
    fails = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex, open(cache_path, "a") as fh:
        futs = [ex.submit(call, rid, kw, tx) for rid, kw, tx in records]
        for i, fut in enumerate(as_completed(futs)):
            rid, pred, status = fut.result()
            if status == "fail" or pred is None:
                fails += 1
                continue
            preds[rid] = pred
            if status == "ok":
                fh.write(json.dumps({"id": int(rid), "pred": int(pred)}) + "\n")
                fh.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(records)} ({time.time()-t0:.0f}s)")
    return preds, fails


def classify(model_key: str, split: str, n: int, shots: int, workers: int,
             qwen_nothink: bool = True) -> dict:
    model = MODELS.get(model_key, model_key)
    key = model_key
    from groq import Groq
    client = Groq()

    tr = data.load_train()
    y_all = tr[data.TARGET].to_numpy()

    if split == "val":
        idx = val_indices(y_all, n)
        df = tr.iloc[idx]
        ids = df[data.ID_COL].to_numpy()
        y = y_all[idx]
        fewshot = _build_fewshot(tr, set(idx), shots)
    else:  # test
        df = data.load_test()
        ids = df[data.ID_COL].to_numpy()
        y = None
        fewshot = _build_fewshot(tr, set(), shots)

    # cache key includes shot count — few-shot answers differ from zero-shot for the same id
    cache_tag = f"{key}_{split}_s{shots}"
    t0 = time.time()
    preds, fails = predict_frame(client, model, fewshot, df, cache_tag, shots, workers, qwen_nothink)
    elapsed = time.time() - t0

    covered = np.array([i in preds for i in ids])
    pred_arr = np.array([preds.get(i, 0) for i in ids])
    np.save(ART / f"{key}_{split}_pred.npy", pred_arr)
    np.save(ART / f"{key}_{split}_ids.npy", ids)

    out = {"model": model, "key": key, "split": split, "n": len(ids),
           "covered": int(covered.sum()), "failures": int(fails),
           "shots": shots, "elapsed_s": round(elapsed, 1)}
    if y is not None:
        yc, pc = y[covered], pred_arr[covered]  # score only covered rows
        out.update({
            "f1": round(f1_score(yc, pc), 5),
            "precision": round(precision_score(yc, pc), 5),
            "recall": round(recall_score(yc, pc), 5),
            "accuracy": round(accuracy_score(yc, pc), 5),
            "pred_pos_rate": round(float(pc.mean()), 4),
            "true_pos_rate": round(float(yc.mean()), 4),
        })
        print(f"[{key}/{split}] F1={out['f1']} P={out['precision']} R={out['recall']} "
              f"acc={out['accuracy']} (covered {covered.sum()}/{len(ids)}, "
              f"shots={shots}, {elapsed:.0f}s)")
    else:
        print(f"[{key}/{split}] {len(ids)} predictions, pos rate "
              f"{pred_arr.mean():.3f} (covered {covered.sum()}/{len(ids)}, {elapsed:.0f}s)")
    with open(ART / f"{key}_{split}_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama4", help="llama4|qwen3|llama33|gptoss|<groq-id>")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=400, help="val sample size")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    classify(args.model, args.split, args.n, args.shots, args.workers)


if __name__ == "__main__":
    main()
