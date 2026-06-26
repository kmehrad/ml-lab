"""Text cleaning and field construction for DisasterTweets.

Two consumers with different needs:
- the classic TF-IDF model wants *cleaned* text + the (decoded) keyword prepended;
- the transformer wants *near-raw* text (its tokenizer handles casing/punctuation), so it
  uses a lighter normalization.

EDA (`reports/EDA_FINDINGS.md`) found 18 groups of exact-duplicate texts with conflicting
labels; :func:`fix_duplicate_labels` majority-votes those in the training frame, applied
behind a flag and validated via CV.
"""
from __future__ import annotations

import re
from html import unescape
from urllib.parse import unquote

import pandas as pd

from src import data

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HTML = re.compile(r"&\w+;")
_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9#@!?'\s]")


def clean_text(text: str, *, light: bool = False) -> str:
    """Normalize a tweet.

    ``light=True`` (transformer): unescape HTML entities, drop URLs, collapse whitespace —
    keep casing and punctuation. ``light=False`` (classic): also lowercase, drop @mentions,
    strip non-word punctuation, keep hashtag words.
    """
    s = unescape(str(text))
    s = _URL.sub(" ", s)
    if light:
        return _WS.sub(" ", s).strip()
    s = s.lower()
    s = _MENTION.sub(" ", s)
    s = s.replace("#", " ")  # keep the hashtag word, drop the '#'
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _decode_keyword(kw: object) -> str:
    if pd.isna(kw):
        return ""
    return unquote(str(kw)).replace("%20", " ")  # 'suicide%20bomber' -> 'suicide bomber'


def build_field(df: pd.DataFrame, *, light: bool = False, use_keyword: bool = True) -> pd.Series:
    """Return the model input series: ``keyword + cleaned text`` (keyword optional)."""
    cleaned = df[data.TEXT_COL].apply(lambda t: clean_text(t, light=light))
    if not use_keyword:
        return cleaned
    kw = df[data.KEYWORD_COL].apply(_decode_keyword)
    out = (kw + " " + cleaned).map(lambda s: _WS.sub(" ", s).strip())
    return out


def fix_duplicate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Majority-vote the label of exact-duplicate texts (ties -> keep original).

    Returns a copy; only touches rows whose text is duplicated with conflicting labels.
    """
    df = df.copy()
    grp = df.groupby(data.TEXT_COL)[data.TARGET]
    # majority label per text; round(0) breaks ties toward 0, so guard ties explicitly
    means = grp.transform("mean")
    counts = grp.transform("count")
    conflicting = (counts > 1) & (means != 0) & (means != 1)
    majority = (means > 0.5).astype(int)
    df.loc[conflicting, data.TARGET] = majority[conflicting]
    return df


if __name__ == "__main__":
    tr = data.load_train()
    print("classic:", build_field(tr).iloc[0])
    print("light  :", build_field(tr, light=True).iloc[0])
    fixed = fix_duplicate_labels(tr)
    changed = (fixed[data.TARGET].values != tr[data.TARGET].values).sum()
    print(f"duplicate-label fixes applied: {changed} rows")
