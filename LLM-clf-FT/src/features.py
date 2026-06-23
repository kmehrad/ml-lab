"""Feature engineering for the LightGBM baseline.

Two feature families, neither of which uses model identity (absent at test time):

1. Hand-crafted numeric features per side (length, formatting markers) plus the
   A-vs-B differences/ratios that actually carry preference signal.
2. TF-IDF over the response text, reduced with TruncatedSVD, exposed as
   svd(a), svd(b), the A-B difference, and prompt-vs-response cosine similarities.

`FeatureBuilder` is fit on train only and reused (transform) on test so the same
vocabulary/SVD basis applies — this is what the inference notebook will import.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import TEXT_COLS, join_turns, parse_turns

_CODE = re.compile(r"```")
_BULLET = re.compile(r"(?m)^\s*[-*]\s")
_HEADER = re.compile(r"(?m)^\s*#{1,6}\s")


def _numeric_side(texts: list[str]) -> dict[str, np.ndarray]:
    """Scalar descriptors for one side's flattened conversation text."""
    char = np.array([len(t) for t in texts], dtype=float)
    words = np.array([t.count(" ") + 1 for t in texts], dtype=float)
    return {
        "char": char,
        "word": words,
        "avg_word": char / np.maximum(words, 1),
        "newline": np.array([t.count("\n") for t in texts], dtype=float),
        "code": np.array([len(_CODE.findall(t)) // 2 for t in texts], dtype=float),
        "bullet": np.array([len(_BULLET.findall(t)) for t in texts], dtype=float),
        "header": np.array([len(_HEADER.findall(t)) for t in texts], dtype=float),
        "digit": np.array([sum(c.isdigit() for c in t) for t in texts], dtype=float),
    }


def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    pa = df["prompt"].map(join_turns).tolist()
    a = df["response_a"].map(join_turns).tolist()
    b = df["response_b"].map(join_turns).tolist()
    n_turns = df["prompt"].map(lambda x: len(parse_turns(x))).to_numpy(float)

    feats: dict[str, np.ndarray] = {"n_turns": n_turns}
    sides = {"p": _numeric_side(pa), "a": _numeric_side(a), "b": _numeric_side(b)}
    for side, d in sides.items():
        for k, v in d.items():
            feats[f"{side}_{k}"] = v
    # A vs B: differences and ratios are the preference-bearing signal
    for k in sides["a"]:
        av, bv = sides["a"][k], sides["b"][k]
        feats[f"diff_{k}"] = av - bv
        feats[f"ratio_{k}"] = av / np.maximum(bv, 1.0)
    return pd.DataFrame(feats, index=df.index)


class FeatureBuilder:
    def __init__(self, svd_dim: int = 64, max_features: int = 50_000, seed: int = 42):
        self.svd_dim = svd_dim
        self.tfidf = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            sublinear_tf=True, min_df=3, strip_accents="unicode",
        )
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
        self.svd_p = TruncatedSVD(n_components=svd_dim, random_state=seed)

    def _texts(self, df: pd.DataFrame):
        return (
            df["prompt"].map(join_turns).tolist(),
            df["response_a"].map(join_turns).tolist(),
            df["response_b"].map(join_turns).tolist(),
        )

    def _embed(self, p, a, b, fit: bool):
        if fit:
            # fit shared response vocabulary on both A and B, prompt SVD on prompts
            self.tfidf.fit(a + b)
            self.svd.fit(self.tfidf.transform(a + b))
            self.svd_p.fit(self.tfidf.transform(p))
        ea = normalize(self.svd.transform(self.tfidf.transform(a)))
        eb = normalize(self.svd.transform(self.tfidf.transform(b)))
        ep = normalize(self.svd_p.transform(self.tfidf.transform(p)))
        cos_ab = (ea * eb).sum(1, keepdims=True)
        cos_pa = (ep * ea).sum(1, keepdims=True)
        cos_pb = (ep * eb).sum(1, keepdims=True)
        mat = np.hstack([ea, eb, ea - eb, ep, cos_ab, cos_pa, cos_pb])
        cols = (
            [f"svd_a{i}" for i in range(self.svd_dim)]
            + [f"svd_b{i}" for i in range(self.svd_dim)]
            + [f"svd_d{i}" for i in range(self.svd_dim)]
            + [f"svd_p{i}" for i in range(self.svd_dim)]
            + ["cos_ab", "cos_pa", "cos_pb"]
        )
        return pd.DataFrame(mat, columns=cols)

    def _build(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        num = _numeric_features(df).reset_index(drop=True)
        p, a, b = self._texts(df)
        emb = self._embed(p, a, b, fit=fit)
        out = pd.concat([num, emb], axis=1)
        out.index = df.index
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._build(df, fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._build(df, fit=False)
