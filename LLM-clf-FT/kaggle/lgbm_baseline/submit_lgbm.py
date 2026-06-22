"""Self-contained LightGBM baseline for the llm-classification-finetuning code
competition. Trains 5 fold models on the full train set and averages their
predictions on the hidden test, writing submission.csv. No internet, CPU only.

Mirrors src/features.py + src/train_baseline.py (inlined so the kernel is fully
self-contained). OOF log loss locally was 1.01541.
"""
import glob
import json
import os
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import vstack  # noqa: F401  (kept for parity with src)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]
SEED = 42
TURN_SEP = "\n\n"
_CODE = re.compile(r"```")
_BULLET = re.compile(r"(?m)^\s*[-*]\s")
_HEADER = re.compile(r"(?m)^\s*#{1,6}\s")


# ----------------------------------------------------------------- text utils
def parse_turns(x):
    if isinstance(x, list):
        return [s for s in x if isinstance(s, str)]
    try:
        v = json.loads(x)
    except Exception:
        return [str(x)]
    if isinstance(v, list):
        return [s if isinstance(s, str) else "" for s in v]
    return [str(v)]


def join_turns(x):
    return TURN_SEP.join(parse_turns(x))


# ----------------------------------------------------------------- features
def _numeric_side(texts):
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


def _numeric_features(df):
    pa = df["prompt"].map(join_turns).tolist()
    a = df["response_a"].map(join_turns).tolist()
    b = df["response_b"].map(join_turns).tolist()
    n_turns = df["prompt"].map(lambda x: len(parse_turns(x))).to_numpy(float)
    feats = {"n_turns": n_turns}
    sides = {"p": _numeric_side(pa), "a": _numeric_side(a), "b": _numeric_side(b)}
    for side, d in sides.items():
        for k, v in d.items():
            feats[f"{side}_{k}"] = v
    for k in sides["a"]:
        av, bv = sides["a"][k], sides["b"][k]
        feats[f"diff_{k}"] = av - bv
        feats[f"ratio_{k}"] = av / np.maximum(bv, 1.0)
    return pd.DataFrame(feats, index=df.index)


class FeatureBuilder:
    def __init__(self, svd_dim=64, max_features=50_000, seed=42):
        self.svd_dim = svd_dim
        self.tfidf = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            sublinear_tf=True, min_df=3, strip_accents="unicode",
        )
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
        self.svd_p = TruncatedSVD(n_components=svd_dim, random_state=seed)

    def _texts(self, df):
        return (df["prompt"].map(join_turns).tolist(),
                df["response_a"].map(join_turns).tolist(),
                df["response_b"].map(join_turns).tolist())

    def _embed(self, p, a, b, fit):
        if fit:
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
        cols = ([f"svd_a{i}" for i in range(self.svd_dim)]
                + [f"svd_b{i}" for i in range(self.svd_dim)]
                + [f"svd_d{i}" for i in range(self.svd_dim)]
                + [f"svd_p{i}" for i in range(self.svd_dim)]
                + ["cos_ab", "cos_pa", "cos_pb"])
        return pd.DataFrame(mat, columns=cols)

    def _build(self, df, fit):
        num = _numeric_features(df).reset_index(drop=True)
        p, a, b = self._texts(df)
        emb = self._embed(p, a, b, fit=fit)
        out = pd.concat([num, emb], axis=1)
        out.index = df.index
        return out

    def fit_transform(self, df):
        return self._build(df, fit=True)

    def transform(self, df):
        return self._build(df, fit=False)


PARAMS = dict(
    objective="multiclass", num_class=3, metric="multi_logloss",
    learning_rate=0.03, num_leaves=63, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, min_child_samples=50,
    lambda_l2=1.0, seed=SEED, num_threads=0, verbose=-1,
)


def find_input_dir():
    """Locate the mounted competition data, robust to the exact folder name."""
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if not hits:
        print("contents of /kaggle/input:")
        for root, dirs, files in os.walk("/kaggle/input"):
            print(" ", root, "->", files[:10])
        raise FileNotFoundError("train.csv not found under /kaggle/input")
    return os.path.dirname(hits[0])


def main():
    IN = find_input_dir()
    print("input dir:", IN)
    train = pd.read_csv(f"{IN}/train.csv")
    test = pd.read_csv(f"{IN}/test.csv")
    print("train", train.shape, "test", test.shape)
    y = train[TARGETS].values.argmax(1)

    fb = FeatureBuilder(seed=SEED)
    X = fb.fit_transform(train).values.astype(np.float32)
    X_test = fb.transform(test).values.astype(np.float32)
    print("features", X.shape, X_test.shape)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3))
    test_pred = np.zeros((len(test), 3))
    n = skf.get_n_splits()
    for f, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.train(
            PARAMS, lgb.Dataset(X[tr], y[tr]), num_boost_round=3000,
            valid_sets=[lgb.Dataset(X[va], y[va])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict(X[va], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / n
        print(f"fold {f}: best_iter={model.best_iteration} "
              f"ll={log_loss(y[va], oof[va], labels=[0, 1, 2]):.5f}")

    print(f"OOF log loss: {log_loss(y, oof, labels=[0, 1, 2]):.5f}")
    sub = test[["id"]].copy()
    sub[TARGETS] = test_pred
    sub.to_csv("submission.csv", index=False)
    print("wrote submission.csv", sub.shape)
    print(sub.head())


if __name__ == "__main__":
    main()
