"""GPU (cuML) combination target-encoding — the S5E6 1st-place lever.

Reproduces the winning recipe with cuML's own ``TargetEncoder``: treat all 8 features as categorical,
form every combination up to ``order`` (singles..quadruples → 162 keys), and target-encode each key
against the 7 binary class indicators ``y==k``. Two encodings are produced per key (so
2 × n_combos × 7 columns when ``use_original``):

  * **synthetic TE** — encoded from the *train* labels. cuML ``TargetEncoder`` does **k-fold
    interleaved out-of-fold** encoding on train (leak-safe) and a full-train mapping on valid/test.
  * **original-data TE** — the encoder is *fit* on the 99-row UCI dataset (``data.load_original``) and
    *transformed* onto train/valid/test. External data → leak-free by construction (the "original data
    as columns" trick).

NB an earlier hand-rolled leave-one-out version failed: LOO + smoothing put the train encodings in a
different distribution than the full-train-map valid encodings, so XGBoost overfit to ``best_iter=0``.
cuML's k-fold OOF keeps train/valid encodings in the same regime and generalizes (see project memory).

Returns cupy float32 arrays ready for ``xgboost`` GPU (``device="cuda"``).
"""
from __future__ import annotations

import numpy as np

from .data import N_CLASSES
from .target_encoding import COMBO_FEATURES, combos  # reuse combo enumeration + feature list

__all__ = ["combos", "COMBO_FEATURES", "build_te_features"]


def build_te_features(Xtr_raw, ytr, Xva_raw, Xte_raw, combo_list,
                      m: float = 0.0, use_original: bool = True, orig_df=None):
    """Leak-safe GPU TE for every combo via cuML. Returns ``(Xtr, Xva, Xte, names)`` as cupy float32.

    ``Xtr/Xva/Xte_raw`` are pandas frames (raw features); ``ytr`` int codes. Columns are laid out per
    combo as ``[synthetic 7 classes]`` then (if ``use_original``) ``[original 7 classes]``. ``m`` is the
    cuML ``smooth`` weight (0 = raw group mean, the winner's default). ``Xte`` is ``None`` when
    ``Xte_raw`` is ``None`` (sample runs skip test).
    """
    import cudf
    import cupy as cp
    from cuml.preprocessing import TargetEncoder
    from . import data as D

    have_te = Xte_raw is not None
    gtr = cudf.from_pandas(Xtr_raw[COMBO_FEATURES].reset_index(drop=True))
    gva = cudf.from_pandas(Xva_raw[COMBO_FEATURES].reset_index(drop=True))
    gte = cudf.from_pandas(Xte_raw[COMBO_FEATURES].reset_index(drop=True)) if have_te else None
    ytr_g = cudf.Series(np.asarray(ytr))

    if use_original:
        if orig_df is None:
            orig_df = D.load_original()
        gor = cudf.from_pandas(orig_df[COMBO_FEATURES].reset_index(drop=True))
        yor_g = cudf.Series(D.encode_target(orig_df[D.TARGET]))

    tr_parts, va_parts, te_parts, names = [], [], [], []

    def _enc(fit_X, fit_y, prefix, tag):
        """Fit a per-class TargetEncoder on (fit_X, fit_y==j); append encoded tr/va/te columns."""
        for j in range(N_CLASSES):
            te = TargetEncoder(smooth=m, output_type="cupy")
            yj = (fit_y == j).astype("float32")
            enc_fit = te.fit_transform(fit_X, yj)                # OOF on the fitted rows
            if fit_X is sub_tr:                                  # synthetic: fitted on train
                tr_parts.append(enc_fit.reshape(-1, 1))
            else:                                                # original: transform train
                tr_parts.append(te.transform(sub_tr).reshape(-1, 1))
            va_parts.append(te.transform(sub_va).reshape(-1, 1))
            if have_te:
                te_parts.append(te.transform(sub_te).reshape(-1, 1))
        names.extend(f"{prefix}_{'_'.join(combo)}_{c}" for c in range(N_CLASSES))

    for combo in combo_list:
        cols = list(combo)
        sub_tr, sub_va = gtr[cols], gva[cols]
        sub_te = gte[cols] if have_te else None
        _enc(sub_tr, ytr_g, "te", "synthetic")
        if use_original:
            _enc(gor[cols], yor_g, "teorig", "original")

    Xtr = cp.hstack(tr_parts)
    Xva = cp.hstack(va_parts)
    Xte = cp.hstack(te_parts) if have_te else None
    return Xtr, Xva, Xte, names
