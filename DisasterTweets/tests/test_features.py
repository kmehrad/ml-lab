"""Tests for text cleaning and field construction."""
from __future__ import annotations

import pandas as pd

from src import data, features


def test_clean_strips_urls_and_mentions():
    s = features.clean_text("Check this http://t.co/abc @user FIRE!! &amp;")
    assert "http" not in s and "@user" not in s and "&amp;" not in s
    assert "fire" in s  # kept, lowercased


def test_clean_light_keeps_case_and_punct():
    s = features.clean_text("Big FIRE downtown! http://x.co/y", light=True)
    assert "FIRE" in s and "!" in s and "http" not in s


def test_build_field_prepends_decoded_keyword():
    df = pd.DataFrame({
        data.KEYWORD_COL: ["suicide%20bomber"],
        data.TEXT_COL: ["A tragic event #news"],
    })
    out = features.build_field(df)
    assert out.iloc[0].startswith("suicide bomber")
    assert "news" in out.iloc[0] and "#" not in out.iloc[0]


def test_build_field_handles_missing_keyword():
    df = pd.DataFrame({data.KEYWORD_COL: [None], data.TEXT_COL: ["hello world"]})
    out = features.build_field(df)
    assert out.iloc[0] == "hello world"


def test_fix_duplicate_labels_majority_vote():
    df = pd.DataFrame({
        data.KEYWORD_COL: [None] * 5,
        data.TEXT_COL: ["dup", "dup", "dup", "unique", "unique2"],
        data.TARGET: [1, 1, 0, 0, 1],  # 'dup' majority = 1
    })
    fixed = features.fix_duplicate_labels(df)
    assert list(fixed[fixed[data.TEXT_COL] == "dup"][data.TARGET]) == [1, 1, 1]
    # non-duplicated rows untouched
    assert fixed.loc[3, data.TARGET] == 0 and fixed.loc[4, data.TARGET] == 1
