"""Sanity checks for the NIPS4B data loaders and shared folds.

These assume the competition archives are extracted under ``data/raw/`` (see
CLAUDE.md). They validate schema shapes and the submission-id contract so the
downstream pipeline can trust the loaders.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import cv, data

pytestmark = pytest.mark.skipif(
    not data._LABELS_CSV.exists(), reason="competition data not downloaded"
)


def test_species_list():
    sp = data.load_species()
    assert len(sp) == data.N_CLASSES == 87
    assert sp["class number"].tolist() == list(range(1, 88))
    names = data.class_names()
    assert names[0] == "Aegcau_call"
    assert len(names) == 87


def test_train_labels_schema():
    file_ids, Y, dur = data.load_train_labels()
    assert file_ids.tolist() == list(range(1, data.N_TRAIN + 1))
    assert Y.shape == (data.N_TRAIN, data.N_CLASSES) == (687, 87)
    assert set(np.unique(Y)).issubset({0, 1})
    assert (dur > 0).all()
    # multi-label: some clips have several classes, some are empty (all-zero rows)
    per_clip = Y.sum(axis=1)
    assert per_clip.max() >= 2
    assert (per_clip == 0).any()


def test_mfcc_load_shape():
    m = data.load_mfcc(data.train_mfcc_path(1))
    assert m.ndim == 2 and m.shape[1] == data.N_MFCC == 17
    assert m.shape[0] > 1  # multiple time frames


def test_paths_exist_for_first_files():
    assert data.train_wav_path(1).exists()
    assert data.test_wav_path(1).exists()
    assert data.train_mfcc_path(1).exists()
    assert data.test_mfcc_path(1).exists()


def test_submission_id_format():
    assert data.submission_id(1, 1) == "nips4b_birds_testfile0001.wav_classnumber_1"
    ids = data.submission_ids()
    assert len(ids) == data.N_TEST * data.N_CLASSES == 87_000
    # matches the organizer example submission's id set
    sample_ids = set(data.load_sample_submission()["ID"])
    assert set(ids) == sample_ids


def test_folds_partition():
    Y = data.train_targets()
    folds = cv.make_folds(Y)
    assert len(folds) == cv.N_SPLITS
    covered = np.zeros(len(Y), dtype=int)
    for tr, va in folds:
        assert len(set(tr) & set(va)) == 0
        covered[va] += 1
    assert (covered == 1).all()  # every row validated exactly once
