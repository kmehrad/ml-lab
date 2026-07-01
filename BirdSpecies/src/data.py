"""Data access for the **NIPS4B 2013 Bird Challenge**
(Kaggle `multilabel-bird-species-classification-nips2013`).

Multi-label audio: **687 train + 1000 test** recordings, **87 classes**
(species × vocalization type, e.g. call vs song counted separately). Each file
ships as raw WAV (44.1 kHz mono 16-bit, ~variable length) **and** precomputed
MFCC cepstra (17 coefficients × N frames). Metric: pooled ROC-AUC over all
`(test file × class)` pairs. Submission id is
``nips4b_birds_testfile{NNNN}.wav_classnumber_{C}`` (C = 1..87).

Loaders raise on schema mismatch and cache with ``lru_cache``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
LABELS_DIR = RAW / "NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS"
MFCC_DIR = RAW / "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_MFCC"
WAV_DIR = RAW / "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV"

# ── Schema constants ─────────────────────────────────────────────────
N_CLASSES = 87        # species×vocalization classes (excludes the 'Empty' sentinel)
N_TRAIN = 687
N_TEST = 1000
N_MFCC = 17           # cepstral coefficients per frame
SR = 44_100           # sample rate (Hz); audio is mono 16-bit

_LABELS_CSV = LABELS_DIR / "numero_file_train.csv"
_SPECIES_CSV = LABELS_DIR / "nips4b_birdchallenge_espece_list.csv"
_SAMPLE_SUB = LABELS_DIR / "example_NIPS4B13_submission_format_test_prediction.csv"


# ── Class / species list ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_species() -> pd.DataFrame:
    """87-row table: ``class number`` (1..87), ``class name`` (code like
    ``Aegcau_call``), English/Scientific name, ``type`` (bird/insect/…).
    The ``Empty`` (no-animal) sentinel row is dropped."""
    df = pd.read_csv(_SPECIES_CSV)
    df = df[df["class number"] != "none"].copy()
    df["class number"] = df["class number"].astype(int)
    df = df.sort_values("class number").reset_index(drop=True)
    if len(df) != N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} classes, got {len(df)}")
    return df


def class_names() -> list[str]:
    """87 class codes ordered by class number (index 0 → class 1 'Aegcau_call')."""
    return load_species()["class name"].tolist()


# ── Train labels ─────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_train_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``numero_file_train.csv`` (687 × 89, no header).

    Returns ``(file_ids[687], Y[687, 87] int8, durations[687] float)`` where
    column 0 is the train file number (1..687), columns 1..87 are the 0/1 class
    indicators, and the last column is the clip duration in seconds.
    """
    arr = np.loadtxt(_LABELS_CSV, delimiter=",")
    if arr.shape != (N_TRAIN, N_CLASSES + 2):
        raise ValueError(
            f"labels: expected {(N_TRAIN, N_CLASSES + 2)}, got {arr.shape}"
        )
    file_ids = arr[:, 0].astype(int)
    Y = arr[:, 1 : 1 + N_CLASSES].astype(np.int8)
    durations = arr[:, -1].astype(float)
    if not np.isin(Y, (0, 1)).all():
        raise ValueError("labels: non-binary entries in class columns")
    return file_ids, Y, durations


def train_targets() -> np.ndarray:
    """Just the 687×87 binary target matrix (row i ↔ train file i+1)."""
    return load_train_labels()[1]


# ── MFCC features (precomputed cepstra) ──────────────────────────────
def train_mfcc_path(file_id: int) -> Path:
    return MFCC_DIR / "train" / f"cepst_conc_cepst_nips4b_birds_trainfile{file_id:03d}.txt"


def test_mfcc_path(file_id: int) -> Path:
    return MFCC_DIR / "test" / f"cepst_conc_cepst_nips4b_birds_testfile{file_id:04d}.txt"


def load_mfcc(path: Path) -> np.ndarray:
    """Load one cepstra file as ``(n_frames, N_MFCC)``.

    Files are stored as ``N_MFCC`` rows × ``n_frames`` columns; we transpose so
    rows are time frames. A clip that yields a single frame is returned as
    ``(1, N_MFCC)``.
    """
    m = np.loadtxt(path)
    if m.ndim == 1:  # single frame OR single coefficient — disambiguate by length
        m = m.reshape(N_MFCC, -1) if m.shape[0] % N_MFCC == 0 else m[None, :]
    if m.shape[0] == N_MFCC:
        m = m.T
    if m.shape[1] != N_MFCC:
        raise ValueError(f"{path.name}: expected {N_MFCC} coeffs, got {m.shape}")
    return m


# ── Raw audio paths ──────────────────────────────────────────────────
def train_wav_path(file_id: int) -> Path:
    return WAV_DIR / "train" / f"nips4b_birds_trainfile{file_id:03d}.wav"


def test_wav_path(file_id: int) -> Path:
    return WAV_DIR / "test" / f"nips4b_birds_testfile{file_id:04d}.wav"


# ── Submission helpers ───────────────────────────────────────────────
def submission_id(test_file_id: int, class_number: int) -> str:
    """``nips4b_birds_testfile{NNNN}.wav_classnumber_{C}`` (C in 1..87)."""
    return f"nips4b_birds_testfile{test_file_id:04d}.wav_classnumber_{class_number}"


@lru_cache(maxsize=1)
def load_sample_submission() -> pd.DataFrame:
    """The organizer example submission (87001 rows: header + 1000×87 pairs)."""
    return pd.read_csv(_SAMPLE_SUB)


def submission_ids() -> list[str]:
    """Canonical 87,000 ids: test file 1..1000 × class 1..87 (file-major)."""
    return [
        submission_id(f, c)
        for f in range(1, N_TEST + 1)
        for c in range(1, N_CLASSES + 1)
    ]
