"""Schema and metadata for the Kaggle ``tpu-getting-started`` competition.

104-class flower image classification (metric: macro F1). The official data ships as
TFRecords designed for TPU. ``src/convert.py`` decodes them once to JPEG files plus a
``metadata_{res}.parquet`` index; this module pins the class list / counts and loads
that index so any drift fails loudly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Classes — canonical 104-name list (index == integer label), from the
# competition's tutorial notebook. Order is fixed; do not sort.
# ---------------------------------------------------------------------------
CLASSES: list[str] = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "wild geranium", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle",                                                            # 00-09
    "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily",                                                    # 10-19
    "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke",
    "sweet william",                                                            # 20-29
    "carnation", "garden phlox", "love in the mist", "cosmos", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose",                                                              # 30-39
    "barberton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup", "daisy", "common dandelion",         # 40-49
    "petunia", "wild pansy", "primula", "sunflower", "lilac hibiscus",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia",                                                       # 50-59
    "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush",
    "californian poppy", "osteospermum", "spring crocus", "iris", "windflower",
    "tree poppy",                                                               # 60-69
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium",                        # 70-79
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",          # 80-89
    "hippeastrum", "bee balm", "pink quill", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",      # 90-99
    "trumpet creeper", "blackberry lily", "common tulip", "wild rose",          # 100-103
]
N_CLASSES = len(CLASSES)
assert N_CLASSES == 104, f"expected 104 classes, got {N_CLASSES}"

# Expected per-split image counts (sanity only — competition may reship slightly).
N_TRAIN = 12753
N_VAL = 3712
N_TEST = 7382

# Resolutions available as TFRecord dirs after unzip.
RESOLUTIONS = (192, 224, 331, 512)

METADATA_COLS = ["id", "split", "label", "class_name", "path"]


def metadata_path(res: int = 224) -> Path:
    return PROCESSED_DIR / f"metadata_{res}.parquet"


def load_metadata(res: int = 224) -> pd.DataFrame:
    """Load the JPEG index produced by ``src/convert.py``.

    Columns: ``id, split, label, class_name, path``. ``label``/``class_name`` are
    ``<NA>`` for the unlabeled ``test`` split. ``path`` is relative to the repo root.
    """
    path = metadata_path(res)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download then convert the data first:\n"
            '  export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"\n'
            "  uv run kaggle competitions download -c tpu-getting-started -p data/raw\n"
            "  unzip -o data/raw/tpu-getting-started.zip -d data/raw\n"
            f"  uv run python -m src.convert --res {res} --splits train,val,test"
        )
    df = pd.read_parquet(path)
    missing = [c for c in METADATA_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; got {list(df.columns)}")
    return df


def split_frame(res: int = 224, split: str = "train") -> pd.DataFrame:
    """Rows for one split (``train`` / ``val`` / ``test``), index reset."""
    df = load_metadata(res)
    return df[df["split"] == split].reset_index(drop=True)


if __name__ == "__main__":
    print(f"{N_CLASSES} classes; expected counts train={N_TRAIN} val={N_VAL} test={N_TEST}")
    for res in RESOLUTIONS:
        p = metadata_path(res)
        print(f"  res {res}: {'present' if p.exists() else 'missing'} ({p})")
