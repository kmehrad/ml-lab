"""One-time: decode the competition TFRecords to JPEG files + a metadata parquet.

The TFRecords store images as raw JPEG bytes, so we write those bytes straight to
``.jpg`` (lossless, no re-encode). Output:

    data/processed/jpeg-{res}/{split}/{id}.jpg
    data/processed/metadata_{res}.parquet   # id, split, label, class_name, path

TensorFlow is used *only* here (install via ``uv sync --extra convert``); the train /
predict path stays TF-free. Run on the GPU box after download.

Usage:
    uv run python -m src.convert --res 224 --splits train,val,test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import data

# TFRecord feature schemas. Labeled splits carry ``class``; test does not.
_LABELED = "image", "id", "class"
_UNLABELED = "image", "id"


def _feature_spec(labeled: bool):
    import tensorflow as tf  # local import: TF only needed for conversion

    spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),
    }
    if labeled:
        spec["class"] = tf.io.FixedLenFeature([], tf.int64)
    return spec


def _tfrec_dir(res: int, split: str) -> Path:
    return data.RAW_DIR / f"tfrecords-jpeg-{res}x{res}" / split


def convert_split(res: int, split: str) -> list[dict]:
    """Decode every ``.tfrec`` in one split, write JPEGs, return metadata rows."""
    import tensorflow as tf

    labeled = split in ("train", "val")
    spec = _feature_spec(labeled)

    src_dir = _tfrec_dir(res, split)
    files = sorted(str(p) for p in src_dir.glob("*.tfrec"))
    if not files:
        raise FileNotFoundError(
            f"No .tfrec files in {src_dir}. Download/unzip the competition data first."
        )

    out_dir = data.PROCESSED_DIR / f"jpeg-{res}" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    rows: list[dict] = []
    for raw in ds:
        ex = tf.io.parse_single_example(raw, spec)
        img_id = ex["id"].numpy().decode("utf-8")
        label = int(ex["class"].numpy()) if labeled else None
        rel_path = f"data/processed/jpeg-{res}/{split}/{img_id}.jpg"
        (data.ROOT / rel_path).write_bytes(ex["image"].numpy())
        rows.append(
            {
                "id": img_id,
                "split": split,
                "label": label,
                "class_name": data.CLASSES[label] if labeled else None,
                "path": rel_path,
            }
        )
    print(f"  {split}: wrote {len(rows)} JPEGs -> {out_dir}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode TFRecords -> JPEG + metadata parquet")
    ap.add_argument("--res", type=int, default=224, choices=data.RESOLUTIONS)
    ap.add_argument("--splits", default="train,val,test",
                    help="comma-separated subset of train,val,test")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    bad = [s for s in splits if s not in ("train", "val", "test")]
    if bad:
        ap.error(f"unknown split(s): {bad}")

    all_rows: list[dict] = []
    for split in splits:
        all_rows.extend(convert_split(args.res, split))

    df = pd.DataFrame(all_rows, columns=data.METADATA_COLS)
    df["label"] = df["label"].astype("Int64")  # nullable: test has no label

    out = data.metadata_path(args.res)
    # Merge with any previously-converted splits so partial runs accumulate.
    if out.exists():
        prev = pd.read_parquet(out)
        prev = prev[~prev["split"].isin(splits)]
        df = pd.concat([prev, df], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\nmetadata -> {out}")
    print(df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
