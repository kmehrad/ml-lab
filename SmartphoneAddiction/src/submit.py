from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def make_submission(sample_path, test_ids, predictions, output_path):
    sample = pd.read_csv(sample_path)
    ids, pred = np.asarray(test_ids), np.asarray(predictions, float)
    if list(sample.columns) != ["id", "addicted_label"] or len(sample) != len(pred):
        raise ValueError("submission shape or columns do not match sample")
    if not np.array_equal(sample.id.to_numpy(), ids):
        raise ValueError("submission IDs are not exactly aligned")
    if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
        raise ValueError("predictions must be finite probabilities in [0, 1]")
    sample["addicted_label"] = pred
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    return sample


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tag",required=True); ap.add_argument("--output")
    a=ap.parse_args(); artifacts=Path("experiments/artifacts")
    make_submission("data/raw/sample_submission.csv", np.load(artifacts/"test_id.npy"),
                    np.load(artifacts/f"{a.tag}_test.npy"), a.output or f"outputs/{a.tag}_submission.csv")
if __name__ == "__main__": main()
