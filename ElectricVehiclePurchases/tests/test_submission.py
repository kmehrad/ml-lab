import numpy as np
import pandas as pd
import pytest

from src import submit


def test_submission_reorders_and_checks_probability(tmp_path, monkeypatch):
    train = pd.DataFrame({"id": [1, 2], "x": [1, 2], "Will_Buy_EV": ["No", "Yes"]})
    test = pd.DataFrame({"id": [11, 10], "x": [3, 4]})
    sample = pd.DataFrame({"id": [10, 11], "Will_Buy_EV": [0.5, 0.5]})
    monkeypatch.setattr(submit.data, "load_all", lambda: (train, test, sample))
    monkeypatch.setattr(submit, "ART", tmp_path)
    monkeypatch.setattr(submit, "OUT", tmp_path)
    np.save(tmp_path / "test_id.npy", np.array([11, 10]))
    np.save(tmp_path / "candidate_test.npy", np.array([0.9, 0.1]))
    path = submit.build_submission("candidate")
    result = pd.read_csv(path)
    assert result.columns.tolist() == ["id", "Will_Buy_EV"]
    assert result["id"].tolist() == [10, 11]
    assert result["Will_Buy_EV"].tolist() == [0.1, 0.9]
    np.save(tmp_path / "candidate_test.npy", np.array([1.2, 0.1]))
    with pytest.raises(ValueError, match="probabilities"):
        submit.build_submission("candidate")
