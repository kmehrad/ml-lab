import numpy as np
import pandas as pd

from src import submit, train


def test_logreg_cv_artifacts_to_submission(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(size=n)
    train_frame = pd.DataFrame({"id": np.arange(n), "income": x,
                                "region": np.where(x > 0, "north", "south"),
                                "Will_Buy_EV": np.where(x + rng.normal(scale=0.8, size=n) > 0,
                                                       "Yes", "No")})
    test_frame = pd.DataFrame({"id": [101, 100], "income": [1.0, -1.0],
                               "region": ["north", "south"]})
    sample = pd.DataFrame({"id": [100, 101], "Will_Buy_EV": [0.5, 0.5]})
    monkeypatch.setattr(train.data, "load_all", lambda: (train_frame, test_frame, sample))
    monkeypatch.setattr(train, "ART", tmp_path)
    monkeypatch.setattr(submit, "ART", tmp_path)
    monkeypatch.setattr(submit, "OUT", tmp_path)
    result = train.run("logreg")
    assert 0.5 < result["oof_auc"] <= 1.0
    path = submit.build_submission("logreg")
    submission = pd.read_csv(path)
    assert submission["id"].tolist() == [100, 101]
    assert submission["Will_Buy_EV"].between(0, 1).all()
