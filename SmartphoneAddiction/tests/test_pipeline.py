import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from src.artifacts import save_run
from src.blend import cross_fitted_stack
from src.cv import make_folds
from src.data import CATEGORICAL, FEATURES, ID, TARGET, _validate, map_original
from src.features import make_features
from src.metric import metrics
from src.submit import make_submission

def frame(n=20):
    rng=np.random.default_rng(3); d={c:rng.normal(size=n) for c in FEATURES if c not in CATEGORICAL}
    d.update(gender=np.resize(["Male","Female",None],n),stress_level=np.resize(["Low","High"],n),academic_work_impact=np.resize(["Yes","No"],n))
    d[ID]=np.arange(n); d[TARGET]=np.arange(n)%2
    return pd.DataFrame(d)

def test_schema_and_target():
    d=frame(); _validate(d,True); d.loc[0,TARGET]=2
    with pytest.raises(ValueError): _validate(d,True)

def test_features_missing_unseen_and_finite_ratios():
    d=frame(); d.loc[0,"daily_screen_time_hours"]=0
    for group in ["base","behavior","compact","source_rule","missing"]:
        x=make_features(d,group)
        assert not np.isinf(x.select_dtypes(include=np.number)).any().any()
    assert make_features(d,"source_rule").loc[0,"source_usage_band"] in [0,1,2]

def test_folds_deterministic_disjoint_complete():
    y=np.resize([0,1],100); a=make_folds(y,5,42); b=make_folds(y,5,42)
    assert np.array_equal(a,b) and set(a)==set(range(5)) and np.bincount(a).sum()==len(y)

def test_original_mapping_drops_leaky_columns():
    d=frame(4).drop(columns=ID).rename(columns={c:next((k for k,v in __import__('src.data',fromlist=['ORIGINAL_ALIASES']).ORIGINAL_ALIASES.items() if v==c),c) for c in FEATURES+[TARGET]})
    d["transaction_id"]=[1,2,3,4]; d["addiction_level"]=[5]*4
    out=map_original(d); assert list(out)==FEATURES+[TARGET]

def test_metrics_match_sklearn():
    y=np.array([0,1,0,1]); p=np.array([.1,.8,.4,.7])
    assert metrics(y,p)["roc_auc"]==roc_auc_score(y,p)

def test_artifact_and_submission_alignment(tmp_path):
    save_run(tmp_path,"m",np.array([.1,.2]),np.array([.3,.4]),{"ok":True})
    assert np.load(tmp_path/"m_oof.npy").shape==(2,) and json.loads((tmp_path/"m_metrics.json").read_text())["ok"]
    sample=tmp_path/"sample.csv"; pd.DataFrame({"id":[4,5],"addicted_label":[.5,.5]}).to_csv(sample,index=False)
    out=make_submission(sample,[4,5],[.2,.8],tmp_path/"sub.csv"); assert list(out)==[ID,TARGET]
    with pytest.raises(ValueError): make_submission(sample,[5,4],[.2,.8],tmp_path/"bad.csv")

def test_cross_fitted_nonnegative_stack():
    y=np.resize([0,1],100); f=make_folds(y,5); x=np.c_[y*.8+.1,1-(y*.8+.1)]; xt=x[:10]
    o,t=cross_fitted_stack(x,y,f,xt); assert o.shape==(100,) and t.shape==(10,) and ((o>=0)&(o<=1)).all()
