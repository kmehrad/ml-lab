"""Optuna tuning on one fixed stratified development sample."""
import argparse, json
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from .data import TARGET, load_data
from .train import run_cv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",choices=["lgbm","xgb","cat"],required=True); ap.add_argument("--trials",type=int,default=30); ap.add_argument("--sample",type=int,default=250000); a=ap.parse_args()
    train,test=load_data(); idx,_=train_test_split(np.arange(len(train)),train_size=min(a.sample,len(train)),stratify=train[TARGET],random_state=42); dev=train.iloc[idx]
    def objective(t):
        common={"learning_rate":t.suggest_float("learning_rate",.015,.12,log=True)}
        if a.model=="lgbm": common|={"num_leaves":t.suggest_int("num_leaves",16,128),"min_child_samples":t.suggest_int("min_child_samples",20,300)}
        elif a.model=="xgb": common|={"max_depth":t.suggest_int("max_depth",3,10),"min_child_weight":t.suggest_float("min_child_weight",1,20)}
        else: common|={"depth":t.suggest_int("depth",5,10),"l2_leaf_reg":t.suggest_float("l2_leaf_reg",1,20)}
        *_, scores=run_cv(dev,test.iloc[:100],a.model,"behavior",3,42,params=common); return float(np.mean(scores))
    study=optuna.create_study(direction="maximize",study_name=f"{a.model}_s6e8"); study.optimize(objective,n_trials=a.trials); print(json.dumps({"value":study.best_value,"params":study.best_params},indent=2))
if __name__=="__main__": main()
