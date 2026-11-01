"""Can the network memorize 32 REAL training trials? Not a generalization score."""
import argparse
from pathlib import Path
import numpy as np
import torch
from neuroedge.accuracy_v2 import load_plan, fit_neural, get_data
from neuroedge.config import json_write
from neuroedge.data import select_reference
from neuroedge.experiment import probabilities
from neuroedge.learning_diagnostics import metrics

parser=argparse.ArgumentParser()
parser.add_argument("--plan",default="configs/accuracy_v2.yaml")
parser.add_argument("--output",default="runs/local_memorization_control")
args=parser.parse_args()
out=Path(args.output)
if (out/"result.json").exists():
    raise SystemExit("Existing control results preserved; choose a new --output.")
plan=load_plan(args.plan)
plan["learning_rate"]=0.003
torch.set_num_threads(plan["threads"])
X,meta,names=get_data(plan,plan["development_subjects"],"development_data")
indices=np.concatenate([np.flatnonzero((meta.subject==1)&(meta.label==label))[:16] for label in [0,1]])
data=select_reference(X[indices],names,16)
small=meta.iloc[indices].reset_index(drop=True)
spec={"name":"memorization_control_32_training_trials","bn_momentum":0.1,"dropout":0.0}
model,scaler,_,_=fit_neural(data,small,[1],[],spec,plan,out,fixed_epochs=150)
result=metrics(small.label.to_numpy(np.int64),probabilities(model,scaler.transform(data)))
result.update({"scope":"TRAINING MEMORIZATION ONLY; NOT a generalization score","trials":len(data),"subject":1})
json_write(out/"result.json",result)
json_write(out/"protocol.json",{"epochs":150,"learning_rate":0.003,"seed":plan["seed"],"candidate":spec,
                                "trial_ids":small.trial_id.tolist(),"test_data_used":False})
print(result)

