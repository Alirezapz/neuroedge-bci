"""Identity, participant separation and metrics for the frontier study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss
import yaml

from .config import digest, environment, json_write, sha256
from .frontier_features import candidates

CRITICAL = ["frontier.py","frontier_protocol.py","frontier_data.py","frontier_alignment.py",
            "frontier_features.py","frontier_neural.py","data.py","models.py","accuracy_v2.py"]


def load_plan(path):
    path=Path(path).resolve()
    plan=yaml.safe_load(path.read_text(encoding="utf-8"))
    groups=[plan[k] for k in ["discovery_subjects","validation_subjects","confirmation_subjects"]]
    flattened=[x for g in groups for x in g]
    if any(not g for g in groups) or len(set(flattened))!=len(flattened):
        raise ValueError("Empty participant group or participant leakage.")
    if any(not isinstance(s,int) or not 1<=s<=109 for s in flattened):
        raise ValueError("Invalid participant identifier.")
    if set(plan["confirmation_subjects"]) & set(range(1,81)):
        raise ValueError("Previously inspected participants cannot be called fresh frontier confirmation.")
    if plan["dataset"]["runs"]!=[4,8,12]:
        raise ValueError("Only imagined left/right fist runs 4,8,12 are supported.")
    if plan["search"]["folds"]<2:
        raise ValueError("At least two discovery group folds required.")
    plan["_root"]=str(path.parent.parent)
    return plan


def freeze(plan):
    out=Path(plan["_root"])/plan["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).parent
    public={k:v for k,v in plan.items() if not k.startswith("_")}
    grid=candidates()
    identity={"plan":public,"sources":{p:sha256(source/p) for p in CRITICAL},
              "candidate_grid_sha256":digest(grid)}
    identity["sha256"]=digest(identity)
    frozen=out/"protocol.json"
    if frozen.exists():
        saved=json.loads(frozen.read_text())
        if saved["sha256"]!=identity["sha256"]:
            raise RuntimeError("Frozen frontier protocol/code differs. Preserve this run and specify a new output directory.")
    else:
        json_write(frozen,identity)
        json_write(out/"candidates.json",grid)
        json_write(out/"environment.json",environment())
    return out,identity


def arm(spec):
    return spec.get("arm","zero_calibration")


def binary_probabilities(p):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():
        raise ValueError("Invalid binary probabilities.")
    if (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=2e-6):
        raise ValueError("Probabilities outside the simplex.")
    right=p[:,1]/p.sum(axis=1)
    return np.column_stack([1-right,right])


def subject_scores(meta,p):
    p=binary_probabilities(p)
    if p.shape!=(len(meta),2) or not np.isfinite(p).all():
        raise ValueError("Invalid class probability array.")
    if (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=2e-6):
        raise ValueError("Probabilities outside the simplex.")
    rows=[]
    for subject in sorted(meta.subject.unique()):
        mask=meta.subject.to_numpy()==subject
        y=meta.loc[mask,"label"].to_numpy(dtype=int)
        pred=p[mask].argmax(1)
        rows.append({"subject":int(subject),"trials":int(mask.sum()),
                     "accuracy":float(np.mean(pred==y)),
                     "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
                     "log_loss":float(log_loss(y,p[mask],labels=[0,1]))})
    return pd.DataFrame(rows)


def score(meta,p):
    scores=subject_scores(meta,p)
    return {"balanced_accuracy":float(scores.balanced_accuracy.mean()),
            "accuracy":float(scores.accuracy.mean()),"log_loss":float(scores.log_loss.mean())}


def bootstrap(values,plan):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(plan["confirmation"]["bootstrap_seed"])
    means=values[rng.integers(0,len(values),(plan["confirmation"]["bootstrap_iterations"],len(values)))].mean(axis=1)
    return list(map(float,np.quantile(means,[.025,.975])))


def predictions(meta,p):
    p=binary_probabilities(p)
    frame=meta[["trial_id","subject","run","onset","label"]].copy().reset_index(drop=True)
    frame["prediction"]=p.argmax(1)
    frame["prob_right"]=p[:,1]
    return frame
