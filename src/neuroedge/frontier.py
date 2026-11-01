"""Broad, resumable development, ensembles and one prospective confirmation.

python -m neuroedge.frontier all --plan configs/frontier.yaml
Only data from participants 1..80 drive search, stacking and selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
from multiprocessing import get_context
import os
from pathlib import Path
import time
import traceback
import uuid

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import torch

from .config import digest, json_write, sha256
from .frontier_protocol import arm, bootstrap, freeze, load_plan, predictions, score, subject_scores
from .frontier_features import (candidates, classifier, CovarianceFeatures,
                                CovarianceModel, feature_identity)
from .frontier_data import load_cohort, prepare_cohort
from .frontier_alignment import load_rest, prepare_rest
from .frontier_neural import fit_neural, load_neural

_WORKER={}


def discovery_splits(plan):
    subjects=np.asarray(sorted(plan["discovery_subjects"]))
    split=KFold(n_splits=plan["search"]["folds"],shuffle=True,random_state=plan["search"]["split_seed"])
    return [{"fold":i,"train_subjects":subjects[train].tolist(),"validation_subjects":subjects[val].tolist()}
            for i,(train,val) in enumerate(split.split(subjects))]


def _worker_init(plan):
    torch.set_num_threads(plan["search"]["threads"])
    torch.set_num_interop_threads(1)
    cohort=load_cohort(plan,"development_data")
    cohort=load_rest(plan,cohort,"development_data")
    _WORKER.update(plan=plan,cohort=cohort)


def _atomic_joblib(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    joblib.dump(value,tmp,compress=3)
    os.replace(tmp,path)


def _feature_cache(plan,cohort,spec,train,val,y):
    source=Path(__file__).parent
    key=digest({"feature":feature_identity(spec),
        "data":cohort.manifest["files"]["cov.npy"]["sha256"],
        "rest":sha256(Path(plan["_root"])/plan["output_dir"]/"development_data/rest_manifest.json"),
        "train_ids":cohort.meta.iloc[train].trial_id.tolist(),
        "validation_ids":cohort.meta.iloc[val].trial_id.tolist(),
        "feature_source":sha256(source/"frontier_features.py")})
    path=Path(plan["_root"])/plan["dataset"]["cache_dir"]/"feature_fits"/(key+".joblib")
    if path.exists():
        return joblib.load(path)
    features=CovarianceFeatures(spec).fit(cohort.covariance_view(spec,train),y)
    result=(features,features.transform(cohort.covariance_view(spec,train)),
            features.transform(cohort.covariance_view(spec,val)))
    _atomic_joblib(result,path)
    return result


def _fit_one(plan,cohort,spec,train,val,directory,*,epochs=None,seeds=None,cache_features=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    y=cohort.meta.iloc[train].label.to_numpy(dtype=np.int64)
    if spec["family"]=="neural":
        X=cohort.neural_view(spec,train)
        val_X=cohort.neural_view(spec,val) if len(val) else None
        output=[];epoch_counts=[]
        for seed in seeds or [plan["search"]["seed"]]:
            path=directory/f"seed_{seed}"
            predictor=fit_neural(X,y,cohort.meta.iloc[train].subject.to_numpy(),spec,plan,path,
                                  epochs=epochs,seed=seed)
            epoch_counts.append(predictor.epochs)
            if val_X is not None:
                output.append(predictor.predict_proba(val_X))
        p=np.mean(output,axis=0) if output else None
        model_info={"type":"neural","seeds":seeds or [plan["search"]["seed"]],
                    "epochs":epoch_counts,"spec":spec,"train_subjects":sorted(cohort.meta.iloc[train].subject.unique().tolist())}
    elif spec["family"]=="raw_csp":
        from .accuracy_v2 import classical
        model=classical("csp_lda",{})
        model.fit(cohort.neural_view(spec,train).astype(np.float64),y)
        p=model.predict_proba(cohort.neural_view(spec,val).astype(np.float64)) if len(val) else None
        _atomic_joblib(model,directory/"model.joblib")
        model_info={"type":"raw_csp","spec":spec}
    else:
        if cache_features and spec["family"]!="mdm" and len(val):
            features,train_f,val_f=_feature_cache(plan,cohort,spec,train,val,y)
            fitted=classifier(spec).fit(train_f,y)
            model=CovarianceModel(spec,features,fitted)
            p=fitted.predict_proba(val_f)
        else:
            model=CovarianceModel(spec).fit(cohort.covariance_view(spec,train),y)
            p=model.predict_proba(cohort.covariance_view(spec,val)) if len(val) else None
        _atomic_joblib(model,directory/"model.joblib")
        model_info={"type":"covariance","spec":spec}
    model_info["files"]={str(p.relative_to(directory)):sha256(p) for p in directory.rglob("*")
                         if p.is_file() and p.suffix in {".joblib",".pt"}}
    model_info["train_subjects"]=sorted(cohort.meta.iloc[train].subject.unique().tolist())
    model_info["train_trial_ids_sha256"]=digest(cohort.meta.iloc[train].trial_id.tolist())
    json_write(directory/"model_info.json",model_info)
    return p,model_info


def predict_saved(cohort,spec,directory,rows):
    directory=Path(directory)
    info=json.loads((directory/"model_info.json").read_text())
    for filename,expected in info["files"].items():
        if sha256(directory/filename)!=expected:
            raise RuntimeError("Model artifact hash changed.")
    if info["type"]=="neural":
        X=cohort.neural_view(spec,rows)
        probabilities=[]
        for seed in info["seeds"]:
            predictor,_=load_neural(directory/f"seed_{seed}"/"model.pt")
            probabilities.append(predictor.predict_proba(X))
        return np.mean(probabilities,axis=0)
    model=joblib.load(directory/"model.joblib")
    X=cohort.neural_view(spec,rows).astype(np.float64) if info["type"]=="raw_csp" else cohort.covariance_view(spec,rows)
    return model.predict_proba(X)


def _screen_candidate(spec):
    plan,cohort=_WORKER["plan"],_WORKER["cohort"]
    out=Path(plan["_root"])/plan["output_dir"]
    directory=out/"screening"/spec["name"]
    complete=directory/"complete.json"
    if complete.exists():
        saved=json.loads(complete.read_text())
        if sha256(directory/"predictions.csv")!=saved["predictions_sha256"]:
            raise RuntimeError("Screening predictions changed.")
        return saved
    directory.mkdir(parents=True,exist_ok=True)
    try:
        started=time.perf_counter();frames=[];epochs=[]
        run_key=digest({k:v for k,v in plan.items() if not k.startswith("_")})[:16]
        cache=Path(plan["_root"])/plan["dataset"]["cache_dir"]/"screen_models"/run_key/spec["name"]
        with threadpool_limits(limits=plan["search"]["threads"]):
            for split in discovery_splits(plan):
                train=np.flatnonzero(cohort.meta.subject.isin(split["train_subjects"]))
                val=np.flatnonzero(cohort.meta.subject.isin(split["validation_subjects"]))
                print(f"Screen {spec['name']} fold {split['fold']+1}/{plan['search']['folds']}",flush=True)
                p,info=_fit_one(plan,cohort,spec,train,val,cache/f"fold_{split['fold']}",cache_features=True)
                if info["type"]=="neural":
                    epochs.extend(info["epochs"])
                frame=predictions(cohort.meta.iloc[val],p);frame["fold"]=split["fold"]
                frames.append(frame)
        frame=pd.concat(frames,ignore_index=True).sort_values("trial_id").reset_index(drop=True)
        if frame.trial_id.duplicated().any() or set(frame.subject)!=set(plan["discovery_subjects"]):
            raise AssertionError("Discovery OOF participant/trial identity mismatch.")
        p=np.column_stack([1-frame.prob_right,frame.prob_right])
        frame.to_csv(directory/"predictions.csv",index=False)
        result={"name":spec["name"],"arm":arm(spec),"family":spec["family"],"spec":spec,
                **score(frame,p),"refit_epochs":int(np.median(epochs)) if epochs else None,
                "seconds":time.perf_counter()-started,"predictions_sha256":sha256(directory/"predictions.csv"),
                "folds":discovery_splits(plan),"status":"complete"}
        json_write(complete,result)
        return result
    except Exception as error:
        failed={"name":spec["name"],"arm":arm(spec),"family":spec["family"],"spec":spec,
                "status":"failed","error":f"{type(error).__name__}: {error}","traceback":traceback.format_exc()}
        json_write(directory/"failed.json",failed)
        return failed


def screen(plan):
    out,identity=freeze(plan)
    data=prepare_cohort(plan,plan["discovery_subjects"]+plan["validation_subjects"],"development_data")
    prepare_rest(plan,data,"development_data")
    json_write(out/"discovery_splits.json",discovery_splits(plan))
    specs=sorted(candidates(),key=lambda s:(s["family"]=="neural",s["family"],s["name"]))
    results=[]
    with ProcessPoolExecutor(max_workers=plan["search"]["workers"],mp_context=get_context("spawn"),
                             initializer=_worker_init,initargs=(plan,)) as pool:
        jobs={pool.submit(_screen_candidate,s):s["name"] for s in specs}
        for future in as_completed(jobs):
            result=future.result();results.append(result)
            json_write(out/"screening_progress.json",{"finished":len(results),"total":len(specs),
                "failed":sum(r["status"]=="failed" for r in results),"last":result["name"],
                "protocol_sha256":identity["sha256"]})
            print(f"SCREEN {len(results)}/{len(specs)} {result['name']} {result['status']} "
                  f"BA={result.get('balanced_accuracy',float('nan')):.4f}",flush=True)
    flat=[{k:v for k,v in r.items() if k not in {"spec","folds","traceback"}} for r in results]
    pd.DataFrame(flat).sort_values(["arm","balanced_accuracy"],ascending=[True,False]).to_csv(out/"screening_summary.csv",index=False)
    failures=[r for r in results if r["status"]!="complete"]
    json_write(out/"screening_complete.json",{"tested":len(results),"failures":failures,"protocol_sha256":identity["sha256"]})
    if failures:
        raise RuntimeError(f"{len(failures)} candidates failed; retain errors and resolve before selecting.")


def _rank(rows):
    return sorted(rows,key=lambda r:(-r["balanced_accuracy"],r["log_loss"],r["name"]))


def _diverse(rows,count):
    chosen=[];families=set()
    for row in rows:
        family=row["family"]
        if family not in families:
            chosen.append(row["name"]);families.add(family)
        if len(chosen)==count:
            return chosen
    for row in rows:
        if row["name"] not in chosen:
            chosen.append(row["name"])
        if len(chosen)==count:
            break
    return chosen


def _logits(P):
    return np.log(np.clip(P,1e-5,1-1e-5)/np.clip(1-P,1e-5,1-1e-5))


def _ensemble_predictions(spec,P,meta=None):
    if spec["kind"]=="average":
        right=P.mean(axis=1)
    elif spec["kind"]=="weighted":
        right=P@np.asarray(spec["weights"])
    elif spec["kind"]=="stacking":
        right=meta.predict_proba(_logits(P))[:,1]
    else:
        raise ValueError(spec["kind"])
    return np.column_stack([1-right,right])


def select(plan):
    out,identity=freeze(plan)
    if (out/"selection.json").exists():
        print("Selection already frozen.",flush=True);return
    completed=json.loads((out/"screening_complete.json").read_text())
    if completed["failures"] or completed["tested"]!=len(candidates()):
        raise RuntimeError("Complete the entire specified candidate screen before selection.")
    cohort=load_rest(plan,load_cohort(plan,"development_data"),"development_data")
    records=[json.loads(p.read_text()) for p in (out/"screening").glob("*/complete.json")]
    shortlisted=[]
    for current_arm in plan["confirmation"]["arms"]:
        ranked=_rank([r for r in records if r["arm"]==current_arm])
        wanted={r["name"] for r in ranked[:plan["search"]["shortlist_global"]]}
        for family in sorted(set(r["family"] for r in ranked)):
            wanted.update(r["name"] for r in [r for r in ranked if r["family"]==family][:plan["search"]["shortlist_per_family"]])
        neural=[r for r in ranked if r["family"]=="neural" and r["name"]!="eegnet16_baseline"]
        if neural:wanted.add(neural[0]["name"])
        if current_arm=="zero_calibration":wanted.update(plan["confirmation"]["baselines"])
        shortlisted.extend(r for r in ranked if r["name"] in wanted)
    json_write(out/"shortlist.json",shortlisted)
    train=np.flatnonzero(cohort.meta.subject.isin(plan["discovery_subjects"]))
    val=np.flatnonzero(cohort.meta.subject.isin(plan["validation_subjects"]))
    val_meta=cohort.meta.iloc[val].reset_index(drop=True)
    torch.set_num_threads(plan["search"]["threads"])
    validation=[]
    with threadpool_limits(limits=plan["search"]["threads"]):
        for row in shortlisted:
            name,spec=row["name"],row["spec"]
            directory=out/"validation"/name
            result_path=directory/"complete.json"
            if result_path.exists():
                validation.append(json.loads(result_path.read_text()));continue
            seeds=(plan["neural"]["final_seeds"] if spec["family"]=="neural" and name!="eegnet16_baseline"
                   else [plan["search"]["seed"]])
            print(f"Validate shortlist {name}",flush=True)
            p,model_info=_fit_one(plan,cohort,spec,train,val,directory,epochs=row["refit_epochs"],seeds=seeds)
            predictions(val_meta,p).to_csv(directory/"predictions.csv",index=False)
            result={"name":name,"arm":arm(spec),"kind":"single","family":spec["family"],"spec":spec,
                    "refit_epochs":row["refit_epochs"],"seeds":seeds,**score(val_meta,p),
                    "predictions_sha256":sha256(directory/"predictions.csv")}
            json_write(result_path,result);validation.append(result)
    # Train every combination rule only on discovery OOF probabilities.
    discovery_meta=cohort.meta.iloc[train].sort_values("trial_id").reset_index(drop=True)
    y=discovery_meta.label.to_numpy(dtype=int)
    for current_arm in plan["confirmation"]["arms"]:
        base=_rank([r for r in shortlisted if r["arm"]==current_arm])
        pools=[]
        for count in plan["search"]["ensemble_top_counts"]:
            pools.append([r["name"] for r in base[:count]])
        pools.extend([_diverse(base,3),_diverse(base,5)])
        unique=[]
        for pool in pools:
            if len(pool)>1 and pool not in unique:unique.append(pool)
        ensemble_specs=[{"kind":"average","members":pool} for pool in unique]
        pool=_diverse(base,5)
        OOF=np.column_stack([pd.read_csv(out/"screening"/name/"predictions.csv").set_index("trial_id").loc[discovery_meta.trial_id].prob_right for name in pool])
        def objective(w):
            right=np.clip(OOF@w,1e-6,1-1e-6)
            return float(-np.mean(y*np.log(right)+(1-y)*np.log(1-right))+.005*np.sum(w*w))
        optimization=minimize(objective,np.full(len(pool),1/len(pool)),method="SLSQP",
            bounds=[(0,1)]*len(pool),constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
            options={"maxiter":500,"ftol":1e-12})
        if not optimization.success:
            raise RuntimeError("Discovery-only ensemble weight fit failed: "+optimization.message)
        ensemble_specs.append({"kind":"weighted","members":pool,"weights":optimization.x.tolist()})
        ensemble_specs.extend({"kind":"stacking","members":pool,"C":C} for C in plan["search"]["stacking_C"])
        for combination in ensemble_specs:
            combination["arm"]=current_arm
            name="ensemble_"+digest(combination)[:12]
            directory=out/"validation"/name;directory.mkdir(parents=True,exist_ok=True)
            members=combination["members"]
            val_P=np.column_stack([pd.read_csv(out/"validation"/member/"predictions.csv").prob_right for member in members])
            meta_model=None
            if combination["kind"]=="stacking":
                meta_model=LogisticRegression(C=combination["C"],max_iter=1000,random_state=plan["search"]["seed"])
                meta_model.fit(_logits(OOF),y)
                _atomic_joblib(meta_model,directory/"meta.joblib")
                combination["meta_sha256"]=sha256(directory/"meta.joblib")
            p=_ensemble_predictions(combination,val_P,meta_model)
            predictions(val_meta,p).to_csv(directory/"predictions.csv",index=False)
            result={"name":name,"arm":current_arm,"kind":"ensemble","family":"ensemble",
                    "combination":combination,**score(val_meta,p),
                    "predictions_sha256":sha256(directory/"predictions.csv")}
            json_write(directory/"complete.json",result);validation.append(result)
    pd.DataFrame([{k:v for k,v in r.items() if k not in {"spec","combination","seeds"}} for r in _rank(validation)]).to_csv(out/"validation_summary.csv",index=False)
    winners={current_arm:_rank([r for r in validation if r["arm"]==current_arm])[0]
             for current_arm in plan["confirmation"]["arms"]}
    selection={"protocol_sha256":identity["sha256"],"selected":winners,
               "confirmation_subjects":plan["confirmation_subjects"],"selection_used_confirmation_data":False,
               "validation_subjects":plan["validation_subjects"],"discovery_subjects":plan["discovery_subjects"],
               "tested_base_candidates":len(records),"tested_validation_candidates":len(validation)}
    json_write(out/"selection.json",selection)
    for current_arm,winner in winners.items():
        print("FROZEN WINNER",current_arm,winner["name"],winner["balanced_accuracy"],flush=True)


def _selected_members(out,selection):
    names=set()
    for winner in selection["selected"].values():
        names.update(winner["combination"]["members"] if winner["kind"]=="ensemble" else [winner["name"]])
    return names


def confirm(plan):
    out,identity=freeze(plan)
    selection_path=out/"selection.json"
    selection=json.loads(selection_path.read_text())
    if selection["protocol_sha256"]!=identity["sha256"] or selection["selection_used_confirmation_data"]:
        raise RuntimeError("Selection/protocol identity mismatch.")
    if (out/"confirmation/complete.json").exists():
        print("Confirmation already complete; run verify rather than selecting again.",flush=True);return
    selection_sha=sha256(selection_path)
    receipt={"selection_sha256":selection_sha,"protocol_sha256":identity["sha256"],
             "confirmation_subjects":plan["confirmation_subjects"],"selected":selection["selected"],
             "target_resting_calibration_only_in_secondary_arm":True}
    json_write(out/"confirmation_started.json",receipt)
    development=load_rest(plan,load_cohort(plan,"development_data"),"development_data")
    rows=np.arange(len(development.meta))
    names=_selected_members(out,selection)|set(plan["confirmation"]["baselines"])
    validation={p.parent.name:json.loads(p.read_text()) for p in (out/"validation").glob("*/complete.json")}
    torch.set_num_threads(plan["search"]["threads"])
    with threadpool_limits(limits=plan["search"]["threads"]):
        for name in sorted(names):
            directory=out/"final_models"/name
            if (directory/"model_info.json").exists():
                # Every constituent artifact is checked before inference below.
                continue
            spec=validation[name]["spec"]
            print("Final source-only refit",name,flush=True)
            _fit_one(plan,development,spec,rows,np.asarray([],dtype=int),directory,
                     epochs=validation[name]["refit_epochs"],seeds=validation[name]["seeds"])
    for name in names:
        directory=out/"final_models"/name
        info=json.loads((directory/"model_info.json").read_text())
        if set(info["train_subjects"])!=set(development.meta.subject):
            raise RuntimeError("Final fit uses different source participants.")
        for filename,expected in info["files"].items():
            if sha256(directory/filename)!=expected:
                raise RuntimeError("Final model is incomplete or changed; inspect before confirmation.")
    # All training, ensemble membership/weights and epochs are now fixed.
    json_write(out/"final_models_frozen.json",{"selection_sha256":selection_sha,
        "model_info_sha256":{name:sha256(out/"final_models"/name/"model_info.json") for name in names},
        "confirmation_acquisition_started_after_fitting":True})
    test=prepare_cohort(plan,plan["confirmation_subjects"],"confirmation_data")
    test=prepare_rest(plan,test,"confirmation_data")
    if set(test.meta.subject)&set(development.meta.subject):
        raise AssertionError("Final participant leakage.")
    all_rows=np.arange(len(test.meta));components={}
    with threadpool_limits(limits=plan["search"]["threads"]):
        for name in sorted(names):
            components[name]=predict_saved(test,validation[name]["spec"],out/"final_models"/name,all_rows)
    root=out/"confirmation";root.mkdir(exist_ok=True)
    evaluated={}
    for name in plan["confirmation"]["baselines"]:
        evaluated[name]={"arm":"zero_calibration","p":components[name],"role":"prespecified_baseline"}
    for current_arm,winner in selection["selected"].items():
        if winner["kind"]=="single":p=components[winner["name"]]
        else:
            combination=winner["combination"]
            meta=joblib.load(out/"validation"/winner["name"]/"meta.joblib") if combination["kind"]=="stacking" else None
            if meta is not None and sha256(out/"validation"/winner["name"]/"meta.joblib")!=combination["meta_sha256"]:
                raise RuntimeError("Frozen stacking coefficients changed.")
            P=np.column_stack([components[name][:,1] for name in combination["members"]])
            p=_ensemble_predictions(combination,P,meta)
        evaluated["selected_"+current_arm]={"arm":current_arm,"p":p,"role":"selected_on_development"}
    summaries=[];per_subject=[];frames=[]
    for name,item in evaluated.items():
        p=item["p"]
        frame=predictions(test.meta,p);frame["model"]=name;frame["arm"]=item["arm"];frames.append(frame)
        scores=subject_scores(test.meta.reset_index(drop=True),p)
        scores["model"]=name;scores["arm"]=item["arm"];per_subject.append(scores)
        lo,hi=bootstrap(scores.balanced_accuracy,plan)
        summaries.append({"model":name,"arm":item["arm"],"role":item["role"],"subjects":len(scores),
            **score(test.meta.reset_index(drop=True),p),"ci_low":lo,"ci_high":hi})
    pd.concat(frames,ignore_index=True).to_csv(root/"predictions.csv",index=False)
    pd.concat(per_subject,ignore_index=True).to_csv(root/"metrics_by_subject.csv",index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(root/"summary.csv",index=False)
    if sha256(selection_path)!=selection_sha:
        raise AssertionError("Development selection changed during final confirmation.")
    json_write(root/"complete.json",{"selection_sha256":selection_sha,"protocol_sha256":identity["sha256"],
        "models":list(evaluated),"participants":len(test.manifest["subjects"]),"trials":len(test.meta),
        "prediction_sha256":sha256(root/"predictions.csv"),"confirmation_used_for_selection":False})
    print(summary.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["all","screen","select","confirm","verify","report"])
    parser.add_argument("--plan",default="configs/frontier.yaml")
    args=parser.parse_args();plan=load_plan(args.plan)
    if args.command in {"all","screen"}:screen(plan)
    if args.command in {"all","select"}:select(plan)
    if args.command in {"all","confirm"}:confirm(plan)
    if args.command in {"all","verify"}:
        from .frontier_verify import verify
        verify(plan)
    if args.command in {"all","report"}:
        from .frontier_report import report
        report(plan)


if __name__=="__main__":
    from neuroedge.frontier import main as canonical_main
    canonical_main()
