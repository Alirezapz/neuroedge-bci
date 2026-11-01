"""Independent recomputation of frozen final predictions and screening metrics."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from .config import digest, json_write, sha256
from .frontier import _ensemble_predictions, _selected_members, predict_saved, discovery_splits
from .frontier_alignment import load_rest
from .frontier_data import load_cohort
from .frontier_features import candidates
from .frontier_neural import load_neural, trial_normalize
from .frontier_protocol import arm, bootstrap, freeze, score, subject_scores
from .models import TrainOnlyScaler


def verify(plan):
    out,identity=freeze(plan)
    selection=json.loads((out/"selection.json").read_text())
    receipt=json.loads((out/"confirmation_started.json").read_text())
    complete=json.loads((out/"confirmation/complete.json").read_text())
    frozen_models=json.loads((out/"final_models_frozen.json").read_text())
    for saved in [receipt,complete,frozen_models]:
        if saved["selection_sha256"]!=sha256(out/"selection.json"):
            raise AssertionError("Development selection changed.")
    if complete["protocol_sha256"]!=identity["sha256"]:
        raise AssertionError("Confirmation protocol changed.")
    prediction_file=out/"confirmation/predictions.csv"
    if sha256(prediction_file)!=complete["prediction_sha256"]:
        raise AssertionError("Final predictions changed.")
    screen_checked=[]
    for spec in candidates():
        directory=out/"screening"/spec["name"]
        result=json.loads((directory/"complete.json").read_text())
        if sha256(directory/"predictions.csv")!=result["predictions_sha256"]:
            raise AssertionError("Discovery predictions changed.")
        frame=pd.read_csv(directory/"predictions.csv")
        if frame.trial_id.duplicated().any() or set(frame.subject)!=set(plan["discovery_subjects"]):
            raise AssertionError("Discovery coverage mismatch.")
        for split in discovery_splits(plan):
            if set(frame.loc[frame.fold==split["fold"],"subject"])!=set(split["validation_subjects"]):
                raise AssertionError("Discovery OOF participant membership changed.")
        p=np.column_stack([1-frame.prob_right,frame.prob_right])
        recomputed=score(frame,p)
        for key,value in recomputed.items():
            np.testing.assert_allclose(value,result[key],atol=1e-12,rtol=0)
        screen_checked.append(spec["name"])
    development=load_rest(plan,load_cohort(plan,"development_data"),"development_data")
    test=load_rest(plan,load_cohort(plan,"confirmation_data"),"confirmation_data")
    if development.names!=test.names:
        raise AssertionError('Source and confirmation sensor order differs.')
    if set(development.meta.subject)&set(test.meta.subject):
        raise AssertionError("Source/target participant overlap.")
    validation={p.parent.name:json.loads(p.read_text()) for p in (out/"validation").glob("*/complete.json")}
    grid={spec['name']:spec for spec in candidates()}
    discovery_meta=development.meta[development.meta.subject.isin(plan['discovery_subjects'])].sort_values('trial_id')
    discovery_y=discovery_meta.label.to_numpy(dtype=int)
    meta_rules_checked=0
    for name,result in validation.items():
        frame=pd.read_csv(out/"validation"/name/"predictions.csv")
        if set(frame.subject)!=set(plan["validation_subjects"]) or frame.trial_id.duplicated().any():
            raise AssertionError("Secondary validation identity differs.")
        if sha256(out/"validation"/name/"predictions.csv")!=result["predictions_sha256"]:
            raise AssertionError("Secondary validation predictions changed.")
        for key,value in score(frame,np.column_stack([1-frame.prob_right,frame.prob_right])).items():
            np.testing.assert_allclose(value,result[key],atol=1e-12,rtol=0)
        if result['kind']=='single':
            if result['spec']!=grid[name]:
                raise AssertionError('Validated candidate differs from frozen grid.')
            screening=json.loads((out/'screening'/name/'complete.json').read_text())
            if result['refit_epochs']!=screening['refit_epochs']:
                raise AssertionError('Neural epoch policy changed after discovery.')
            expected_seeds=(plan['neural']['final_seeds'] if result['spec']['family']=='neural'
                            and name!='eegnet16_baseline' else [plan['search']['seed']])
            if result['seeds']!=expected_seeds:
                raise AssertionError('Validation seed policy differs.')
            info=json.loads((out/'validation'/name/'model_info.json').read_text())
            if set(info['train_subjects'])!=set(plan['discovery_subjects']):
                raise AssertionError('Secondary validation used nonsource training participants.')
        else:
            combination=result['combination']
            members=combination['members']
            if any(member not in grid or arm(grid[member])!=result['arm'] for member in members):
                raise AssertionError('Combination membership/operating condition differs.')
            val_P=np.column_stack([pd.read_csv(out/'validation'/member/'predictions.csv').prob_right for member in members])
            OOF=np.column_stack([pd.read_csv(out/'screening'/member/'predictions.csv').set_index('trial_id').loc[discovery_meta.trial_id].prob_right for member in members])
            meta=None
            if combination['kind']=='stacking':
                path=out/'validation'/name/'meta.joblib'
                if sha256(path)!=combination['meta_sha256']:
                    raise AssertionError('Source stacking coefficients changed.')
                meta=joblib.load(path)
                features=np.log(np.clip(OOF,1e-5,1-1e-5)/np.clip(1-OOF,1e-5,1-1e-5))
                refit=LogisticRegression(C=combination['C'],max_iter=1000,random_state=plan['search']['seed']).fit(features,discovery_y)
                np.testing.assert_allclose(meta.coef_,refit.coef_,atol=1e-7,rtol=1e-7)
                np.testing.assert_allclose(meta.intercept_,refit.intercept_,atol=1e-7,rtol=1e-7)
                meta_rules_checked+=1
            elif combination['kind']=='weighted':
                def objective(weights):
                    right=np.clip(OOF@weights,1e-6,1-1e-6)
                    return float(-np.mean(discovery_y*np.log(right)+(1-discovery_y)*np.log(1-right))+.005*np.sum(weights*weights))
                refit=minimize(objective,np.full(len(members),1/len(members)),method='SLSQP',
                    bounds=[(0,1)]*len(members),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                    options={'maxiter':500,'ftol':1e-12})
                assert refit.success
                np.testing.assert_allclose(combination['weights'],refit.x,atol=1e-6,rtol=1e-6)
                meta_rules_checked+=1
            expected_combination=_ensemble_predictions(combination,val_P,meta)
            np.testing.assert_allclose(frame.prob_right,expected_combination[:,1],atol=1e-12,rtol=0)
    for current_arm,winner in selection['selected'].items():
        ranked=sorted([r for r in validation.values() if r['arm']==current_arm],
                      key=lambda r:(-r['balanced_accuracy'],r['log_loss'],r['name']))
        if winner!=ranked[0]:
            raise AssertionError('Frozen selection is not the specified validation winner.')
    rows=np.arange(len(test.meta))
    final_names=_selected_members(out,selection)|set(plan["confirmation"]["baselines"])
    components={}
    torch.set_num_threads(plan["search"]["threads"])
    with threadpool_limits(limits=plan["search"]["threads"]):
        for name in sorted(final_names):
            directory=out/"final_models"/name
            if sha256(directory/"model_info.json")!=frozen_models["model_info_sha256"][name]:
                raise AssertionError("Final model receipt differs.")
            info=json.loads((directory/"model_info.json").read_text())
            if set(info["train_subjects"])!=set(plan["discovery_subjects"]+plan["validation_subjects"]):
                raise AssertionError("Final model training membership differs.")
            if info["train_trial_ids_sha256"]!=digest(development.meta.trial_id.tolist()):
                raise AssertionError("Final training trial identities differ.")
            spec=validation[name]["spec"]
            if info['spec']!=spec:
                raise AssertionError('Final candidate settings differ from validation.')
            if info['type']=='neural':
                if info['seeds']!=validation[name]['seeds'] or info['epochs']!=[validation[name]['refit_epochs']]*len(info['seeds']):
                    raise AssertionError('Final neural seed/epoch policy changed.')
            components[name]=predict_saved(test,spec,directory,rows)
            for i in [0,len(rows)//2,len(rows)-1]:
                individual=predict_saved(test,spec,directory,np.asarray([i]))
                np.testing.assert_allclose(individual,components[name][i:i+1],atol=2e-5,rtol=2e-5)
            if info["type"]=="neural":
                X=development.neural_view(spec,np.arange(len(development.meta)))
                expected=TrainOnlyScaler().fit(trial_normalize(X,spec))
                for seed in info["seeds"]:
                    predictor,cp=load_neural(directory/f"seed_{seed}"/"model.pt")
                    np.testing.assert_allclose(predictor.scaler.mean_,expected.mean_,atol=1e-7,rtol=0)
                    np.testing.assert_allclose(predictor.scaler.scale_,expected.scale_,atol=1e-7,rtol=0)
                    assert set(cp["train_subjects"])==set(info["train_subjects"])
    expected_predictions={name:components[name] for name in plan["confirmation"]["baselines"]}
    for current_arm,winner in selection["selected"].items():
        if winner["kind"]=="single":p=components[winner["name"]]
        else:
            combination=winner["combination"]
            meta_path=out/"validation"/winner["name"]/"meta.joblib"
            meta=joblib.load(meta_path) if combination["kind"]=="stacking" else None
            if meta is not None and sha256(meta_path)!=combination["meta_sha256"]:
                raise AssertionError("Frozen stacking model changed.")
            P=np.column_stack([components[name][:,1] for name in combination["members"]])
            p=_ensemble_predictions(combination,P,meta)
        expected_predictions["selected_"+current_arm]=p
    frame=pd.read_csv(prediction_file)
    saved_subject=pd.read_csv(out/"confirmation/metrics_by_subject.csv").set_index(["model","subject"])
    summary=pd.read_csv(out/"confirmation/summary.csv").set_index("model")
    if set(frame.model)!=set(expected_predictions):
        raise AssertionError("Unexpected evaluated final models.")
    for name,p in expected_predictions.items():
        saved=frame[frame.model==name].set_index("trial_id").loc[test.meta.trial_id]
        np.testing.assert_allclose(saved.prob_right.to_numpy(),p[:,1],atol=2e-5,rtol=2e-5)
        np.testing.assert_array_equal(saved.prediction.to_numpy(),p.argmax(1))
        np.testing.assert_array_equal(saved.label.to_numpy(),test.meta.label.to_numpy())
        scores=subject_scores(test.meta.reset_index(drop=True),p).set_index("subject")
        for key in ["balanced_accuracy","accuracy","log_loss"]:
            np.testing.assert_allclose(scores[key],saved_subject.loc[name,key],atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(scores[key].mean(),summary.loc[name,key],atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(bootstrap(scores.balanced_accuracy,plan),summary.loc[name,["ci_low","ci_high"]].to_numpy(dtype=float),atol=1e-12,rtol=0)
    result={"status":"PASS","screening_prediction_tables_recomputed":len(screen_checked),
            "validation_prediction_tables_recomputed":len(validation),"final_models_reinferred":sorted(final_names),
            "frozen_selections_unchanged":True,"source_target_participants_disjoint":True,
            "source_only_neural_scalers_recomputed":True,"individual_inference_matches_batch":True,
            "source_OOF_combination_rules_refitted_and_matched":meta_rules_checked,
            "specified_validation_winners_unchanged":True,
            "verifier_source_sha256":sha256(__file__),
            "secondary_calibration_scope":"R01 only; no task/test statistic or label used for alignment",
            "note":"Screening metrics verified from saved OOF predictions; final selected models and baselines independently reinferred from real EEG."}
    json_write(out/"verification.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result
