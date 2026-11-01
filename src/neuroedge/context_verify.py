"""Recompute metrics/rules and rerun the frozen transfer and nine LOSO fits."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from .config import digest,json_write,sha256
from .context_data import load_cohort
from .context_models import apply_temperature,fit_temperature
from .context_neural import inputs,load_neural
from .context_protocol import freeze,grid,interval,score,splits,subject_scores
from .context_study import (_combination,ensemble_probability,evaluate_pipelines,
                            members,predict_base,rank,reference_names,seed_policy)
from .models import TrainOnlyScaler


def _read_predictions(path):
    # Preserve serialized binary64 probabilities, especially near zero/one.
    # Default CSV parsing can change log loss despite an unchanged file hash.
    return pd.read_csv(path,float_precision='round_trip')


def verify_frame(frame,meta):
    if frame.trial_id.duplicated().any():raise AssertionError('Duplicate prediction trial.')
    lookup=meta.set_index('trial_id')
    if set(frame.trial_id)!=set(meta.trial_id):raise AssertionError('Prediction coverage differs.')
    expected=lookup.loc[frame.trial_id]
    for key in ['label','subject','session','run']:
        np.testing.assert_array_equal(frame[key].to_numpy(),expected[key].to_numpy())
    np.testing.assert_array_equal(frame.prediction.to_numpy(),(frame.prob_right.to_numpy()>.5).astype(int))


def compare_predictions(actual,expected):
    keys=['model','trial_id']
    a=actual.sort_values(keys).reset_index(drop=True);b=expected.sort_values(keys).reset_index(drop=True)
    if len(a)!=len(b):raise AssertionError('Final prediction count differs.')
    for key in keys+['label','subject','prediction','arm']:
        np.testing.assert_array_equal(a[key].to_numpy(),b[key].to_numpy())
    np.testing.assert_allclose(a.prob_right,b.prob_right,atol=2e-5,rtol=2e-5)


def check_fit(plan,cohort,rows,spec,directory,seeds,allow_missing_temporary=False):
    info=json.loads((directory/'model_info.json').read_text());meta=cohort.meta.iloc[rows]
    if info['spec']!=spec or info['seeds']!=seeds or info['train_subjects']!=sorted(set(meta.subject.tolist())):
        raise AssertionError('Fitted settings or source people differ.')
    if info['train_trial_ids_sha256']!=digest(meta.trial_id.tolist()):raise AssertionError('Source trial identities differ.')
    for filename,expected in info['files'].items():
        path=directory/filename
        if allow_missing_temporary and not path.exists() and path.suffix in {'.pt','.joblib'}:continue
        if sha256(path)!=expected:raise AssertionError('Fitted model/evidence checksum differs.')
    if spec['family']=='neural' and spec['architecture']!='filterbank_variance' and not allow_missing_temporary:
        A,B=inputs(cohort,rows,spec);expected=TrainOnlyScaler().fit(A)
        for seed in seeds:
            fitted,cp=load_neural(directory/f'seed_{seed}/model.pt')
            np.testing.assert_allclose(fitted.scaler.mean_,expected.mean_,atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(fitted.scaler.scale_,expected.scale_,atol=1e-6,rtol=1e-6)
    return info


def check_summary(plan,frame,directory):
    summary=pd.read_csv(directory/'summary.csv');per=pd.read_csv(directory/'metrics_by_subject.csv')
    for row in summary.itertuples():
        current=frame[frame.model==row.model].reset_index(drop=True)
        p=np.column_stack([1-current.prob_right,current.prob_right])
        expected=subject_scores(current,p)
        for key,value in score(current,p).items():np.testing.assert_allclose(value,getattr(row,key),atol=1e-12,rtol=0)
        low,high=interval(expected.balanced_accuracy,plan)
        np.testing.assert_allclose([row.ci_low,row.ci_high],[low,high],atol=1e-12,rtol=0)
        actual=per[per.model==row.model].sort_values('subject')
        np.testing.assert_array_equal(actual.subject,expected.sort_values('subject').subject)
        np.testing.assert_allclose(actual.balanced_accuracy,expected.sort_values('subject').balanced_accuracy,atol=1e-12,rtol=0)


def verify(plan):
    out,identity=freeze(plan);selection=json.loads((out/'selection.json').read_text());selection_sha=sha256(out/'selection.json')
    if selection['protocol_sha256']!=identity['sha256'] or selection['external_data_used_for_selection']:
        raise AssertionError('Selection identity differs.')
    development=load_cohort(plan,'development_data');external=load_cohort(plan,'external_data')
    if development.names!=external.names or set(development.meta.subject)&set(external.meta.subject):raise AssertionError('Sensor order or source/target separation differs.')
    source_meta=development.meta[development.meta.subject.isin(plan['source_subjects'])].sort_values('trial_id')
    selection_meta=development.meta[development.meta.subject.isin(plan['selection_subjects'])]
    specs={s['name']:s for s in grid()};discovery={};fold_receipts=0;temperature_differences=[]
    for name,spec in specs.items():
        directory=out/'screening'/name;result=json.loads((directory/'complete.json').read_text())
        if result['protocol_sha256']!=identity['sha256'] or sha256(directory/'predictions.csv')!=result['prediction_sha256']:
            raise AssertionError('Discovery identity changed.')
        frame=_read_predictions(directory/'predictions.csv');verify_frame(frame,source_meta)
        for key,value in score(frame,np.column_stack([1-frame.prob_right,frame.prob_right])).items():
            np.testing.assert_allclose(value,result[key],atol=1e-12,rtol=0)
        for split in splits(plan):
            if set(frame.loc[frame.fold==split['fold'],'subject'])!=set(split['validation_subjects']):raise AssertionError('Source OOF groups differ.')
            train=np.flatnonzero(development.meta.subject.isin(split['train_subjects']))
            check_fit(plan,development,train,spec,directory/'fits'/f"fold_{split['fold']}",[plan['search']['seed']],True);fold_receipts+=1
        temperature=1. if name in reference_names() else fit_temperature(frame.prob_right,frame.label)
        stored_temperature=result['source_oof_temperature']
        if name in reference_names() and stored_temperature!=1.:raise AssertionError('Reference temperature changed.')
        # CSV roundtrip can perturb a scalar optimizer by ~1e-7 even when the
        # objective/probabilities are unchanged. Verify the actual calibration,
        # its optimum objective, and the separately frozen selection value.
        recalibrated=apply_temperature(frame.prob_right,temperature)
        stored_calibrated=apply_temperature(frame.prob_right,stored_temperature)
        np.testing.assert_allclose(stored_calibrated,recalibrated,atol=2e-7,rtol=2e-7)
        y=frame.label.to_numpy(dtype=int)
        losses=[float(-np.log(np.clip(p[np.arange(len(p)),y],1e-12,1)).mean()) for p in [recalibrated,stored_calibrated]]
        np.testing.assert_allclose(losses[0],losses[1],atol=1e-10,rtol=0)
        temperature_differences.append(abs(temperature-stored_temperature))
        if result['temperature_training_trial_ids_sha256']!=digest(frame.trial_id.tolist()):raise AssertionError('Temperature calibration used different source trials.')
        discovery[name]=result
    validated={};combination_rules=0
    source_rows=np.flatnonzero(development.meta.subject.isin(plan['source_subjects']))
    for path in (out/'validation').glob('*/complete.json'):
        result=json.loads(path.read_text());name=result['name'];directory=path.parent
        if sha256(directory/'predictions.csv')!=result['prediction_sha256']:raise AssertionError('Validation prediction checksum differs.')
        frame=_read_predictions(directory/'predictions.csv');verify_frame(frame,selection_meta)
        for key,value in score(frame,np.column_stack([1-frame.prob_right,frame.prob_right])).items():
            np.testing.assert_allclose(value,result[key],atol=1e-12,rtol=0)
        if result['kind']=='single':
            if result['spec']!=specs[name] or result['temperature']!=discovery[name]['source_oof_temperature']:raise AssertionError('Source-calibrated single changed.')
            expected=apply_temperature(frame.raw_prob_right,result['temperature'])
            np.testing.assert_allclose(frame.prob_right,expected[:,1],atol=1e-12,rtol=0)
            check_fit(plan,development,source_rows,specs[name],directory,seed_policy(specs[name],plan),True)
        else:
            rule=result['combination'];names=rule['members']
            OOF=np.column_stack([pd.read_csv(out/'screening'/n/'predictions.csv').set_index('trial_id').loc[source_meta.trial_id].prob_right for n in names])
            rebuilt,refit=_combination(rule['kind'],names,rule['arm'],OOF,source_meta.label.to_numpy(),rule.get('C'))
            if rule['kind']=='weighted':np.testing.assert_allclose(rebuilt['weights'],rule['weights'],atol=1e-6,rtol=1e-6)
            meta=None
            if rule['kind']=='stacking':
                p=directory/'meta.joblib'
                if sha256(p)!=rule['meta_sha256']:raise AssertionError('Stacking coefficient file changed.')
                meta=joblib.load(p);np.testing.assert_allclose(meta.coef_,refit.coef_,atol=1e-8,rtol=1e-8)
                np.testing.assert_allclose(meta.intercept_,refit.intercept_,atol=1e-8,rtol=1e-8)
            P=np.column_stack([pd.read_csv(out/'validation'/n/'predictions.csv').raw_prob_right for n in names])
            np.testing.assert_allclose(frame.prob_right,ensemble_probability(rule,P,meta)[:,1],atol=1e-12,rtol=0)
            combination_rules+=1
        validated[name]=result
    for arm,winner in selection['selected'].items():
        if winner!=rank([r for r in validated.values() if r['arm']==arm])[0]:raise AssertionError('Winner differs from declared selection rule.')
    frozen=json.loads((out/'final_models_frozen.json').read_text())
    if frozen['selection_sha256']!=selection_sha or frozen['protocol_sha256']!=identity['sha256']:raise AssertionError('Final model freeze differs.')
    names=members(selection);all_rows=np.arange(len(external.meta));torch.set_num_threads(plan['search']['threads'])
    final_reinferences=0;loso_reinferences=0
    with threadpool_limits(limits=plan['search']['threads']):
        components={}
        for name in names:
            directory=out/'final_models'/name
            if sha256(directory/'model_info.json')!=frozen['model_info_sha256'][name]:raise AssertionError('Final model receipt changed.')
            check_fit(plan,development,np.arange(len(development.meta)),specs[name],directory,seed_policy(specs[name],plan))
            components[name]=predict_base(external,all_rows,specs[name],directory)
            for index in [0,len(all_rows)//2,len(all_rows)-1]:
                p=predict_base(external,np.asarray([index]),specs[name],directory)
                np.testing.assert_allclose(p,components[name][index:index+1],atol=2e-5,rtol=2e-5)
            final_reinferences+=1
        expected=evaluate_pipelines(plan,out,selection,external,all_rows,components)
        actual=_read_predictions(out/'external_transfer/predictions.csv');complete=json.loads((out/'external_transfer/complete.json').read_text())
        if complete['selection_sha256']!=selection_sha or complete['prediction_sha256']!=sha256(out/'external_transfer/predictions.csv'):raise AssertionError('Final predictions changed.')
        compare_predictions(actual,expected);check_summary(plan,actual,out/'external_transfer')
        folds=[]
        for person in sorted(external.meta.subject.unique()):
            directory=out/'external_loso'/'folds'/f'S{person:04d}'
            held=np.flatnonzero(external.meta.subject==person);train=np.flatnonzero(external.meta.subject!=person)
            complete=json.loads((directory/'complete.json').read_text())
            if complete['held_subject']!=person or set(complete['train_subjects'])!=set(external.meta.iloc[train].subject):raise AssertionError('LOSO people or session grouping differ.')
            components={}
            for name in names:
                current=directory/'models'/name
                check_fit(plan,external,train,specs[name],current,seed_policy(specs[name],plan))
                components[name]=predict_base(external,held,specs[name],current);loso_reinferences+=1
                for position in [0,len(held)//2,len(held)-1]:
                    individual=predict_base(external,np.asarray([held[position]]),specs[name],current)
                    np.testing.assert_allclose(individual,components[name][position:position+1],atol=2e-5,rtol=2e-5)
            frame=evaluate_pipelines(plan,out,selection,external,held,components)
            actual=_read_predictions(directory/'predictions.csv')
            if complete['prediction_sha256']!=sha256(directory/'predictions.csv'):raise AssertionError('LOSO predictions changed.')
            compare_predictions(actual,frame);folds.append(frame)
        expected=pd.concat(folds,ignore_index=True);actual=_read_predictions(out/'external_loso/predictions.csv')
        complete=json.loads((out/'external_loso/complete.json').read_text())
        if complete['selection_sha256']!=selection_sha or complete['prediction_sha256']!=sha256(out/'external_loso/predictions.csv'):raise AssertionError('LOSO aggregate changed.')
        compare_predictions(actual,expected);check_summary(plan,actual,out/'external_loso')
    result={'status':'PASS','discovery_prediction_tables_recomputed':len(discovery),'source_fold_training_receipts_checked':fold_receipts,
            'source_temperature_policies_recomputed':len(discovery),'learned_source_temperatures_refitted':len(discovery)-len(reference_names()),
            'maximum_reoptimized_temperature_difference':float(max(temperature_differences)),
            'validation_prediction_tables_recomputed':len(validated),
            'source_combination_rules_rebuilt':combination_rules,'final_source_constituents_reinferred':final_reinferences,
            'external_loso_constituent_fits_reinferred':loso_reinferences,'source_only_neural_scalers_recomputed':True,
            'individual_inference_matches_batch':True,'held_external_people_and_both_sessions_excluded_from_their_loso_fit':True,
            'external_hyperparameter_or_ensemble_selection':False,'selection_sha256':selection_sha,
            'verifier_source_sha256':sha256(__file__),
            'scope':'Discovery fitted-model receipts and saved OOF predictions checked; final transfer and all LOSO saved fits independently rerun on real EEG. Temporary discovery/validation binaries may be omitted from distribution.'}
    json_write(out/'verification.json',result);print(json.dumps(result,indent=2),flush=True);return result
