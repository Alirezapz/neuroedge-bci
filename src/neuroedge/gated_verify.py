"""Independent reinference and frozen-choice checks for the gate follow-up."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from .config import digest,json_write,sha256
from .context_data import load_cohort
from .context_models import apply_temperature,fit_temperature
from .context_protocol import predictions,score,splits
from .context_verify import check_fit,check_summary,compare_predictions,verify_frame
from .gated_neural import predict_saved
from .gated_study import freeze,specs


def verify(plan):
    out,identity=freeze(plan);parent=plan['_parent']
    development=load_cohort(parent,'development_data');external=load_cohort(parent,'external_data')
    selection=json.loads((out/'selection.json').read_text());frozen=json.loads((out/'final_models_frozen.json').read_text())
    if selection['protocol_sha256']!=identity['sha256'] or frozen['selection_sha256']!=sha256(out/'selection.json'):raise AssertionError('Gated frozen selection changed.')
    source_meta=development.meta[development.meta.subject.isin(parent['source_subjects'])]
    val_meta=development.meta[development.meta.subject.isin(parent['selection_subjects'])]
    source_rows=np.flatnonzero(development.meta.subject.isin(parent['source_subjects']))
    val_rows=np.flatnonzero(development.meta.subject.isin(parent['selection_subjects']))
    torch.set_num_threads(parent['search']['threads']);validated=[];transfer=[];checked_source=0
    with threadpool_limits(limits=parent['search']['threads']):
        for spec in specs(plan):
            variant=spec['variant'];directory=out/'candidates'/variant
            complete=json.loads((directory/'source_complete.json').read_text());validated_result=json.loads((directory/'validation_metrics.json').read_text())
            if complete['protocol_sha256']!=identity['sha256'] or complete['validation']!=validated_result:raise AssertionError('Gated source result changed.')
            if sha256(directory/'source_predictions.csv')!=validated_result['source_prediction_sha256'] or sha256(directory/'validation_predictions.csv')!=validated_result['validation_prediction_sha256']:
                raise AssertionError('Gated development predictions changed.')
            actual=pd.read_csv(directory/'source_predictions.csv');verify_frame(actual,source_meta);frames=[]
            for split in splits(parent):
                train=np.flatnonzero(development.meta.subject.isin(split['train_subjects']));held=np.flatnonzero(development.meta.subject.isin(split['validation_subjects']))
                current=directory/'source_folds'/f"fold_{split['fold']}";check_fit(parent,development,train,spec,current,[7])
                p=predict_saved(development,held,spec,current);f=predictions(development.meta.iloc[held].reset_index(drop=True),p)
                f['model']='source_fixture_key';frames.append(f);checked_source+=1
            expected=pd.concat(frames,ignore_index=True);actual['model']='source_fixture_key';actual['arm']='pre_cue';expected['arm']='pre_cue'
            compare_predictions(actual,expected)
            metrics=json.loads((directory/'source_metrics.json').read_text())
            for k,v in score(actual,np.column_stack([1-actual.prob_right,actual.prob_right])).items():np.testing.assert_allclose(v,metrics[k],atol=1e-12,rtol=0)
            T=fit_temperature(actual.prob_right,actual.label);stored=validated_result['temperature']
            np.testing.assert_allclose(apply_temperature(actual.prob_right,T),apply_temperature(actual.prob_right,stored),atol=2e-7,rtol=2e-7)
            check_fit(parent,development,source_rows,spec,directory/'validation_models',plan['seeds'])
            raw=predict_saved(development,val_rows,spec,directory/'validation_models');p=apply_temperature(raw[:,1],stored)
            expected=predictions(val_meta.reset_index(drop=True),p);expected['model']='validation_key';expected['arm']='pre_cue'
            actual=pd.read_csv(directory/'validation_predictions.csv');verify_frame(actual,val_meta);actual['model']='validation_key';actual['arm']='pre_cue'
            compare_predictions(actual,expected)
            # Reinference is checked above with the declared float32 tolerance.
            # Archived metrics must match the archived CSV probabilities exactly;
            # a fresh CPU reinference need not reproduce log loss to 12 decimals.
            saved_p=np.column_stack([1-actual.prob_right,actual.prob_right])
            for k,v in score(actual,saved_p).items():np.testing.assert_allclose(v,validated_result[k],atol=1e-12,rtol=0)
            validated.append(validated_result)
            final=out/'final_models'/variant
            if sha256(final/'model_info.json')!=frozen['model_info_sha256'][variant] or sha256(final/'model_info.json')!=complete['final_model_info_sha256']:
                raise AssertionError('Gated final model receipt changed.')
            check_fit(parent,development,np.arange(len(development.meta)),spec,final,plan['seeds'])
            rows=np.arange(len(external.meta));raw=predict_saved(external,rows,spec,final)
            for index in [0,len(rows)//2,len(rows)-1]:
                np.testing.assert_allclose(predict_saved(external,np.asarray([index]),spec,final),raw[index:index+1],atol=2e-5,rtol=2e-5)
            p=apply_temperature(raw[:,1],stored);f=predictions(external.meta,p);f['model']=variant;f['arm']='pre_cue';transfer.append(f)
        winner=sorted(validated,key=lambda r:(-r['balanced_accuracy'],r['log_loss'],r['name']))[0]
        if selection['selected']!=winner:raise AssertionError('Gated winner changed after source validation.')
        expected=pd.concat(transfer,ignore_index=True);actual=pd.read_csv(out/'external_transfer/predictions.csv')
        complete=json.loads((out/'external_transfer/complete.json').read_text())
        if complete['prediction_sha256']!=sha256(out/'external_transfer/predictions.csv') or complete['selection_sha256']!=sha256(out/'selection.json'):raise AssertionError('Gated external predictions changed.')
        compare_predictions(actual,expected)
        bootstrap={'evaluation':{'bootstrap_seed':plan['bootstrap_seed'],'bootstrap_iterations':plan['bootstrap_iterations']}}
        check_summary(bootstrap,actual,out/'external_transfer')
        spec=winner['spec'];frames=[];checked_loso=0
        for person in sorted(external.meta.subject.unique()):
            directory=out/'external_loso'/'folds'/f'S{person:04d}';train=np.flatnonzero(external.meta.subject!=person);held=np.flatnonzero(external.meta.subject==person)
            check_fit(parent,external,train,spec,directory/'model',plan['seeds']);raw=predict_saved(external,held,spec,directory/'model')
            for position in [0,len(held)//2,len(held)-1]:
                np.testing.assert_allclose(predict_saved(external,np.asarray([held[position]]),spec,directory/'model'),raw[position:position+1],atol=2e-5,rtol=2e-5)
            p=apply_temperature(raw[:,1],winner['temperature']);f=predictions(external.meta.iloc[held].reset_index(drop=True),p);f['model']='selected_fusion';f['arm']='pre_cue'
            actual=pd.read_csv(directory/'predictions.csv');complete=json.loads((directory/'complete.json').read_text())
            if complete['held_subject']!=person or complete['prediction_sha256']!=sha256(directory/'predictions.csv'):raise AssertionError('Gated LOSO identity changed.')
            compare_predictions(actual,f);frames.append(f);checked_loso+=1
        expected=pd.concat(frames,ignore_index=True);actual=pd.read_csv(out/'external_loso/predictions.csv')
        complete=json.loads((out/'external_loso/complete.json').read_text())
        if complete['prediction_sha256']!=sha256(out/'external_loso/predictions.csv'):raise AssertionError('Gated LOSO aggregate changed.')
        compare_predictions(actual,expected);check_summary(bootstrap,actual,out/'external_loso')
    result={'status':'PASS','source_group_fits_reinferred':checked_source,'validation_and_final_variants_reinferred':len(validated),
            'external_loso_fits_reinferred':checked_loso,'individual_batch_agreement':True,'all_source_temperature_and_selection_rules_checked':True,
            'external_used_for_selection':False,'verifier_source_sha256':sha256(__file__)}
    json_write(out/'verification.json',result);print(json.dumps(result,indent=2),flush=True);return result
