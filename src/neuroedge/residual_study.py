"""Source-selected supplementary personal kernels; every test session stays out."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import digest,json_write,sha256
from .context_protocol import interval,now,predictions,score,subject_scores
from .context_data import load_cohort
from .context_protocol import load_plan as context_plan
from .personal_kernels import features,fit_scaler,transform,subset,kernel,fit_residual,predict_residual
from .personalization import fit_intercept,adapt


def load_plan(path):
    path=Path(path).resolve();p=yaml.safe_load(path.read_text());p['_root']=str(path.parent.parent)
    p['_parent']=context_plan(path.parent.parent/p['parent_plan']);return p


def register(plan):
    root=Path(plan['_root']);out=root/plan['output_dir'];parent=root/plan['parent_output']
    specification=json.loads((out/'specification.json').read_text())
    clean={k:v for k,v in plan.items() if not k.startswith('_')}
    if specification['plan']!=clean or specification['parent_selection_sha256']!=sha256(parent/'selection.json'):
        raise RuntimeError('Registered residual specification or source choice changed.')
    identity={'specification_sha256':specification['sha256'],
        'scientific_source_sha256':{n:sha256(root/'src/neuroedge'/n) for n in ['personal_kernels.py','residual_study.py','personalization.py']}}
    identity['sha256']=digest(identity);path=out/'implementation.json'
    if path.exists():
        if json.loads(path.read_text())['sha256']!=identity['sha256']:raise RuntimeError('Residual implementation changed; preserve the previous output.')
    else:
        if (parent/'external_transfer/complete.json').exists():raise RuntimeError('Freeze this implementation before the first external transfer scores.')
        json_write(path,{**identity,'frozen_utc':now(),'frozen_before_parent_external_transfer_scoring':True,
            'external_raw_already_acquired':bool(list((root/plan['_parent']['external']['raw_dir']).glob('*.mat')))})
    return out,identity


def bias_config(plan):
    a=plan['adaptation'];return {'probability_clip':a['probability_clip'],
        'prior_standard_deviation':a['intercept_prior_standard_deviation'],'bounds':[-16.,16.]}


def predict_groups(plan,base,values,kind,C,budget,source=False):
    frames=[];receipts=[]
    for person,group in base.groupby('subject',sort=True):
        pool=(group[group.run.isin(plan['source_calibration_runs'])] if source else group[group.session==plan['calibration_session']])
        support=pool.sort_values(['run','onset','trial_id']).iloc[:budget]
        test=group[group.run==plan['source_test_run']] if source else group[group.session==plan['test_session']]
        if len(support)!=budget or not len(test) or set(support.trial_id)&set(test.trial_id):raise AssertionError('Support/test coverage or separation differs.')
        si=support.index.to_numpy();ti=test.index.to_numpy()
        if not budget:p=test.prob_right.to_numpy();fit={'alpha':[],'intercept':0.,'status':'no_personal_calibration'}
        elif kind=='bias':
            b,status=fit_intercept(support.prob_right,support.label,bias_config(plan))
            p=adapt(test.prob_right,b,bias_config(plan))[:,1];fit={'alpha':[],'intercept':b,'status':status}
        else:
            S=subset(values,si);T=subset(values,ti)
            fit=fit_residual(kernel(S,S,kind),support.prob_right,support.label,C,plan['adaptation'])
            p=predict_residual(kernel(T,S,kind),test.prob_right,fit,plan['adaptation'])
        frame=predictions(test.reset_index(drop=True),np.column_stack([1-p,p]));frame['model']=kind;frame['support_budget']=budget;frames.append(frame)
        receipts.append({'subject':int(person),'model':kind,'support_budget':budget,'C':C,
            'support_trial_ids':support.trial_id.tolist(),'test_trial_ids_sha256':digest(test.trial_id.tolist()),
            'alpha':np.asarray(fit['alpha']).tolist(),'intercept':float(fit['intercept']),'status':fit['status']})
    return pd.concat(frames,ignore_index=True),receipts


def base_frame(plan,cohort,source):
    root=Path(plan['_root']);parent=root/plan['parent_output']
    if source:
        selection=json.loads((parent/'selection.json').read_text());name=selection['selected']['zero_calibration']['name']
        frame=pd.read_csv(parent/'validation'/name/'predictions.csv')
    else:
        frame=pd.read_csv(parent/'external_transfer/predictions.csv');frame=frame[frame.model==plan['base_pipeline']]
    frame=frame.reset_index(drop=True);lookup=pd.Series(np.arange(len(cohort.meta)),index=cohort.meta.trial_id)
    rows=lookup.loc[frame.trial_id].to_numpy()
    expected=cohort.meta.iloc[rows]
    for key in ['subject','label','run','session']:np.testing.assert_array_equal(frame[key],expected[key].to_numpy())
    return frame,rows


def source(plan):
    out,identity=register(plan);root=Path(plan['_root']);parent=root/plan['parent_output']
    if (out/'source_selection.json').exists():
        d=json.loads((out/'source_selection.json').read_text())
        if d['implementation_sha256']!=identity['sha256'] or d['final_scaler_sha256']!=sha256(out/'final_source_scaler.npz'):raise RuntimeError('Frozen residual selection/scaler changed.')
        return d
    if (parent/'external_transfer/complete.json').exists():raise RuntimeError('Finish source-only residual selection before first external scores.')
    cohort=load_cohort(plan['_parent'],'development_data');values=features(cohort,np.arange(len(cohort.meta)))
    source_rows=np.flatnonzero(cohort.meta.subject.isin(plan['_parent']['source_subjects']))
    development_scaler=fit_scaler(subset(values,source_rows));base,rows=base_frame(plan,cohort,True)
    if set(base.subject)!=set(plan['_parent']['selection_subjects']):raise AssertionError('Residual source-selection cohort changed.')
    calibrated=transform(subset(values,rows),development_scaler);results=[]
    for kind in plan['kernels']:
        for C in plan['C_grid']:
            frame,receipts=predict_groups(plan,base,calibrated,kind,C,plan['source_selection_budget'],True)
            directory=out/'source_candidates'/f'{kind}_C{C:g}';directory.mkdir(parents=True,exist_ok=True)
            frame.to_csv(directory/'predictions.csv',index=False);json_write(directory/'receipts.json',receipts)
            p=np.column_stack([1-frame.prob_right,frame.prob_right])
            result={'model':kind,'C':float(C),**score(frame,p),'prediction_sha256':sha256(directory/'predictions.csv')}
            results.append(result);print('Residual source selection',kind,C,result['balanced_accuracy'],flush=True)
    pd.DataFrame(results).to_csv(out/'source_summary.csv',index=False)
    chosen={kind:sorted([r for r in results if r['model']==kind],key=lambda r:(-r['balanced_accuracy'],r['log_loss'],r['C']))[0] for kind in plan['kernels']}
    final_scaler=fit_scaler(values);np.savez_compressed(out/'final_source_scaler.npz',**final_scaler)
    np.savez_compressed(out/'development_source_scaler.npz',**development_scaler)
    frozen={'implementation_sha256':identity['sha256'],'selected':chosen,'primary_kernel':plan['primary_kernel'],
        'primary_budget':plan['primary_budget'],'final_scaler_sha256':sha256(out/'final_source_scaler.npz'),
        'development_scaler_sha256':sha256(out/'development_source_scaler.npz'),
        'development_scaler_train_subjects':sorted(set(cohort.meta.iloc[source_rows].subject.tolist())),
        'development_scaler_train_trial_ids_sha256':digest(cohort.meta.iloc[source_rows].trial_id.tolist()),
        'final_scaler_train_subjects':sorted(set(cohort.meta.subject.tolist())),
        'final_scaler_train_trial_ids_sha256':digest(cohort.meta.trial_id.tolist()),
        'selection_frozen_utc':now(),'external_used_for_selection':False,
        'frozen_before_parent_external_transfer_scoring':not (parent/'external_transfer/complete.json').exists()}
    if not frozen['frozen_before_parent_external_transfer_scoring']:raise RuntimeError('External scoring raced source selection; do not call this a prospectively frozen selection.')
    json_write(out/'source_selection.json',frozen);return frozen


def experiment(plan):
    out,identity=register(plan);root=Path(plan['_root']);parent=root/plan['parent_output'];chosen=source(plan)
    if json.loads((parent/'verification.json').read_text())['status']!='PASS':raise RuntimeError('Independently verify the parent external predictions first.')
    cohort=load_cohort(plan['_parent'],'external_data');base,rows=base_frame(plan,cohort,False)
    with np.load(out/'final_source_scaler.npz') as f:scaler={k:f[k] for k in f.files}
    values=transform(features(cohort,rows),scaler);frames=[];receipts=[]
    for kind in ['bias']+plan['kernels']:
        for budget in [0]+plan['support_budgets']:
            C=None if kind=='bias' else chosen['selected'][kind]['C']
            frame,current=predict_groups(plan,base,values,kind,C,budget);frames.append(frame);receipts.extend(current)
    all_frames=pd.concat(frames,ignore_index=True);all_frames.to_csv(out/'predictions.csv',index=False)
    json_write(out/'adaptation_receipts.json',receipts);per=[];summaries=[]
    bootstrap={'evaluation':{'bootstrap_iterations':plan['bootstrap_iterations'],'bootstrap_seed':plan['bootstrap_seed']}}
    for (model,budget),frame in all_frames.groupby(['model','support_budget']):
        p=np.column_stack([1-frame.prob_right,frame.prob_right]);scores=subject_scores(frame.reset_index(drop=True),p)
        lo,hi=interval(scores.balanced_accuracy,bootstrap);scores['model']=model;scores['support_budget']=budget;per.append(scores)
        summaries.append({'model':model,'support_budget':int(budget),'subjects':len(scores),**score(frame.reset_index(drop=True),p),'ci_low':lo,'ci_high':hi})
    pd.concat(per,ignore_index=True).to_csv(out/'metrics_by_subject.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    json_write(out/'complete.json',{'implementation_sha256':identity['sha256'],'source_selection_sha256':sha256(out/'source_selection.json'),
        'parent_transfer_prediction_sha256':sha256(parent/'external_transfer/predictions.csv'),
        'prediction_sha256':sha256(out/'predictions.csv'),'receipt_sha256':sha256(out/'adaptation_receipts.json'),
        'external_used_for_method_or_budget_selection':False})
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['register','source','run'])
    parser.add_argument('--plan',default='configs/residual_personal.yaml');args=parser.parse_args();p=load_plan(args.plan)
    {'register':register,'source':source,'run':experiment}[args.command](p)


if __name__=='__main__':main()
