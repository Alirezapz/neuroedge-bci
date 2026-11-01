"""Predeclared supplementary personal calibration, session-disjoint test.

Only an intercept is adapted from the first 4/8/16 labeled session-T examples.
All session-E labels remain out of that fit. Source algorithms are inherited
from context-study selection, never chosen with the external scores.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
import yaml

from .config import digest,json_write,sha256
from .context_protocol import binary_probabilities,interval,predictions,score,subject_scores


def fit_intercept(probabilities,labels,config):
    p=np.clip(np.asarray(probabilities),config['probability_clip'],1-config['probability_clip'])
    y=np.asarray(labels,dtype=int)
    if set(y)!={0,1}:return 0.,'single_class_support_keep_source'
    z=logit(p);variance=config['prior_standard_deviation']**2
    def objective(b):
        # Balanced pseudo-likelihood; fixed Gaussian intercept prior.
        loss=.5*np.mean(np.logaddexp(0,z[y==0]+b))+.5*np.mean(np.logaddexp(0,-z[y==1]-b))
        return float(loss+b*b/(2*len(y)*variance))
    fit=minimize_scalar(objective,bounds=config['bounds'],method='bounded',options={'xatol':1e-10})
    if not fit.success:raise RuntimeError('Personal intercept fit failed.')
    return float(fit.x),'fitted_from_first_session_support_only'


def adapt(probabilities,intercept,config):
    right=expit(logit(np.clip(np.asarray(probabilities),config['probability_clip'],1-config['probability_clip']))+intercept)
    return binary_probabilities(np.column_stack([1-right,right]))


def load_plan(path):
    path=Path(path).resolve();p=yaml.safe_load(path.read_text());p['_root']=str(path.parent.parent)
    if p['calibration_session']==p['test_session'] or p['primary_budget'] not in p['support_budgets']:raise ValueError('Invalid session-disjoint calibration protocol.')
    return p


def register(plan):
    root=Path(plan['_root']);out=root/plan['output_dir'];parent=root/plan['parent_output'];out.mkdir(parents=True,exist_ok=True)
    identity={'plan':{k:v for k,v in plan.items() if not k.startswith('_')},'adaptation_source_sha256':sha256(__file__),
              'parent_protocol_sha256':json.loads((parent/'protocol.json').read_text())['sha256']}
    identity['sha256']=digest(identity);path=out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())['sha256']!=identity['sha256']:raise RuntimeError('Frozen personal-calibration methods changed; use a new output.')
    else:
        parent_plan=yaml.safe_load((root/plan['parent_plan']).read_text())
        if list((root/parent_plan['external']['raw_dir']).glob('*.mat')) or (parent/'external_data/manifest.json').exists():
            raise RuntimeError('Register this supplementary protocol before external acquisition; do not call a retrospective analysis prospective.')
        json_write(path,{**identity,'registered_utc':datetime.now(timezone.utc).isoformat(),'registered_before_external_acquisition':True})
    return out,identity


def bind_selection(plan):
    out,identity=register(plan);root=Path(plan['_root']);parent=root/plan['parent_output']
    selection=parent/'selection.json'
    if not selection.exists():raise FileNotFoundError('Parent source-only selection is not complete yet.')
    value={'parent_selection_sha256':sha256(selection),'parent_protocol_sha256':identity['parent_protocol_sha256']}
    path=out/'parent_selection.json'
    if path.exists():
        if any(json.loads(path.read_text())[k]!=v for k,v in value.items()):raise RuntimeError('Bound source selection changed.')
    else:
        if (parent/'external_data/manifest.json').exists() or (parent/'external_transfer/complete.json').exists():
            raise RuntimeError('Bind the source selection before external data preparation.')
        json_write(path,{**value,'bound_utc':datetime.now(timezone.utc).isoformat()})
    return out,identity


def experiment(plan):
    out,identity=bind_selection(plan);root=Path(plan['_root']);parent=root/plan['parent_output']
    if json.loads((parent/'verification.json').read_text())['status']!='PASS':raise RuntimeError('Verify the parent real-EEG predictions first.')
    transfer=parent/'external_transfer/predictions.csv';original=pd.read_csv(transfer)
    frames=[];receipts=[]
    for (model,person),group in original.groupby(['model','subject']):
        support_pool=group[group.session==plan['calibration_session']].sort_values(['run','onset','trial_id'])
        test=group[group.session==plan['test_session']].reset_index(drop=True)
        if len(test)!=144 or len(support_pool)<max(plan['support_budgets']):raise AssertionError('Unexpected calibration/test cohort.')
        for budget in [0]+plan['support_budgets']:
            support=support_pool.iloc[:budget];b,status=(0.,'no_personal_calibration') if not budget else fit_intercept(support.prob_right,support.label,plan['adaptation'])
            p=adapt(test.prob_right,b,plan['adaptation']);frame=predictions(test,p)
            # Preserve actual unadapted probabilities at budget 0 (no clipping).
            if budget==0:
                frame['prob_right']=test.prob_right.to_numpy();frame['prediction']=test.prediction.to_numpy()
            frame['model']=model;frame['support_budget']=budget;frame['personal_intercept']=b;frames.append(frame)
            receipts.append({'model':model,'subject':int(person),'support_budget':budget,'intercept':b,'status':status,
                             'support_trial_ids':support.trial_id.tolist(),'test_trial_ids_sha256':digest(test.trial_id.tolist()),
                             'support_session':plan['calibration_session'],'test_session':plan['test_session']})
    all_frames=pd.concat(frames,ignore_index=True);all_frames.to_csv(out/'predictions.csv',index=False)
    json_write(out/'adaptation_receipts.json',receipts);summaries=[];per=[]
    for (model,budget),frame in all_frames.groupby(['model','support_budget']):
        p=np.column_stack([1-frame.prob_right,frame.prob_right]);scores=subject_scores(frame.reset_index(drop=True),p)
        low,high=interval(scores.balanced_accuracy,{'evaluation':{'bootstrap_seed':plan['bootstrap_seed'],'bootstrap_iterations':plan['bootstrap_iterations']}})
        summaries.append({'model':model,'support_budget':int(budget),'subjects':len(scores),**score(frame.reset_index(drop=True),p),'ci_low':low,'ci_high':high})
        scores['model']=model;scores['support_budget']=budget;per.append(scores)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False);pd.concat(per,ignore_index=True).to_csv(out/'metrics_by_subject.csv',index=False)
    json_write(out/'complete.json',{'protocol_sha256':identity['sha256'],'parent_transfer_prediction_sha256':sha256(transfer),
                'prediction_sha256':sha256(out/'predictions.csv'),'adaptation_receipts_sha256':sha256(out/'adaptation_receipts.json'),
                'external_method_or_budget_selection':False,'primary_budget':plan['primary_budget']})
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def verify(plan):
    out,identity=bind_selection(plan);root=Path(plan['_root']);parent=root/plan['parent_output']
    complete=json.loads((out/'complete.json').read_text());transfer=parent/'external_transfer/predictions.csv'
    if complete['protocol_sha256']!=identity['sha256'] or complete['parent_transfer_prediction_sha256']!=sha256(transfer):raise AssertionError('Parent or adaptation protocol changed.')
    if sha256(out/'predictions.csv')!=complete['prediction_sha256'] or sha256(out/'adaptation_receipts.json')!=complete['adaptation_receipts_sha256']:
        raise AssertionError('Personal adaptation predictions or receipts changed.')
    original=pd.read_csv(transfer);stored=pd.read_csv(out/'predictions.csv');receipts=json.loads((out/'adaptation_receipts.json').read_text())
    for r in receipts:
        group=original[(original.model==r['model'])&(original.subject==r['subject'])]
        support=group[group.session==plan['calibration_session']].sort_values(['run','onset','trial_id']).iloc[:r['support_budget']]
        test=group[group.session==plan['test_session']].reset_index(drop=True)
        if support.trial_id.tolist()!=r['support_trial_ids'] or set(support.trial_id)&set(test.trial_id) or digest(test.trial_id.tolist())!=r['test_trial_ids_sha256']:
            raise AssertionError('Calibration/test chronology or session separation differs.')
        b=0. if not r['support_budget'] else fit_intercept(support.prob_right,support.label,plan['adaptation'])[0]
        np.testing.assert_allclose(b,r['intercept'],atol=1e-10,rtol=1e-10)
        actual=stored[(stored.model==r['model'])&(stored.subject==r['subject'])&(stored.support_budget==r['support_budget'])].set_index('trial_id').loc[test.trial_id]
        expected=test.prob_right.to_numpy() if not r['support_budget'] else adapt(test.prob_right,b,plan['adaptation'])[:,1]
        np.testing.assert_allclose(actual.prob_right,expected,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(actual.prediction.to_numpy(),(expected>.5).astype(int))
    result={'status':'PASS','adaptation_receipts_recomputed':len(receipts),'support_labels_from_first_session_only':True,
            'test_labels_used_for_fitting':False,'external_budget_or_method_selection':False,'verifier_source_sha256':sha256(__file__)}
    json_write(out/'verification.json',result);print(json.dumps(result,indent=2),flush=True);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['register','bind-wait','run','verify'])
    parser.add_argument('--plan',default='configs/personalization.yaml');args=parser.parse_args();plan=load_plan(args.plan)
    if args.command=='register':print(register(plan)[0])
    elif args.command=='bind-wait':
        register(plan)
        while not (Path(plan['_root'])/plan['parent_output']/'selection.json').exists():time.sleep(2)
        print(bind_selection(plan)[0],flush=True)
    elif args.command=='run':experiment(plan);verify(plan)
    else:verify(plan)


if __name__=='__main__':main()
