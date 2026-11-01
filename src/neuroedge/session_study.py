"""Frozen exploratory session-disjoint calibration with donor-only selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,log_loss,f1_score
from threadpoolctl import threadpool_limits
import yaml

from .config import digest,json_write,sha256
from .context_data import load_cohort
from .context_protocol import load_plan as context_plan
from .session_models import BASE,TUNED,METHODS,SessionModel


def load_plan(path):
    path=Path(path).resolve();p=yaml.safe_load(path.read_text());p['_root']=str(path.parent.parent)
    if p['calibration_session']==p['test_session'] or p['primary_budget'] not in p['support_budgets']:
        raise ValueError('Invalid session isolation or primary budget')
    return p


def register(plan):
    root=Path(plan['_root']);out=root/plan['output_dir'];out.mkdir(parents=True,exist_ok=True)
    parent=root/plan['parent_output'];sources=['session_models.py','session_study.py','personal_kernels.py','personalization.py','context_data.py']
    identity={'plan':{k:v for k,v in plan.items() if not k.startswith('_')},'methods':METHODS,
        'sources':{n:sha256(root/'src/neuroedge'/n) for n in sources},
        'parent_predictions_sha256':sha256(parent/'external_transfer/predictions.csv'),
        'parent_manifest_sha256':sha256(parent/'external_data/manifest.json')}
    identity['sha256']=digest(identity);path=out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())['sha256']!=identity['sha256']:raise RuntimeError('Frozen study changed; use a new output')
    else:json_write(path,{**identity,'registered_utc':datetime.now(timezone.utc).isoformat(),
        'previous_external_results_already_inspected':True,'new_untouched_dataset_confirmation':False})
    return out,identity


def inputs(plan):
    root=Path(plan['_root']);cohort=load_cohort(context_plan(root/plan['parent_plan']),'external_data')
    frame=pd.read_csv(root/plan['parent_output']/'external_transfer/predictions.csv',float_precision='round_trip')
    result=cohort.meta.copy()
    for model,column in [(plan['source_pipeline'],'source'),(plan['source_reference'],'reference')]:
        one=frame[frame.model==model].set_index('trial_id').loc[result.trial_id]
        for key in ['subject','label','session','run']:np.testing.assert_array_equal(one[key],result[key])
        result[column]=one.prob_right.to_numpy()
    if sorted(result.subject.unique())!=plan['subjects']:raise AssertionError('Cohort differs')
    result['row']=np.arange(len(result))
    return cohort,result


def chronological(frame):return frame.sort_values(['run','onset','trial_id'])


def support_test(meta,person,budget,donor=False):
    group=meta[meta.subject==person];pool=chronological(group[group.session==1])
    support=pool.iloc[:min(budget,96) if donor else budget]
    test=pool[pool.run>support.run.max()] if donor else group[group.session==2]
    if len(support)!=(min(budget,96) if donor else budget) or not len(test):raise AssertionError('Incomplete support or test')
    if set(support.trial_id)&set(test.trial_id):raise AssertionError('Support/test overlap')
    return support,test


def metrics(y,p):
    return {'balanced_accuracy':float(balanced_accuracy_score(y,p>.5)),
        'log_loss':float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
        'macro_f1':float(f1_score(y,p>.5,average='macro',zero_division=0))}


def fit_one(cohort,support,test,method,C):
    column='reference' if method=='source_eegnet' else 'source'
    predictor=SessionModel(method,C).fit(np.asarray(cohort.task_cov[support.row]),support[column].to_numpy(),support.label.to_numpy())
    p=predictor.predict(np.asarray(cohort.task_cov[test.row]),test[column].to_numpy())
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise AssertionError('Invalid predictions')
    return predictor,p


def frame_for(test,p,method,budget,C=None):
    columns=['trial_id','subject','session','run','onset','label']
    result=test[columns].copy();result['prob_right']=p;result['prediction']=(p>.5).astype(int)
    result['method']=method;result['support_budget']=budget;result['C']=C
    return result


def develop(plan):
    out,identity=register(plan);cohort,meta=inputs(plan);frames=[];receipts=[];start=time.time()
    if (out/'donor_complete.json').exists():
        saved=json.loads((out/'donor_complete.json').read_text())
        if saved['prediction_sha256']!=sha256(out/'donor_predictions.csv'):raise AssertionError('Donor evidence changed')
        return pd.read_csv(out/'donor_predictions.csv',float_precision='round_trip')
    for budget in plan['support_budgets']:
        for person in plan['subjects']:
            support,test=support_test(meta,person,budget,True)
            for method in BASE+TUNED:
                for C in plan['C_grid'] if method in TUNED else [None]:
                    _,p=fit_one(cohort,support,test,method,C)
                    frames.append(frame_for(test,p,method,budget,C))
            receipts.append({'subject':person,'support_budget':budget,'actual_donor_support':len(support),
                'support_trial_ids':support.trial_id.tolist(),'validation_trial_ids':test.trial_id.tolist()})
            print('Donor development',budget,person,flush=True)
    all_frames=pd.concat(frames,ignore_index=True);all_frames.to_csv(out/'donor_predictions.csv',index=False)
    json_write(out/'donor_splits.json',receipts)
    json_write(out/'donor_complete.json',{'protocol_sha256':identity['sha256'],'prediction_sha256':sha256(out/'donor_predictions.csv'),
        'fits':len(plan['subjects'])*len(plan['support_budgets'])*(len(BASE)+len(TUNED)*len(plan['C_grid'])),
        'seconds':time.time()-start,'validation_session':1,'test_session_E_used':False})
    return all_frames


def chosen_settings(plan,donor,person,budget):
    pool=donor[(donor.subject!=person)&(donor.support_budget==budget)]
    if set(pool.subject)!=(set(plan['subjects'])-{person}) or set(pool.session)!={1}:raise AssertionError('Donor isolation failed')
    selected={};selection_scores=[]
    for method in TUNED:
        rows=[]
        for C,group in pool[pool.method==method].groupby('C'):
            stats=[metrics(g.label.to_numpy(),g.prob_right.to_numpy()) for _,g in group.groupby('subject')]
            rows.append({'C':float(C),'balanced_accuracy':float(np.mean([s['balanced_accuracy'] for s in stats])),
                         'log_loss':float(np.mean([s['log_loss'] for s in stats]))})
        selected[method]=sorted(rows,key=lambda r:(-r['balanced_accuracy'],r['log_loss'],r['C']))[0]['C']
        selection_scores.extend([{'method':method,**r} for r in rows])
    def selected_frame(method):
        rows=pool[pool.method==method]
        if method in selected:rows=rows[rows.C==selected[method]]
        return rows.set_index('trial_id').sort_index()
    bias=selected_frame('intercept');parts=[selected_frame(m).loc[bias.index].prob_right.to_numpy() for m in ['fbcsp_lda','tangent_rbf_svm','anchored_dual']]
    local=np.mean(parts,axis=0);weights=[]
    for weight in plan['fusion_weights']:
        current=bias.copy();current['prob_right']=(1-weight)*bias.prob_right.to_numpy()+weight*local
        stats=[metrics(g.label.to_numpy(),g.prob_right.to_numpy()) for _,g in current.groupby('subject')]
        weights.append({'weight':weight,'balanced_accuracy':float(np.mean([s['balanced_accuracy'] for s in stats])),
                        'log_loss':float(np.mean([s['log_loss'] for s in stats]))})
    weight=sorted(weights,key=lambda r:(-r['balanced_accuracy'],r['log_loss'],r['weight']))[0]['weight']
    return {'subject':person,'support_budget':budget,'donor_subjects':sorted(set(pool.subject)),
        'selected_C':selected,'fusion_weight':weight,'C_scores':selection_scores,'weight_scores':weights}


def evaluate(plan):
    out,identity=register(plan);donor=develop(plan);cohort,meta=inputs(plan)
    settings=[chosen_settings(plan,donor,p,b) for b in plan['support_budgets'] for p in plan['subjects']]
    selection_path=out/'selection.json';selection={'protocol_sha256':identity['sha256'],
        'donor_predictions_sha256':sha256(out/'donor_predictions.csv'),'settings':settings,'session_E_used':False}
    if selection_path.exists():
        if json.loads(selection_path.read_text())!=selection:raise AssertionError('Frozen selection differs')
    else:json_write(selection_path,selection)
    if (out/'complete.json').exists():return
    frames=[];receipts=[];start=time.time()
    for setting in settings:
        person=setting['subject'];budget=setting['support_budget'];support,test=support_test(meta,person,budget)
        directory=out/'models'/f'S{person}_n{budget}';directory.mkdir(parents=True,exist_ok=True)
        predictions={}
        for method in BASE+TUNED:
            C=setting['selected_C'].get(method)
            model,p=fit_one(cohort,support,test,method,C);predictions[method]=p
            path=directory/(method+'.joblib');joblib.dump(model,path,compress=3)
            frames.append(frame_for(test,p,method,budget,C))
            receipts.append({'subject':person,'support_budget':budget,'method':method,'C':C,
                'model_path':str(path.relative_to(out)),'model_sha256':sha256(path),
                'support_trial_ids':support.trial_id.tolist(),'support_labels_sha256':digest(support.label.tolist()),
                'test_trial_ids':test.trial_id.tolist(),'test_labels_used_for_fit':False})
        local=np.mean([predictions[m] for m in ['fbcsp_lda','tangent_rbf_svm','anchored_dual']],axis=0)
        weight=setting['fusion_weight'];p=(1-weight)*predictions['intercept']+weight*local
        frames.append(frame_for(test,p,'donor_selected_fusion',budget))
        print('External session evaluation',budget,person,flush=True)
    pd.concat(frames,ignore_index=True).to_csv(out/'predictions.csv',index=False);json_write(out/'fit_receipts.json',receipts)
    import importlib.metadata
    json_write(out/'environment.json',{'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','scikit-learn','mne','torch','pyriemann']}})
    json_write(out/'complete.json',{'protocol_sha256':identity['sha256'],'selection_sha256':sha256(selection_path),
        'prediction_sha256':sha256(out/'predictions.csv'),'receipt_sha256':sha256(out/'fit_receipts.json'),
        'model_fits':len(receipts),'methods':METHODS,'budgets':plan['support_budgets'],'subjects':plan['subjects'],
        'seconds':time.time()-start,'test_labels_used_for_fitting_or_selection':False})


def interval(values,plan):
    values=np.asarray(values);rng=np.random.default_rng(plan['bootstrap_seed'])
    means=values[rng.integers(0,len(values),(plan['bootstrap_iterations'],len(values)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()


def summarize(plan):
    out,_=register(plan);frame=pd.read_csv(out/'predictions.csv',float_precision='round_trip');per=[];summary=[];paired=[]
    for (method,budget,person),group in frame.groupby(['method','support_budget','subject']):
        per.append({'method':method,'support_budget':int(budget),'subject':int(person),**metrics(group.label.to_numpy(),group.prob_right.to_numpy())})
    per=pd.DataFrame(per);per.to_csv(out/'metrics_by_subject.csv',index=False)
    for (method,budget),group in per.groupby(['method','support_budget']):
        lo,hi=interval(group.balanced_accuracy,plan)
        summary.append({'method':method,'support_budget':int(budget),'subjects':len(group),
            **{n:float(group[n].mean()) for n in ['balanced_accuracy','log_loss','macro_f1']},'ci_low':lo,'ci_high':hi})
        baseline=per[(per.method=='intercept')&(per.support_budget==budget)].set_index('subject')
        delta=group.set_index('subject').balanced_accuracy-baseline.balanced_accuracy;dl,dh=interval(delta,plan)
        paired.append({'method':method,'support_budget':int(budget),'reference':'intercept','mean_difference':float(delta.mean()),
                       'ci_low':dl,'ci_high':dh,'subjects':len(delta)})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    return pd.DataFrame(summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['register','develop','run','verify','report','all'])
    parser.add_argument('--plan',default='configs/session_adaptation.yaml');args=parser.parse_args();plan=load_plan(args.plan)
    with threadpool_limits(limits=1):
        if args.command=='register':register(plan)
        elif args.command=='develop':develop(plan)
        elif args.command=='run':evaluate(plan);summarize(plan)
        elif args.command=='verify':
            from .session_verify import verify
            verify(plan)
        elif args.command=='report':
            from .session_report import report
            report(plan)
        else:
            evaluate(plan);summarize(plan)
            from .session_verify import verify
            from .session_report import report
            verify(plan);report(plan)


if __name__=='__main__':main()
