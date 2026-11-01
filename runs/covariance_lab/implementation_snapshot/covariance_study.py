"""Expanded covariance screen with nested, participant-disjoint donor stacking."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml
from .config import digest,json_write,sha256
from .covariance_models import candidates,CovarianceModel
from .covariance_data import prepare22,parent_cohort,covariances
from .personalization import fit_intercept,adapt
from .session_models import BIAS
from .session_study import support_test,metrics,interval


def load_plan(path):
    path=Path(path).resolve();p=yaml.safe_load(path.read_text());p['_root']=str(path.parent.parent);return p


def register(plan):
    root=Path(plan['_root']);out=root/plan['output_dir'];out.mkdir(parents=True,exist_ok=True)
    files=['covariance_models.py','covariance_data.py','covariance_study.py','personal_kernels.py','personalization.py','session_models.py','session_study.py','context_data.py']
    identity={'plan':{k:v for k,v in plan.items() if not k.startswith('_')},'candidates':candidates(),
        'sources':{n:sha256(root/'src/neuroedge'/n) for n in files},
        'parent_prediction_sha256':sha256(root/plan['parent_output']/'external_transfer/predictions.csv'),
        'previous_prediction_sha256':sha256(root/plan['previous_output']/'predictions.csv')}
    identity['sha256']=digest(identity);p=out/'protocol.json'
    if p.exists():
        if json.loads(p.read_text())['sha256']!=identity['sha256']:raise RuntimeError('Frozen covariance study changed; use another output')
    else:json_write(p,{**identity,'registered_utc':datetime.now(timezone.utc).isoformat(),'previous_external_results_inspected':True,'untouched_confirmation':False})
    json_write(out/'candidates.json',candidates());return out,identity


def inputs(plan):
    root=Path(plan['_root']);cohort=parent_cohort(plan);meta=cohort.meta.copy();meta['row']=np.arange(len(meta))
    original=pd.read_csv(root/plan['parent_output']/'external_transfer/predictions.csv',float_precision='round_trip')
    base=original[original.model=='selected_zero_calibration'].set_index('trial_id').loc[meta.trial_id]
    np.testing.assert_array_equal(base.label.to_numpy(),meta.label.to_numpy());meta['source']=base.prob_right.to_numpy()
    paths={16:str(root/cohort.manifest['files']['task_cov']['path'])}
    if 22 in plan['sensor_counts']:
        prepare22(plan);covariances(plan,22);paths[22]=str(root/plan['cache_dir']/'cov22.npy')
    return meta,paths


def _key(sensors,condition,budget,person):return f'c{sensors}_{condition}_n{budget}_S{person}'


def _worker(plan,records,path,sensors,condition,budget,person):
    with threadpool_limits(limits=1):
        out=Path(plan['_root'])/plan['output_dir'];directory=out/'donors'/_key(sensors,condition,budget,person);directory.mkdir(parents=True,exist_ok=True)
        receipt=directory/'complete.json'
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r['predictions_sha256']!=sha256(directory/'predictions.csv'):raise AssertionError('Donor predictions changed')
            return r
        meta=pd.DataFrame(records);support,test=support_test(meta,person,budget,True);cov=np.load(path,mmap_mode='r')
        A=np.asarray(cov[support.row]);B=np.asarray(cov[test.row]);y=support.label.to_numpy();p0=support.source.to_numpy();q0=test.source.to_numpy();frames=[];rows=[]
        for spec in candidates():
            model=CovarianceModel(spec,condition).fit(A,y,p0);p=model.predict(B,q0)
            frame=test[['trial_id','subject','session','run','onset','label']].copy();frame['prob_right']=p;frame['candidate']=spec['name'];frames.append(frame)
            rows.append({'candidate':spec['name'],'family':spec['family'],**metrics(test.label.to_numpy(),p)})
        pd.concat(frames,ignore_index=True).to_csv(directory/'predictions.csv',index=False);pd.DataFrame(rows).to_csv(directory/'scores.csv',index=False)
        r={'sensors':sensors,'condition':condition,'support_budget':budget,'subject':person,
            'support_trial_ids':support.trial_id.tolist(),'validation_trial_ids':test.trial_id.tolist(),
            'support_labels_sha256':digest(support.label.tolist()),'predictions_sha256':sha256(directory/'predictions.csv'),
            'scores_sha256':sha256(directory/'scores.csv'),'fits':len(rows)}
        json_write(receipt,r);return r


def develop(plan,workers=4):
    out,identity=register(plan);meta,paths=inputs(plan);jobs=[];receipts=[];start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sensors in plan['sensor_counts']:
            for condition in plan['conditions']:
                for budget in plan['support_budgets']:
                    for person in plan['subjects']:
                        jobs.append(pool.submit(_worker,plan,meta.to_dict('records'),paths[sensors],sensors,condition,budget,person))
        for i,job in enumerate(as_completed(jobs),1):
            r=job.result();receipts.append(r);print('Covariance donor screen',i,len(jobs),r['sensors'],r['condition'],r['support_budget'],r['subject'],flush=True)
    frames=[]
    for r in receipts:
        p=out/'donors'/_key(r['sensors'],r['condition'],r['support_budget'],r['subject']);f=pd.read_csv(p/'scores.csv')
        for k in ['sensors','condition','support_budget','subject']:f[k]=r[k]
        frames.append(f)
    all_scores=pd.concat(frames,ignore_index=True).sort_values(['sensors','condition','support_budget','subject','candidate'])
    all_scores.to_csv(out/'donor_summary.csv',index=False);json_write(out/'donor_receipts.json',receipts)
    json_write(out/'donor_complete.json',{'protocol_sha256':identity['sha256'],'fits':len(all_scores),'seconds':time.time()-start,
        'summary_sha256':sha256(out/'donor_summary.csv'),'session_E_used':False})
    return all_scores,meta,paths


def select_specs(scores,excluded):
    pool=scores[~scores.subject.isin(excluded)];selected={};specs={s['name']:s for s in candidates()}
    for family,group in pool.groupby('family'):
        rank=group.groupby('candidate')[['balanced_accuracy','log_loss']].mean().reset_index().sort_values(['balanced_accuracy','log_loss','candidate'],ascending=[False,True,True])
        selected[family]=specs[rank.iloc[0].candidate]
    return selected


def source_bias(support,test):
    b,status=fit_intercept(support.source,support.label,BIAS)
    return adapt(test.source,b,BIAS)[:,1],{'intercept':b,'status':status}


def stack_features(P):return logit(np.clip(np.asarray(P),1e-6,1-1e-6))


def fit_stack(X,y,C):return make_pipeline(StandardScaler(),LogisticRegression(C=C,class_weight='balanced',max_iter=2000,random_state=7)).fit(X,y)


def nested_meta(plan,meta,scores,person,sensors,condition,budget):
    out=Path(plan['_root'])/plan['output_dir'];frames=[];settings=[];families=sorted(scores.family.unique())
    for donor in sorted(set(plan['subjects'])-{person}):
        choices=select_specs(scores,[person,donor]);directory=out/'donors'/_key(sensors,condition,budget,donor)
        predictions=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
        support,test=support_test(meta,donor,budget,True);frame=test[['trial_id','subject','label']].copy()
        for family in families:
            r=predictions[predictions.candidate==choices[family]['name']].set_index('trial_id').loc[test.trial_id]
            frame[family]=r.prob_right.to_numpy()
        frame['source_intercept']=source_bias(support,test)[0];frames.append(frame)
        settings.append({'donor':donor,'excluded_subjects':[person,donor],'selected':choices,'validation_trial_ids':test.trial_id.tolist()})
    data=pd.concat(frames,ignore_index=True);columns=families+['source_intercept'];X=stack_features(data[columns]);y=data.label.to_numpy();results=[]
    for C in plan['stack_C_grid']:
        stats=[]
        for donor in sorted(data.subject.unique()):
            train=data.subject.to_numpy()!=donor;model=fit_stack(X[train],y[train],C);p=model.predict_proba(X[~train])[:,1];stats.append(metrics(y[~train],p))
        results.append({'C':C,'balanced_accuracy':float(np.mean([s['balanced_accuracy'] for s in stats])),'log_loss':float(np.mean([s['log_loss'] for s in stats]))})
    chosen=sorted(results,key=lambda s:(-s['balanced_accuracy'],s['log_loss'],s['C']))[0]
    return fit_stack(X,y,chosen['C']),data,{'columns':columns,'meta_C':chosen['C'],'meta_C_scores':results,'nested_choices':settings}


def evaluate(plan,workers=4):
    out,identity=register(plan)
    if (out/'complete.json').exists():return
    scores,meta,paths=develop(plan,workers);settings=[];fitted_stacks={};meta_tables=[]
    for sensors in plan['sensor_counts']:
        for condition in plan['conditions']:
            for budget in plan['support_budgets']:
                current=scores[(scores.sensors==sensors)&(scores.condition==condition)&(scores.support_budget==budget)]
                for person in plan['subjects']:
                    chosen=select_specs(current,[person]);stack,data,details=nested_meta(plan,meta,current,person,sensors,condition,budget)
                    key=_key(sensors,condition,budget,person);fitted_stacks[key]=stack;data['outer_key']=key;meta_tables.append(data)
                    settings.append({'key':key,'sensors':sensors,'condition':condition,'support_budget':budget,'subject':person,'selected':chosen,**details})
    selection={'protocol_sha256':identity['sha256'],'donor_summary_sha256':sha256(out/'donor_summary.csv'),'settings':settings,'session_E_used':False}
    p=out/'selection.json'
    if p.exists():
        if json.loads(p.read_text())!=selection:raise RuntimeError('Frozen family/meta choices differ')
    else:json_write(p,selection)
    pd.concat(meta_tables,ignore_index=True).to_csv(out/'nested_meta_predictions.csv',index=False)
    old=pd.read_csv(Path(plan['_root'])/plan['previous_output']/'predictions.csv',float_precision='round_trip');frames=[];receipts=[]
    for i,s in enumerate(settings,1):
        sensors=s['sensors'];condition=s['condition'];budget=s['support_budget'];person=s['subject'];support,test=support_test(meta,person,budget)
        cov=np.load(paths[sensors],mmap_mode='r');A=np.asarray(cov[support.row]);B=np.asarray(cov[test.row]);y=support.label.to_numpy();P={}
        directory=out/'models'/s['key'];directory.mkdir(parents=True,exist_ok=True)
        def save_frame(method,probability,actual_sensors=sensors,actual_condition=condition):
            if not np.isfinite(probability).all() or np.min(probability)<0 or np.max(probability)>1:raise AssertionError('Invalid scores')
            f=test[['trial_id','subject','session','run','onset','label']].copy();f['prob_right']=probability;f['prediction']=(probability>.5).astype(int)
            f['method']=method;f['support_budget']=budget;f['sensors']=sensors;f['condition']=condition;f['actual_sensors']=actual_sensors;f['actual_condition']=actual_condition;frames.append(f)
        for family,spec in s['selected'].items():
            model=CovarianceModel(spec,condition).fit(A,y,support.source.to_numpy());p=model.predict(B,test.source.to_numpy());P[family]=p;save_frame(family,p)
            file=directory/(family+'.joblib');joblib.dump(model,file,compress=3)
            receipts.append({'key':s['key'],'method':family,'spec':spec,'model_path':str(file.relative_to(out)),'sha256':sha256(file),
                'support_trial_ids':support.trial_id.tolist(),'support_labels_sha256':digest(support.label.tolist()),'test_trial_ids':test.trial_id.tolist()})
        P['source_intercept']=source_bias(support,test)[0];matrix=np.column_stack([P[c] for c in s['columns']]);stack=fitted_stacks[s['key']]
        save_frame('donor_stack',stack.predict_proba(stack_features(matrix))[:,1]);save_frame('equal_probability',matrix.mean(axis=1));save_frame('equal_logit',expit(stack_features(matrix).mean(axis=1)))
        file=directory/'donor_stack.joblib';joblib.dump(stack,file,compress=3)
        receipts.append({'key':s['key'],'method':'donor_stack','model_path':str(file.relative_to(out)),'sha256':sha256(file),'meta_train_subjects':sorted(set(plan['subjects'])-{person})})
        for old_method,label in [('donor_selected_fusion','reference_previous_fusion'),('tangent_rbf_svm','reference_previous_tangent'),('intercept','reference_intercept')]:
            previous=old[(old.method==old_method)&(old.support_budget==budget)&(old.subject==person)].set_index('trial_id').loc[test.trial_id]
            save_frame(label,previous.prob_right.to_numpy(),16,'strict')
        print('Covariance final evaluation',i,len(settings),s['key'],flush=True)
    pd.concat(frames,ignore_index=True).to_csv(out/'predictions.csv',index=False);json_write(out/'fit_receipts.json',receipts)
    json_write(out/'complete.json',{'protocol_sha256':identity['sha256'],'selection_sha256':sha256(out/'selection.json'),
        'prediction_sha256':sha256(out/'predictions.csv'),'fit_receipts_sha256':sha256(out/'fit_receipts.json'),
        'nested_meta_sha256':sha256(out/'nested_meta_predictions.csv'),'saved_models':len(receipts),'test_labels_used_for_selection':False})


def summarize(plan):
    out,_=register(plan);frame=pd.read_csv(out/'predictions.csv',float_precision='round_trip');per=[];summary=[];paired=[]
    keys=['sensors','condition','support_budget','method']
    for values,g in frame.groupby(keys+['subject']):per.append({**dict(zip(keys+['subject'],values)),**metrics(g.label.to_numpy(),g.prob_right.to_numpy())})
    per=pd.DataFrame(per);per.to_csv(out/'metrics_by_subject.csv',index=False)
    for values,g in per.groupby(keys):
        row=dict(zip(keys,values));lo,hi=interval(g.balanced_accuracy,plan)
        summary.append({**row,'balanced_accuracy':float(g.balanced_accuracy.mean()),'ci_low':lo,'ci_high':hi,'log_loss':float(g.log_loss.mean()),'macro_f1':float(g.macro_f1.mean()),'subjects':len(g)})
        ref=per[(per.sensors==row['sensors'])&(per.condition==row['condition'])&(per.support_budget==row['support_budget'])&(per.method=='reference_previous_fusion')].set_index('subject')
        d=g.set_index('subject').balanced_accuracy-ref.balanced_accuracy;l,h=interval(d,plan)
        paired.append({**row,'reference':'previous_fusion_16_strict','difference':float(d.mean()),'ci_low':l,'ci_high':h,'same_sensors':row['sensors']==16,'same_input_condition':row['condition']=='strict'})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    return pd.DataFrame(summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['register','prepare','develop','run','verify','report','all']);parser.add_argument('--plan',default='configs/covariance_lab.yaml');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();p=load_plan(args.plan)
    with threadpool_limits(limits=1):
        register(p)
        if args.command=='prepare':inputs(p)
        elif args.command=='develop':develop(p,args.workers)
        elif args.command in ['run','all']:evaluate(p,args.workers);summarize(p)
        if args.command in ['verify','all']:
            from .covariance_verify import verify
            verify(p)
        if args.command in ['report','all']:
            from .covariance_report import report
            report(p)


if __name__=='__main__':main()
