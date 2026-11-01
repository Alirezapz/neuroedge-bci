"""Independently check source scalers, chronological support and all refits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .config import digest,json_write,sha256
from .context_data import load_cohort
from .context_protocol import interval,score,subject_scores
from .personal_kernels import features,fit_scaler,transform,subset,kernel,fit_residual,predict_residual
from .personalization import fit_intercept,adapt
from .residual_study import register,base_frame,bias_config


def check_predictions(plan,cohort,prior,values,actual,receipts,kind,C,budget,source):
    meta=cohort.meta;lookup=pd.Series(np.arange(len(meta)),index=meta.trial_id)
    probabilities=prior.set_index('trial_id').prob_right
    people=plan['_parent']['selection_subjects'] if source else [1000+s for s in plan['_parent']['external']['subjects']]
    if set(actual.subject)!=set(people) or actual.trial_id.duplicated().any():raise AssertionError('Residual person/trial coverage differs.')
    if len(receipts)!=len(people) or len({r['subject'] for r in receipts})!=len(people):raise AssertionError('Residual fit receipt coverage differs.')
    all_test=[]
    for person in people:
        group=meta[meta.subject==person]
        pool=group[group.run.isin(plan['source_calibration_runs'])] if source else group[group.session==plan['calibration_session']]
        support=pool.sort_values(['run','onset','trial_id']).iloc[:budget]
        test=group[group.run==plan['source_test_run']] if source else group[group.session==plan['test_session']]
        test=test.sort_values(['run','onset','trial_id']);all_test.extend(test.trial_id)
        if (not source and len(test)!=144) or len(support)!=budget or set(support.trial_id)&set(test.trial_id):raise AssertionError('Personal support/test separation differs.')
        receipt=next(r for r in receipts if r['subject']==person)
        if receipt['model']!=kind or receipt['C']!=C or receipt['support_budget']!=budget or receipt['support_trial_ids']!=support.trial_id.tolist():raise AssertionError('Personal support or frozen setting differs.')
        if receipt['test_trial_ids_sha256']!=digest(test.trial_id.tolist()):raise AssertionError('Personal test identities differ.')
        ps=probabilities.loc[support.trial_id].to_numpy();pt=probabilities.loc[test.trial_id].to_numpy()
        if not budget:
            expected=pt;fit={'alpha':[],'intercept':0.,'status':'no_personal_calibration'}
        elif kind=='bias':
            b,status=fit_intercept(ps,support.label,bias_config(plan));expected=adapt(pt,b,bias_config(plan))[:,1]
            fit={'alpha':[],'intercept':b,'status':status}
        else:
            S=subset(values,lookup.loc[support.trial_id].to_numpy());T=subset(values,lookup.loc[test.trial_id].to_numpy())
            fit=fit_residual(kernel(S,S,kind),ps,support.label,C,plan['adaptation'])
            expected=predict_residual(kernel(T,S,kind),pt,fit,plan['adaptation'])
        if receipt['status']!=fit['status']:raise AssertionError('Fit status differs.')
        np.testing.assert_allclose(receipt['alpha'],fit['alpha'],atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(receipt['intercept'],fit['intercept'],atol=2e-7,rtol=2e-7)
        frame=actual.set_index('trial_id').loc[test.trial_id]
        for key in ['subject','label','run','session']:np.testing.assert_array_equal(frame[key],test[key].to_numpy())
        np.testing.assert_allclose(frame.prob_right,expected,atol=2e-7,rtol=2e-7)
        np.testing.assert_array_equal(frame.prediction,(expected>.5).astype(int))
    if set(actual.trial_id)!=set(all_test):raise AssertionError('Residual test coverage differs.')


def verify(plan,bias_reference_dir=None):
    out,identity=register(plan);root=Path(plan['_root']);parent=root/plan['parent_output']
    specification=json.loads((out/'specification.json').read_text());claimed=specification.pop('sha256')
    if digest(specification)!=claimed or not specification['mathematical_specification_before_external_acquisition']:raise AssertionError('Residual mathematical registration differs.')
    selection=json.loads((out/'source_selection.json').read_text())
    if selection['implementation_sha256']!=identity['sha256'] or selection['external_used_for_selection'] or not selection['frozen_before_parent_external_transfer_scoring']:
        raise AssertionError('Residual source-only selection differs.')
    development=load_cohort(plan['_parent'],'development_data');raw=features(development,np.arange(len(development.meta)))
    rows=np.flatnonzero(development.meta.subject.isin(plan['_parent']['source_subjects']))
    for filename,expected,people,ids in [
        ('development_source_scaler.npz',fit_scaler(subset(raw,rows)),selection['development_scaler_train_subjects'],selection['development_scaler_train_trial_ids_sha256']),
        ('final_source_scaler.npz',fit_scaler(raw),selection['final_scaler_train_subjects'],selection['final_scaler_train_trial_ids_sha256'])]:
        training=development.meta.iloc[rows] if filename.startswith('development') else development.meta
        if people!=sorted(set(training.subject.tolist())) or ids!=digest(training.trial_id.tolist()):raise AssertionError('Residual scaler training population differs.')
        with np.load(out/filename) as saved:
            if set(saved.files)!=set(expected):raise AssertionError('Residual scaler feature roles differ.')
            for key in expected:np.testing.assert_allclose(saved[key],expected[key],atol=1e-9,rtol=1e-9)
    if sha256(out/'final_source_scaler.npz')!=selection['final_scaler_sha256'] or sha256(out/'development_source_scaler.npz')!=selection['development_scaler_sha256']:raise AssertionError('Residual source scaler checksum differs.')
    prior,_=base_frame(plan,development,True)
    with np.load(out/'development_source_scaler.npz') as f:scaler={k:f[k] for k in f.files}
    values=transform(raw,scaler);table=pd.read_csv(out/'source_summary.csv')
    if set(zip(table.model,table.C))!={(kind,float(C)) for kind in plan['kernels'] for C in plan['C_grid']}:raise AssertionError('Residual source search coverage differs.')
    for row in table.itertuples():
        directory=out/'source_candidates'/f'{row.model}_C{row.C:g}'
        if sha256(directory/'predictions.csv')!=row.prediction_sha256:raise AssertionError('Residual source predictions changed.')
        frame=pd.read_csv(directory/'predictions.csv');receipts=json.loads((directory/'receipts.json').read_text())
        check_predictions(plan,development,prior,values,frame,receipts,row.model,row.C,plan['source_selection_budget'],True)
        for key,value in score(frame,np.column_stack([1-frame.prob_right,frame.prob_right])).items():np.testing.assert_allclose(value,getattr(row,key),atol=1e-12,rtol=0)
    for kind in plan['kernels']:
        expected=table[table.model==kind].sort_values(['balanced_accuracy','log_loss','C'],ascending=[False,True,True]).iloc[0]
        if selection['selected'][kind]['C']!=expected.C or selection['selected'][kind]['prediction_sha256']!=expected.prediction_sha256:raise AssertionError('Residual source winner differs.')
    if selection['primary_kernel']!=plan['primary_kernel'] or selection['primary_budget']!=plan['primary_budget']:raise AssertionError('Primary residual condition changed.')
    complete=json.loads((out/'complete.json').read_text())
    if complete['implementation_sha256']!=identity['sha256'] or complete['source_selection_sha256']!=sha256(out/'source_selection.json') or complete['parent_transfer_prediction_sha256']!=sha256(parent/'external_transfer/predictions.csv') or complete['external_used_for_method_or_budget_selection']:
        raise AssertionError('Residual evaluation dependencies differ.')
    if complete['prediction_sha256']!=sha256(out/'predictions.csv') or complete['receipt_sha256']!=sha256(out/'adaptation_receipts.json'):raise AssertionError('Residual final results changed.')
    if json.loads((parent/'verification.json').read_text())['status']!='PASS':raise AssertionError('Parent predictions must be verified.')
    external=load_cohort(plan['_parent'],'external_data');prior,_=base_frame(plan,external,False)
    with np.load(out/'final_source_scaler.npz') as f:scaler={k:f[k] for k in f.files}
    values=transform(features(external,np.arange(len(external.meta))),scaler)
    final=pd.read_csv(out/'predictions.csv');receipts=json.loads((out/'adaptation_receipts.json').read_text())
    expected_groups={(k,b) for k in ['bias']+plan['kernels'] for b in [0]+plan['support_budgets']}
    if set(zip(final.model,final.support_budget))!=expected_groups:raise AssertionError('Residual method/budget coverage differs.')
    summary=pd.read_csv(out/'summary.csv');per=pd.read_csv(out/'metrics_by_subject.csv')
    if set(zip(summary.model,summary.support_budget))!=expected_groups:raise AssertionError('Residual summary coverage differs.')
    bootstrap={'evaluation':{'bootstrap_iterations':plan['bootstrap_iterations'],'bootstrap_seed':plan['bootstrap_seed']}}
    for kind,budget in expected_groups:
        frame=final[(final.model==kind)&(final.support_budget==budget)].reset_index(drop=True)
        current=[r for r in receipts if r['model']==kind and r['support_budget']==budget]
        C=None if kind=='bias' else selection['selected'][kind]['C']
        check_predictions(plan,external,prior,values,frame,current,kind,C,budget,False)
        p=np.column_stack([1-frame.prob_right,frame.prob_right]);row=summary[(summary.model==kind)&(summary.support_budget==budget)].iloc[0]
        for key,value in score(frame,p).items():np.testing.assert_allclose(value,row[key],atol=1e-12,rtol=0)
        scores=subject_scores(frame,p);lo,hi=interval(scores.balanced_accuracy,bootstrap)
        np.testing.assert_allclose([row.ci_low,row.ci_high],[lo,hi],atol=1e-12,rtol=0)
        stored=per[(per.model==kind)&(per.support_budget==budget)].sort_values('subject')
        np.testing.assert_array_equal(stored.subject,scores.sort_values('subject').subject)
        np.testing.assert_allclose(stored.balanced_accuracy,scores.sort_values('subject').balanced_accuracy,atol=1e-12,rtol=0)
    matched_bias=False
    if bias_reference_dir:
        other=pd.read_csv(root/bias_reference_dir/'predictions.csv');other=other[other.model==plan['base_pipeline']]
        actual=final[final.model=='bias'];keys=['subject','support_budget','trial_id']
        a=actual.sort_values(keys).reset_index(drop=True);b=other.sort_values(keys).reset_index(drop=True)
        for key in keys+['label','prediction']:np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_allclose(a.prob_right,b.prob_right,atol=2e-7,rtol=2e-7);matched_bias=True
    result={'status':'PASS','source_candidates_recomputed':len(table),'source_support_fits_recomputed':len(table)*len(plan['_parent']['selection_subjects']),
        'external_method_budget_person_checks':len(expected_groups)*len(plan['_parent']['external']['subjects']),
        'source_only_scalers_recomputed':2,'all_supports_chronological_and_session_disjoint':True,
        'matches_existing_bias_calibration':matched_bias,'implementation_sha256':identity['sha256'],
        'prediction_sha256':sha256(out/'predictions.csv'),'parent_verification_sha256':sha256(parent/'verification.json')}
    json_write(out/'verification.json',result);print(result,flush=True);return result


if __name__=='__main__':
    import argparse
    from .residual_study import load_plan
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='configs/residual_personal.yaml')
    parser.add_argument('--bias-reference');args=parser.parse_args();verify(load_plan(args.plan),args.bias_reference)
