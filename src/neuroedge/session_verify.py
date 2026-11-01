"""Checkpoint replay, support refit, donor isolation, and independent metrics."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,log_loss,f1_score
from .config import digest,json_write,sha256
from .session_models import BASE,TUNED,METHODS
from .session_study import register,inputs,support_test,fit_one,chosen_settings,interval


def verify(plan):
    out,identity=register(plan);complete=json.loads((out/'complete.json').read_text())
    for key,path in [('prediction_sha256','predictions.csv'),('receipt_sha256','fit_receipts.json'),('selection_sha256','selection.json')]:
        if sha256(out/path)!=complete[key]:raise AssertionError('Changed evidence: '+path)
    if complete['protocol_sha256']!=identity['sha256']:raise AssertionError('Protocol differs')
    donor=pd.read_csv(out/'donor_predictions.csv',float_precision='round_trip')
    ds=json.loads((out/'donor_complete.json').read_text())
    if ds['prediction_sha256']!=sha256(out/'donor_predictions.csv') or ds['protocol_sha256']!=identity['sha256']:raise AssertionError('Donor evidence changed')
    selection=json.loads((out/'selection.json').read_text())
    if selection['session_E_used']:raise AssertionError('Selection used E')
    for setting in selection['settings']:
        rebuilt=chosen_settings(plan,donor,setting['subject'],setting['support_budget'])
        if rebuilt!=setting:raise AssertionError('Donor-only parameter choice differs')
    cohort,meta=inputs(plan);stored=pd.read_csv(out/'predictions.csv',float_precision='round_trip')
    receipts=json.loads((out/'fit_receipts.json').read_text());replays=0;refits=0
    expected_count=len(METHODS)*len(plan['support_budgets'])*len(plan['subjects'])*144
    if len(stored)!=expected_count or stored.duplicated(['method','support_budget','trial_id']).any():raise AssertionError('Coverage or duplicates')
    for r in receipts:
        person=r['subject'];budget=r['support_budget'];method=r['method'];support,test=support_test(meta,person,budget)
        if r['support_trial_ids']!=support.trial_id.tolist() or r['test_trial_ids']!=test.trial_id.tolist():raise AssertionError('Support/test identities differ')
        if r['support_labels_sha256']!=digest(support.label.tolist()) or r['test_labels_used_for_fit']:raise AssertionError('Training labels differ')
        path=out/r['model_path']
        if sha256(path)!=r['model_sha256']:raise AssertionError('Changed checkpoint')
        chosen=next(s for s in selection['settings'] if s['subject']==person and s['support_budget']==budget)
        if r['C']!=chosen['selected_C'].get(method):raise AssertionError('Fit used unselected C')
        actual=stored[(stored.method==method)&(stored.subject==person)&(stored.support_budget==budget)].set_index('trial_id').loc[test.trial_id]
        column='reference' if method=='source_eegnet' else 'source';fitted=joblib.load(path)
        p=fitted.predict(np.asarray(cohort.task_cov[test.row]),test[column].to_numpy())
        np.testing.assert_allclose(p,actual.prob_right,atol=1e-11,rtol=1e-11)
        np.testing.assert_array_equal(p>.5,actual.prediction.to_numpy().astype(bool));replays+=1
        _,refit=fit_one(cohort,support,test,method,r['C'])
        np.testing.assert_allclose(refit,p,atol=1e-10,rtol=1e-10);refits+=1
        for key in ['subject','label','session','run']:np.testing.assert_array_equal(actual[key].to_numpy(),test[key].to_numpy())
    for s in selection['settings']:
        group=stored[(stored.subject==s['subject'])&(stored.support_budget==s['support_budget'])]
        parts={m:g.set_index('trial_id').sort_index() for m,g in group.groupby('method')}
        local=np.mean([parts[m].prob_right for m in ['fbcsp_lda','tangent_rbf_svm','anchored_dual']],axis=0)
        p=(1-s['fusion_weight'])*parts['intercept'].prob_right.to_numpy()+s['fusion_weight']*local
        np.testing.assert_allclose(p,parts['donor_selected_fusion'].prob_right,atol=1e-12,rtol=0)
    per=pd.read_csv(out/'metrics_by_subject.csv');summary=pd.read_csv(out/'summary.csv');paired=pd.read_csv(out/'paired_differences.csv')
    for row in per.itertuples():
        group=stored[(stored.method==row.method)&(stored.subject==row.subject)&(stored.support_budget==row.support_budget)]
        p=group.prob_right.to_numpy();y=group.label.to_numpy()
        np.testing.assert_allclose([row.balanced_accuracy,row.log_loss,row.macro_f1],
            [balanced_accuracy_score(y,p>.5),log_loss(y,np.column_stack([1-p,p]),labels=[0,1]),f1_score(y,p>.5,average='macro',zero_division=0)],atol=1e-12,rtol=0)
    for row in summary.itertuples():
        g=per[(per.method==row.method)&(per.support_budget==row.support_budget)]
        np.testing.assert_allclose([row.balanced_accuracy,row.ci_low,row.ci_high],[g.balanced_accuracy.mean(),*interval(g.balanced_accuracy,plan)],atol=1e-12,rtol=0)
    for row in paired.itertuples():
        g=per[(per.method==row.method)&(per.support_budget==row.support_budget)].set_index('subject')
        ref=per[(per.method=='intercept')&(per.support_budget==row.support_budget)].set_index('subject')
        delta=g.balanced_accuracy-ref.balanced_accuracy
        np.testing.assert_allclose([row.mean_difference,row.ci_low,row.ci_high],[delta.mean(),*interval(delta,plan)],atol=1e-12,rtol=0)
    result={'status':'PASS','checkpoint_replays':replays,'independent_support_refits':refits,
        'donor_choices_reconstructed':len(selection['settings']),'final_prediction_rows':len(stored),
        'subject_metric_rows':len(per),'paired_comparisons':len(paired),'all_test_trials_session_E':True,
        'no_test_labels_or_batch_statistics_in_fit':True,'new_untouched_dataset_confirmation':False,
        'verifier_source_sha256':sha256(__file__)}
    json_write(out/'verification.json',result);print(json.dumps(result,indent=2),flush=True);return result
