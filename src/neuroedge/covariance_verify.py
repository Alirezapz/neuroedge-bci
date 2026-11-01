"""Replay every covariance artifact, refit support, reconstruct nested choices."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import digest,json_write,sha256
from .covariance_models import CovarianceModel,candidates
from .covariance_study import register,inputs,select_specs,nested_meta,source_bias,stack_features,_key
from .session_study import support_test,metrics,interval


def verify(plan):
    out,identity=register(plan);complete=json.loads((out/'complete.json').read_text())
    for key,path in [('prediction_sha256','predictions.csv'),('selection_sha256','selection.json'),('fit_receipts_sha256','fit_receipts.json'),('nested_meta_sha256','nested_meta_predictions.csv')]:
        if complete[key]!=sha256(out/path):raise AssertionError('Evidence changed: '+path)
    if complete['protocol_sha256']!=identity['sha256']:raise AssertionError('Protocol identity differs')
    meta,paths=inputs(plan);selection=json.loads((out/'selection.json').read_text());stored=pd.read_csv(out/'predictions.csv',float_precision='round_trip')
    frozen_inputs=json.loads((out/'input_manifest.json').read_text())
    if not frozen_inputs['frozen_before_final_E_scoring'] or frozen_inputs['identity']['protocol_sha256']!=identity['sha256']:raise AssertionError('Input freeze identity differs')
    for sensors,path in paths.items():
        if sha256(path)!=frozen_inputs['identity']['covariances'][str(sensors)]['sha256']:raise AssertionError('Original covariance fingerprint differs')
    if digest(meta.trial_id.tolist())!=frozen_inputs['identity']['trial_ids_sha256'] or digest(meta.label.tolist())!=frozen_inputs['identity']['labels_sha256']:raise AssertionError('Input metadata changed')
    if selection['session_E_used'] or set(stored.session)!={2}:raise AssertionError('Selection/test session differs')
    if stored.duplicated(['sensors','condition','support_budget','method','trial_id']).any():raise AssertionError('Duplicate final trial')
    # Recreate original parse path and independently check every candidate metric.
    receipts=json.loads((out/'donor_receipts.json').read_text());tables=[];donor_metric_checks=0
    for r in receipts:
        directory=out/'donors'/_key(r['sensors'],r['condition'],r['support_budget'],r['subject'])
        if sha256(directory/'predictions.csv')!=r['predictions_sha256'] or sha256(directory/'scores.csv')!=r['scores_sha256']:raise AssertionError('Donor evidence changed')
        support,test=support_test(meta,r['subject'],r['support_budget'],True)
        if support.trial_id.tolist()!=r['support_trial_ids'] or test.trial_id.tolist()!=r['validation_trial_ids'] or digest(support.label.tolist())!=r['support_labels_sha256']:raise AssertionError('Donor support differs')
        raw=pd.read_csv(directory/'predictions.csv',float_precision='round_trip');scores=pd.read_csv(directory/'scores.csv')
        if set(raw.session)!={1} or set(raw.trial_id)!=set(test.trial_id):raise AssertionError('Donor session/trial coverage')
        for row in scores.itertuples():
            g=raw[raw.candidate==row.candidate].set_index('trial_id').loc[test.trial_id]
            np.testing.assert_array_equal(g.label.to_numpy(),test.label.to_numpy())
            for metric,value in metrics(g.label.to_numpy(),g.prob_right.to_numpy()).items():np.testing.assert_allclose(value,getattr(row,metric),atol=1e-12,rtol=0)
            donor_metric_checks+=1
        for k in ['sensors','condition','support_budget','subject']:scores[k]=r[k]
        tables.append(scores)
    donor=pd.concat(tables,ignore_index=True).sort_values(['sensors','condition','support_budget','subject','candidate'])
    all_receipts=json.loads((out/'fit_receipts.json').read_text());replays=refits=meta_checks=0
    for i,s in enumerate(selection['settings'],1):
        person=s['subject'];budget=s['support_budget'];sensors=s['sensors'];condition=s['condition'];support,test=support_test(meta,person,budget)
        scores=donor[(donor.sensors==sensors)&(donor.condition==condition)&(donor.support_budget==budget)]
        if select_specs(scores,[person])!=s['selected']:raise AssertionError('Outer-person family choice differs')
        stack,meta_data,details=nested_meta(plan,meta,scores,person,sensors,condition,budget)
        for key in ['columns','meta_C','nested_choices']:
            if details[key]!=s[key]:raise AssertionError('Nested source choices differ: '+key)
        if set(meta_data.subject)&{person}:raise AssertionError('Outer person leaked into meta fit')
        meta_checks+=1;cov=np.load(paths[sensors],mmap_mode='r');A=np.asarray(cov[support.row]);B=np.asarray(cov[test.row]);P={}
        def actual(method):
            g=stored[(stored.sensors==sensors)&(stored.condition==condition)&(stored.support_budget==budget)&(stored.subject==person)&(stored.method==method)]
            if len(g)!=144:raise AssertionError('Expected all144 E trials')
            g=g.set_index('trial_id').loc[test.trial_id]
            for key in ['subject','label','session','run']:np.testing.assert_array_equal(g[key].to_numpy(),test[key].to_numpy())
            np.testing.assert_array_equal(g.prediction.to_numpy(),g.prob_right.to_numpy()>.5)
            return g
        for family,spec in s['selected'].items():
            r=next(r for r in all_receipts if r['key']==s['key'] and r['method']==family)
            if r['support_trial_ids']!=support.trial_id.tolist() or r['test_trial_ids']!=test.trial_id.tolist() or r['support_labels_sha256']!=digest(support.label.tolist()):raise AssertionError('Final support differs')
            file=out/r['model_path']
            if sha256(file)!=r['sha256']:raise AssertionError('Checkpoint changed')
            fitted=joblib.load(file);p=fitted.predict(B,test.source.to_numpy());P[family]=p
            np.testing.assert_allclose(p,actual(family).prob_right,atol=2e-10,rtol=2e-10);replays+=1
            model=CovarianceModel(spec,condition).fit(A,support.label.to_numpy(),support.source.to_numpy())
            np.testing.assert_allclose(model.predict(B,test.source.to_numpy()),p,atol=2e-10,rtol=2e-10);refits+=1
        P['source_intercept']=source_bias(support,test)[0];matrix=np.column_stack([P[c] for c in s['columns']]);r=next(r for r in all_receipts if r['key']==s['key'] and r['method']=='donor_stack');file=out/r['model_path']
        if sha256(file)!=r['sha256'] or set(r['meta_train_subjects'])!=(set(plan['subjects'])-{person}):raise AssertionError('Stack identity differs')
        fitted=joblib.load(file);p=fitted.predict_proba(stack_features(matrix))[:,1]
        np.testing.assert_allclose(p,actual('donor_stack').prob_right,atol=2e-10,rtol=2e-10)
        np.testing.assert_allclose(stack.predict_proba(stack_features(matrix))[:,1],p,atol=2e-10,rtol=2e-10);replays+=1;refits+=1
        np.testing.assert_allclose(matrix.mean(axis=1),actual('equal_probability').prob_right,atol=1e-12,rtol=0)
        from scipy.special import expit
        np.testing.assert_allclose(expit(stack_features(matrix).mean(axis=1)),actual('equal_logit').prob_right,atol=1e-12,rtol=0)
        print('Covariance verification',i,len(selection['settings']),flush=True)
    old=pd.read_csv(Path(plan['_root'])/plan['previous_output']/'predictions.csv',float_precision='round_trip')
    mapping={'reference_previous_fusion':'donor_selected_fusion','reference_previous_tangent':'tangent_rbf_svm','reference_intercept':'intercept'}
    for method,old_method in mapping.items():
        for (sensors,condition,budget),g in stored[stored.method==method].groupby(['sensors','condition','support_budget']):
            reference=old[(old.method==old_method)&(old.support_budget==budget)].set_index('trial_id').loc[g.trial_id]
            np.testing.assert_allclose(g.prob_right.to_numpy(),reference.prob_right.to_numpy(),atol=0,rtol=0)
            if set(g.actual_sensors)!={16} or set(g.actual_condition)!={'strict'}:raise AssertionError('Reference requirement changed')
    per=pd.read_csv(out/'metrics_by_subject.csv');summary=pd.read_csv(out/'summary.csv');paired=pd.read_csv(out/'paired_differences.csv')
    for r in per.itertuples():
        g=stored[(stored.sensors==r.sensors)&(stored.condition==r.condition)&(stored.support_budget==r.support_budget)&(stored.subject==r.subject)&(stored.method==r.method)]
        for metric,value in metrics(g.label.to_numpy(),g.prob_right.to_numpy()).items():np.testing.assert_allclose(value,getattr(r,metric),atol=1e-12,rtol=0)
    for r in summary.itertuples():
        g=per[(per.sensors==r.sensors)&(per.condition==r.condition)&(per.support_budget==r.support_budget)&(per.method==r.method)]
        np.testing.assert_allclose([r.balanced_accuracy,r.ci_low,r.ci_high],[g.balanced_accuracy.mean(),*interval(g.balanced_accuracy,plan)],atol=1e-12,rtol=0)
    for r in paired.itertuples():
        g=per[(per.sensors==r.sensors)&(per.condition==r.condition)&(per.support_budget==r.support_budget)]
        pivot=g.pivot(index='subject',columns='method',values='balanced_accuracy');d=pivot[r.method]-pivot.reference_previous_fusion
        np.testing.assert_allclose([r.difference,r.ci_low,r.ci_high],[d.mean(),*interval(d,plan)],atol=1e-12,rtol=0)
    result={'status':'PASS','checkpoint_replays':replays,'independent_support_meta_refits':refits,'nested_settings_checked':meta_checks,
        'donor_candidate_metrics_checked':donor_metric_checks,'prediction_rows':len(stored),'subjects':9,
        'strict_and_batch_conditions_separate':True,'test_labels_used_for_fit_or_selection':False,
        'untouched_confirmation':False,'verifier_source_sha256':sha256(__file__)}
    json_write(out/'verification.json',result);print(json.dumps(result,indent=2),flush=True);return result
