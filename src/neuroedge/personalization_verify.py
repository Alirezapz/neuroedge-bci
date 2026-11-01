"""Independent coverage, summary and paired-condition checks for calibration."""
import json
import numpy as np
import pandas as pd

from .config import json_write,sha256
from .context_protocol import interval,score,subject_scores
from .personalization import bind_selection,verify as verify_predictions


def verify(plan):
    result=verify_predictions(plan);out,identity=bind_selection(plan)
    from pathlib import Path
    parent=Path(plan['_root'])/plan['parent_output'];original=pd.read_csv(parent/'external_transfer/predictions.csv')
    frames=pd.read_csv(out/'predictions.csv');receipts=json.loads((out/'adaptation_receipts.json').read_text())
    budgets=[0]+plan['support_budgets'];models=sorted(original.model.unique());people=sorted(original.subject.unique())
    expected={(m,int(s),b) for m in models for s in people for b in budgets}
    observed={(r['model'],r['subject'],r['support_budget']) for r in receipts}
    if observed!=expected or len(receipts)!=len(expected):raise AssertionError('Missing or duplicate adaptation condition.')
    if frames.duplicated(['model','trial_id','support_budget']).any():raise AssertionError('Duplicate personal-calibration prediction.')
    for (model,person,budget),frame in frames.groupby(['model','subject','support_budget']):
        target=original[(original.model==model)&(original.subject==person)&(original.session==plan['test_session'])].set_index('trial_id')
        if set(frame.trial_id)!=set(target.index) or set(frame.session)!={plan['test_session']}:raise AssertionError('Personal test-session coverage differs.')
        np.testing.assert_array_equal(frame.label.to_numpy(),target.loc[frame.trial_id].label.to_numpy())
    if len(frames)!=len(original[original.session==plan['test_session']])*len(budgets):raise AssertionError('Personal prediction total differs.')
    summary=pd.read_csv(out/'summary.csv');per=pd.read_csv(out/'metrics_by_subject.csv')
    if set(zip(summary.model,summary.support_budget))!={(m,b) for m in models for b in budgets}:raise AssertionError('Personal summary coverage differs.')
    bootstrap={'evaluation':{'bootstrap_seed':plan['bootstrap_seed'],'bootstrap_iterations':plan['bootstrap_iterations']}}
    for row in summary.itertuples():
        frame=frames[(frames.model==row.model)&(frames.support_budget==row.support_budget)].reset_index(drop=True)
        p=np.column_stack([1-frame.prob_right,frame.prob_right]);scores=subject_scores(frame,p)
        for key,value in score(frame,p).items():np.testing.assert_allclose(value,getattr(row,key),atol=1e-12,rtol=0)
        lo,hi=interval(scores.balanced_accuracy,bootstrap)
        np.testing.assert_allclose([row.ci_low,row.ci_high],[lo,hi],atol=1e-12,rtol=0)
        actual=per[(per.model==row.model)&(per.support_budget==row.support_budget)].sort_values('subject')
        np.testing.assert_array_equal(actual.subject,scores.sort_values('subject').subject)
        np.testing.assert_allclose(actual.balanced_accuracy,scores.sort_values('subject').balanced_accuracy,atol=1e-12,rtol=0)
    result.update({'summary_metrics_and_participant_intervals_recomputed':True,'all_prespecified_budgets_retained':True,
                   'condition_count':len(expected),'independent_verifier_source_sha256':sha256(__file__)})
    json_write(out/'verification.json',result);return result
