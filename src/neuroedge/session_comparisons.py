"""Reporting-only complete paired comparisons; never changes trained choices."""
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import json_write,sha256
from .session_study import interval


def save_pairwise(plan):
    out=Path(plan['_root'])/plan['output_dir'];per=pd.read_csv(out/'metrics_by_subject.csv')
    rows=[]
    for budget,group in per.groupby('support_budget'):
        pivot=group.pivot(index='subject',columns='method',values='balanced_accuracy')
        for a,b in combinations(sorted(pivot.columns),2):
            d=pivot[a]-pivot[b];lo,hi=interval(d,plan)
            rows.append({'method_a':a,'method_b':b,'support_budget':int(budget),'subjects':len(d),
                'difference_a_minus_b':float(d.mean()),'ci_low':lo,'ci_high':hi})
    frame=pd.DataFrame(rows);frame.to_csv(out/'all_pairwise_comparisons.csv',index=False)
    # Independent arithmetic on serialized raw predictions checks the means.
    raw=pd.read_csv(out/'predictions.csv',float_precision='round_trip')
    raw['correct']=(raw.prediction==raw.label).astype(float)
    byclass=raw.groupby(['method','support_budget','subject','label']).correct.mean()
    byperson=byclass.groupby(['method','support_budget','subject']).mean()
    for r in frame.itertuples():
        delta=byperson.loc[(r.method_a,r.support_budget)]-byperson.loc[(r.method_b,r.support_budget)]
        np.testing.assert_allclose(delta.mean(),r.difference_a_minus_b,atol=1e-12,rtol=0)
    json_write(out/'comparison_verification.json',{'status':'PASS','comparisons':len(frame),
        'paired_by_person_and_same_budget':True,'multiplicity_adjusted':False,
        'means_recomputed_from_actual_class_recalls':True,'source_sha256':sha256(__file__),
        'all_pairwise_sha256':sha256(out/'all_pairwise_comparisons.csv')})
    return frame
