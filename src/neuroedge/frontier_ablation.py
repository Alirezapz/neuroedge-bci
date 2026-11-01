"""Descriptive matched development comparisons; never used for model selection."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import digest
from .frontier_features import candidates
from .frontier_protocol import arm,bootstrap,subject_scores


def development_ablations(plan):
    out=Path(plan['_root'])/plan['output_dir']
    specs=[s for s in candidates() if arm(s)=='zero_calibration']
    scores={}
    def get_scores(name):
        if name not in scores:
            frame=pd.read_csv(out/'screening'/name/'predictions.csv')
            p=np.column_stack([1-frame.prob_right,frame.prob_right])
            scores[name]=subject_scores(frame,p).set_index('subject').balanced_accuracy
        return scores[name]
    factors=[('channels','channels',16,64),('time_window','window','late','full'),
             ('broad_band','bands',[1],[0]),('mixup','augmentation','none','mixup'),
             ('segment_recombine','augmentation','none','segment_recombine'),
             ('trace_normalization','normalize',False,True),
             ('csp_shrinkage_01_to_1','reg',.01,.1),('csp_shrinkage_1_to_3','reg',.1,.3)]
    rows=[]
    for factor,field,reference,variant in factors:
        def key(spec):return digest({k:v for k,v in spec.items() if k not in {'name',field}})
        bases={key(s):s for s in specs if s.get(field,object())==reference}
        for spec in [s for s in specs if s.get(field,object())==variant]:
            if key(spec) not in bases:continue
            base=bases[key(spec)]
            a,b=get_scores(base['name']),get_scores(spec['name'])
            if set(a.index)!=set(b.index):raise AssertionError('Matched development subjects differ.')
            difference=b-a;lo,hi=bootstrap(difference,plan)
            rows.append({'factor':factor,'family':spec['family'],'architecture':spec.get('architecture',''),
                         'base':base['name'],'variant':spec['name'],'reference_value':str(reference),
                         'variant_value':str(variant),'subjects':len(difference),
                         'base_balanced_accuracy':float(a.mean()),'variant_balanced_accuracy':float(b.mean()),
                         'difference':float(difference.mean()),'ci_low':lo,'ci_high':hi,
                         'scope':'Discovery OOF only; conditional descriptive intervals, overlapping source folds, no multiplicity-adjusted claim'})
    result=pd.DataFrame(rows)
    result.to_csv(out/'matched_development_ablations.csv',index=False)
    return result
