"""Source-only diagnostics; these are NOT held-out research accuracy results.

Checks architecture capacity on 32 real training examples and whether shuffled
within-person labels remove predictable group-CV signal. Does not change the
frozen model search, selection, or confirmation.
"""
from pathlib import Path
import argparse
import copy

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from neuroedge.config import json_write, sha256
from neuroedge.frontier import discovery_splits
from neuroedge.frontier_data import load_cohort
from neuroedge.frontier_features import candidates, CovarianceModel
from neuroedge.frontier_neural import fit_neural
from neuroedge.frontier_protocol import load_plan, score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='configs/frontier.yaml')
    args=parser.parse_args()
    plan=load_plan(args.plan)
    cohort=load_cohort(plan,'development_data')
    source=np.flatnonzero(cohort.meta.subject.isin(plan['discovery_subjects']))
    out=Path(plan['_root'])/plan['output_dir']/'diagnostics'
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    rng=np.random.default_rng(43)
    chosen=np.concatenate([rng.choice(source[cohort.meta.iloc[source].label.to_numpy()==label],16,replace=False)
                           for label in [0,1]])
    y=cohort.meta.iloc[chosen].label.to_numpy(dtype=np.int64)
    records=[]
    control_plan=copy.deepcopy(plan)
    control_plan['neural']['learning_rate']=.003
    control_plan['neural']['weight_decay']=0
    with threadpool_limits(limits=2):
        for architecture in ['eegnet','shallow','attention']:
            spec=copy.deepcopy(next(s for s in candidates() if s.get('architecture')==architecture
                and s['channels']==16 and s.get('augmentation')=='none' and not s.get('arm')))
            spec['dropout']=0
            X=cohort.neural_view(spec,chosen)
            directory=Path(plan['_root'])/plan['dataset']['cache_dir']/'controls'/architecture
            model=fit_neural(X,y,cohort.meta.iloc[chosen].subject.to_numpy(),spec,control_plan,directory,
                             epochs=120,seed=7)
            train_accuracy=float(np.mean(model.predict_proba(X).argmax(1)==y))
            records.append({'architecture':architecture,'training_examples':32,'epochs':120,
                            'training_accuracy':train_accuracy,'heldout_examples':0})
            print('MEMORIZATION',architecture,train_accuracy,flush=True)
        spec=next(s for s in candidates() if s['family']=='csp' and s['channels']==16
                  and s['window']=='late' and s['bands']==[1] and s['components']==4 and s['reg']==.1
                  and s['classifier']=='lda' and not s.get('arm'))
        cov=cohort.covariance_view(spec,source)
        meta=cohort.meta.iloc[source].reset_index(drop=True)
        permutation_rows=[]
        for repetition in range(10):
            permuted=meta.label.to_numpy(dtype=int).copy()
            for subject in sorted(meta.subject.unique()):
                rows=np.flatnonzero(meta.subject==subject)
                permuted[rows]=rng.permutation(permuted[rows])
            p=np.zeros((len(meta),2))
            for split in discovery_splits(plan):
                train=np.flatnonzero(meta.subject.isin(split['train_subjects']))
                val=np.flatnonzero(meta.subject.isin(split['validation_subjects']))
                model=CovarianceModel(spec).fit(cov[train],permuted[train])
                p[val]=model.predict_proba(cov[val])
            shuffled_meta=meta.copy();shuffled_meta['label']=permuted
            result={'repetition':repetition,**score(shuffled_meta,p)}
            permutation_rows.append(result)
            print('LABEL SHUFFLE',repetition,result['balanced_accuracy'],flush=True)
    pd.DataFrame(records).to_csv(out/'memorization_control.csv',index=False)
    pd.DataFrame(permutation_rows).to_csv(out/'label_permutation_control.csv',index=False)
    json_write(out/'controls.json',{'scope':'Real discovery-training data only; software/learning diagnostics, not held-out accuracy.',
        'source_subjects':plan['discovery_subjects'],'memorization_trial_ids':cohort.meta.iloc[chosen].trial_id.tolist(),
        'memorization':records,'permutations':10,
        'permuted_group_cv_mean_balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in permutation_rows])),
        'permuted_group_cv_min':float(min(r['balanced_accuracy'] for r in permutation_rows)),
        'permuted_group_cv_max':float(max(r['balanced_accuracy'] for r in permutation_rows)),
        'control_source_sha256':sha256(__file__),'confirmation_data_used':False,
        'memorization_setting':'dropout0,AdamWlr.003,weightdecay0,120epochs; diagnostic architecture-capacity setting, not the evaluated research configuration'})


if __name__=='__main__':main()
