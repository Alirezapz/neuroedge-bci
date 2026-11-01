"""Source-only learning diagnostics; scores are never held-out research results."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from neuroedge.config import digest,json_write,sha256
from neuroedge.context_data import load_cohort
from neuroedge.context_models import ClassicalPredictor
from neuroedge.context_neural import fit_neural
from neuroedge.context_protocol import freeze,grid,load_plan,score,splits


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',default='configs/context_fusion.yaml')
    args=parser.parse_args();plan=load_plan(args.plan);out,identity=freeze(plan)
    if (out/'diagnostics/controls.json').exists():print('Source controls already complete.');return
    cohort=load_cohort(plan,'development_data');selected=[]
    for person in plan['source_subjects'][:4]:
        for label in [0,1]:selected.extend(np.flatnonzero((cohort.meta.subject==person)&(cohort.meta.label==label))[:4].tolist())
    rows=np.asarray(selected);assert len(rows)==32 and set(cohort.meta.iloc[rows].subject)<=set(plan['source_subjects'])
    diagnostic=copy.deepcopy(plan);diagnostic['neural']['dropout']=0.;diagnostic['neural']['learning_rate']=.003
    memorization=[];torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for context,symmetry in [(False,'none'),(True,'none'),(True,'equivariant')]:
            spec=copy.deepcopy(next(s for s in grid() if s.get('architecture')=='filterbank_variance' and s['context']==context and s['symmetry']==symmetry and not s['dro']))
            spec['epochs']=150;directory=out/'diagnostics'/f'capacity_context_{context}_{symmetry}'
            predictor=fit_neural(cohort,rows,spec,diagnostic,directory,7);p=predictor.predict_proba(cohort,rows)
            memorization.append({'context':context,'symmetry':symmetry,'training_accuracy':float(np.mean(p.argmax(axis=1)==cohort.meta.iloc[rows].label.to_numpy())),
                'training_log_loss':float(-np.log(np.clip(p[np.arange(len(p)),cohort.meta.iloc[rows].label.to_numpy()],1e-7,1)).mean()),
                'examples':32,'held_out_examples':0,'epochs':150,'dropout':0.,'learning_rate':.003})
            print('SOURCE MEMORIZATION ONLY',memorization[-1],flush=True)
        spec=next(s for s in grid() if s['family']=='power' and s['mode']=='erd' and s['classifier']=='logistic' and s['C']==.1 and s['symmetry']=='equivariant')
        shuffled=[]
        for seed in range(10):
            changed=copy.copy(cohort);changed.meta=cohort.meta.copy();rng=np.random.default_rng(20261101+seed)
            for person in plan['source_subjects']:
                indices=np.flatnonzero(changed.meta.subject==person)
                changed.meta.loc[indices,'label']=rng.permutation(changed.meta.loc[indices,'label'].to_numpy())
            predictions=[];metas=[]
            for split in splits(plan):
                train=np.flatnonzero(changed.meta.subject.isin(split['train_subjects']))
                held=np.flatnonzero(changed.meta.subject.isin(split['validation_subjects']))
                model=ClassicalPredictor(spec).fit(changed,train);predictions.append(model.predict_proba(changed,held));metas.append(changed.meta.iloc[held])
            import pandas as pd
            result=score(pd.concat(metas,ignore_index=True),np.concatenate(predictions))
            shuffled.append({'permutation_seed':20261101+seed,**result})
            print('SOURCE LABEL SHUFFLE',seed,result,flush=True)
    json_write(out/'diagnostics/controls.json',{'protocol_sha256':identity['sha256'],'script_sha256':sha256(__file__),
        'scope':'Only already-inspected PhysioNet source people; no reused selection or external participants. Capacity accuracy is on the SAME training examples, not generalization.',
        'capacity_train_trial_ids_sha256':digest(cohort.meta.iloc[rows].trial_id.tolist()),'memorization':memorization,
        'within_participant_permutations':shuffled,'permuted_group_cv_mean_balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in shuffled]))})


if __name__=='__main__':main()
