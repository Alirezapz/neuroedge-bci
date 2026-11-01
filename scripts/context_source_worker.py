"""Schedule specified FAR-AHEAD source candidates with the frozen fit functions.

No method changes, extra candidates or external data. One optional CPU worker;
use only candidates several positions beyond the main sequential cursor.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from neuroedge.config import digest,json_write,sha256
from neuroedge.context_protocol import freeze,grid,load_plan,predictions,score,splits
from neuroedge.context_data import load_cohort
from neuroedge.context_models import fit_temperature
from neuroedge.context_study import fit_base,predict_base,reference_names


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',default='configs/context_fusion.yaml')
    parser.add_argument('--indices',type=int,nargs='+',required=True)
    args=parser.parse_args();plan=load_plan(args.plan);out,identity=freeze(plan);cohort=load_cohort(plan,'development_data');specs=grid()
    torch.set_num_threads(plan['search']['threads'])
    with threadpool_limits(limits=plan['search']['threads']):
        for index in args.indices:
            spec=specs[index];directory=out/'screening'/spec['name']
            if (directory/'complete.json').exists():continue
            progress=json.loads((out/'screening_progress.json').read_text())
            if progress['finished']>index-2:raise RuntimeError('Worker candidate is too close to the main cursor; do not race fits.')
            if directory.exists():raise RuntimeError('Candidate already started; do not race source workers.')
            frames=[]
            for split in splits(plan):
                train=np.flatnonzero(cohort.meta.subject.isin(split['train_subjects']))
                held=np.flatnonzero(cohort.meta.subject.isin(split['validation_subjects']))
                current=directory/'fits'/f"fold_{split['fold']}"
                print('Independent source CPU job',index,spec['name'],'fold',split['fold'],flush=True)
                fit_base(cohort,train,spec,plan,current,[plan['search']['seed']])
                p=predict_base(cohort,held,spec,current)
                frame=predictions(cohort.meta.iloc[held].reset_index(drop=True),p);frame['fold']=split['fold'];frames.append(frame)
            frame=pd.concat(frames,ignore_index=True).sort_values('trial_id').reset_index(drop=True)
            if frame.trial_id.duplicated().any() or set(frame.subject)!=set(plan['source_subjects']):raise AssertionError('OOF coverage differs.')
            frame.to_csv(directory/'predictions.csv',index=False)
            T=1. if spec['name'] in reference_names() else fit_temperature(frame.prob_right,frame.label)
            result={'name':spec['name'],'kind':'single','family':spec['family'],'arm':spec['arm'],'spec':spec,
                    **score(frame,np.column_stack([1-frame.prob_right,frame.prob_right])),
                    'source_oof_temperature':T,'temperature_training_trial_ids_sha256':digest(frame.trial_id.tolist()),
                    'prediction_sha256':sha256(directory/'predictions.csv'),'protocol_sha256':identity['sha256']}
            json_write(directory/'worker_receipt.json',{'source_worker_sha256':sha256(__file__),
                'source_protocol_sha256':identity['sha256'],'candidate_index':index,'threads':plan['search']['threads'],
                'note':'Same frozen source fitting/prediction functions; scheduling only, no external data or extra hyperparameters.'})
            json_write(directory/'complete.json',result);print('SOURCE CPU JOB COMPLETE',index,spec['name'],flush=True)


if __name__=='__main__':main()
