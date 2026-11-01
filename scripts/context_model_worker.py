"""Precompute one distant LOSO constituent using the frozen canonical fitter.

Use only a destination that the main queue has not started and will reach well
after this job finishes. This changes CPU scheduling, never model selection.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from neuroedge.config import json_write,sha256
from neuroedge.context_data import load_cohort
from neuroedge.context_protocol import freeze,grid,load_plan,now
from neuroedge.context_study import fit_base,members,seed_policy

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan',default='configs/context_fusion.yaml')
parser.add_argument('--subject',type=int,required=True)
parser.add_argument('--name',required=True)
args=parser.parse_args();plan=load_plan(args.plan);out,identity=freeze(plan)
selection=json.loads((out/'selection.json').read_text());specs={s['name']:s for s in grid()}
if args.name not in members(selection) or specs[args.name]['family']!='neural':raise RuntimeError('Only a frozen selected neural constituent is allowed.')
if not (out/'final_models_frozen.json').exists():raise RuntimeError('Freeze final source models before external fitting.')
cohort=load_cohort(plan,'external_data')
if args.subject not in set(cohort.meta.subject):raise RuntimeError('Unknown held person.')
directory=out/'external_loso/folds'/f'S{args.subject:04d}'/'models'/args.name
if directory.exists():raise RuntimeError('Constituent already started; never race its writer.')
train=np.flatnonzero(cohort.meta.subject!=args.subject);torch.set_num_threads(plan['search']['threads'])
with threadpool_limits(limits=plan['search']['threads']):
    fit_base(cohort,train,specs[args.name],plan,directory,seed_policy(specs[args.name],plan))
json_write(out/'scheduling'/f'constituent_{args.subject}_{args.name}.json',{
    'worker_sha256':sha256(__file__),'protocol_sha256':identity['sha256'],
    'selection_sha256':sha256(out/'selection.json'),'completed_utc':now(),
    'note':'One distant LOSO constituent, canonical method and frozen source membership; scheduling only.'})
print('Distant constituent complete:',args.subject,args.name,flush=True)
