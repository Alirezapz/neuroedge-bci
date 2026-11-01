"""Verify direct transfer alone; never mark the unfinished LOSO study complete."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from neuroedge.config import json_write,sha256
from neuroedge.context_data import load_cohort
from neuroedge.context_protocol import freeze,grid,load_plan
from neuroedge.context_study import members,seed_policy,predict_base,evaluate_pipelines
from neuroedge.context_verify import check_fit,compare_predictions,verify_frame,check_summary

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',default='configs/context_fusion.yaml')
args=parser.parse_args();plan=load_plan(args.plan);out,identity=freeze(plan)
selection=json.loads((out/'selection.json').read_text());frozen=json.loads((out/'final_models_frozen.json').read_text())
assert frozen['selection_sha256']==sha256(out/'selection.json') and frozen['protocol_sha256']==identity['sha256']
development=load_cohort(plan,'development_data');external=load_cohort(plan,'external_data')
assert not set(development.meta.subject)&set(external.meta.subject) and development.names==external.names
specs={s['name']:s for s in grid()};rows=np.arange(len(external.meta));components={};seeds=0
torch.set_num_threads(plan['search']['threads'])
with threadpool_limits(limits=plan['search']['threads']):
    for name in members(selection):
        directory=out/'final_models'/name;expected_seeds=seed_policy(specs[name],plan)
        assert sha256(directory/'model_info.json')==frozen['model_info_sha256'][name]
        check_fit(plan,development,np.arange(len(development.meta)),specs[name],directory,expected_seeds)
        components[name]=predict_base(external,rows,specs[name],directory)
        if specs[name]['family']=='neural':seeds+=len(expected_seeds)
    expected=evaluate_pipelines(plan,out,selection,external,rows,components)
actual=pd.read_csv(out/'external_transfer/predictions.csv');compare_predictions(actual,expected)
for model,frame in actual.groupby('model'):verify_frame(frame,external.meta)
check_summary(plan,actual,out/'external_transfer')
complete=json.loads((out/'external_transfer/complete.json').read_text())
assert complete['prediction_sha256']==sha256(out/'external_transfer/predictions.csv')
result={'status':'PASS','scope':'Direct source-to-external transfer only; full-study LOSO verification remains separate.',
    'source_people':len(set(development.meta.subject)),'external_people':len(set(external.meta.subject)),
    'external_trials':len(external.meta),'final_constituents_reinferred':len(components),'neural_repetitions_reinferred':seeds,
    'prediction_sha256':complete['prediction_sha256'],'source_model_freeze_sha256':sha256(out/'final_models_frozen.json')}
json_write(out/'external_transfer/verification.json',result);print(result,flush=True)
