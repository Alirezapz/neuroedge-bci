"""Schedule disjoint, distant frozen validation/final fits or LOSO folds.

Use only a destination far beyond the main sequential cursor. This calls the
canonical frozen fit functions and changes no scientific setting or candidate.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from neuroedge.config import digest,json_write,sha256
from neuroedge.context_data import load_cohort
from neuroedge.context_models import apply_temperature
from neuroedge.context_neural import fit_neural
from neuroedge.context_protocol import freeze,grid,load_plan,predictions,score
from neuroedge.context_study import (fit_base,predict_base,seed_policy,members,
                                    evaluate_pipelines,summarize)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['validation','validation-seed','final','loso'])
    parser.add_argument('--plan',default='configs/context_fusion.yaml')
    parser.add_argument('--names',nargs='+',default=[])
    parser.add_argument('--subjects',type=int,nargs='+',default=[])
    parser.add_argument('--seed',type=int)
    parser.add_argument('--resume',action='store_true',help='Resume an explicitly stopped destination; caller must exclude another writer.')
    args=parser.parse_args();plan=load_plan(args.plan);out,identity=freeze(plan)
    specs={s['name']:s for s in grid()}
    selection=None;selection_sha=None
    if args.mode not in ['validation','validation-seed']:
        selection=json.loads((out/'selection.json').read_text());selection_sha=sha256(out/'selection.json')
    torch.set_num_threads(plan['search']['threads'])
    with threadpool_limits(limits=plan['search']['threads']):
        if args.mode=='validation-seed':
            cohort=load_cohort(plan,'development_data');train=np.flatnonzero(cohort.meta.subject.isin(plan['source_subjects']))
            meta=cohort.meta.iloc[train]
            for name in args.names:
                spec=specs[name];seeds=seed_policy(spec,plan)
                if spec['family']!='neural' or args.seed not in seeds:raise RuntimeError('Only a registered neural repetition is allowed.')
                directory=out/'validation'/name/f'seed_{args.seed}'
                if directory.exists():raise RuntimeError('Seed already started: never race its writer.')
                directory.mkdir(parents=True)
                expected={'spec':spec,'train_subjects':sorted(set(meta.subject.tolist())),
                          'train_trial_ids_sha256':digest(meta.trial_id.tolist()),'seeds':list(seeds),'seed':args.seed}
                print('Spare CPU distant validation seed',name,args.seed,flush=True)
                fit_neural(cohort,train,spec,plan,directory,args.seed)
                json_write(directory/'seed_receipt.json',{**expected,'model_sha256':sha256(directory/'model.pt')})
                json_write(out/'scheduling'/f'validation_{name}_seed_{args.seed}.json',{'worker_sha256':sha256(__file__),
                    'protocol_sha256':identity['sha256'],'note':'One distant fixed repetition; canonical fitting and seed receipt, no scientific change.'})
        elif args.mode in ['final','validation']:
            cohort=load_cohort(plan,'development_data')
            allowed=members(selection) if selection else list(specs)
            if not args.names or set(args.names)-set(allowed):raise RuntimeError('Unregistered constituent.')
            if (out/'final_models_frozen.json').exists():raise RuntimeError('Final source fits already frozen.')
            for name in args.names:
                directory=out/('final_models' if args.mode=='final' else 'validation')/name
                if directory.exists() and not args.resume:raise RuntimeError('Destination already started: do not race another process.')
                if args.mode=='validation' and (directory/'complete.json').exists():continue
                train=np.arange(len(cohort.meta)) if args.mode=='final' else np.flatnonzero(cohort.meta.subject.isin(plan['source_subjects']))
                print('Spare CPU',args.mode,'source fit',name,flush=True)
                fit_base(cohort,train,specs[name],plan,directory,seed_policy(specs[name],plan))
                if args.mode=='validation':
                    discovery=json.loads((out/'screening'/name/'complete.json').read_text())
                    held=np.flatnonzero(cohort.meta.subject.isin(plan['selection_subjects']))
                    raw=predict_base(cohort,held,specs[name],directory);T=discovery['source_oof_temperature']
                    p=apply_temperature(raw[:,1],T)
                    frame=predictions(cohort.meta.iloc[held].reset_index(drop=True),p);frame['raw_prob_right']=raw[:,1]
                    frame.to_csv(directory/'predictions.csv',index=False)
                    result={'name':name,'kind':'single','family':specs[name]['family'],'arm':specs[name]['arm'],
                            'spec':specs[name],'temperature':T,**score(frame,p),'prediction_sha256':sha256(directory/'predictions.csv')}
                    json_write(directory/'complete.json',result)
                json_write(out/'scheduling'/f'{args.mode}_{name}.json',{'worker_sha256':sha256(__file__),
                    'selection_sha256':selection_sha,'protocol_sha256':identity['sha256'],
                    'note':'Canonical frozen fit; an independent process changes scheduling only.'})
        else:
            cohort=load_cohort(plan,'external_data');names=members(selection)
            if not args.subjects or set(args.subjects)-set(cohort.meta.subject):raise RuntimeError('Unknown external person.')
            for person in args.subjects:
                directory=out/'external_loso'/'folds'/f'S{person:04d}'
                if (directory/'complete.json').exists():continue
                if directory.exists() and not args.resume:raise RuntimeError('Destination already started: do not race another process.')
                held=np.flatnonzero(cohort.meta.subject==person);train=np.flatnonzero(cohort.meta.subject!=person)
                components={}
                for name in names:
                    print('Spare CPU frozen LOSO',person,name,flush=True)
                    fit_base(cohort,train,specs[name],plan,directory/'models'/name,seed_policy(specs[name],plan))
                    components[name]=predict_base(cohort,held,specs[name],directory/'models'/name)
                frame=evaluate_pipelines(plan,out,selection,cohort,held,components)
                summarize(plan,frame,directory,{'selection_sha256':selection_sha,'protocol_sha256':identity['sha256'],
                    'external_used_for_selection':False,'held_subject':int(person),
                    'train_subjects':sorted(set(cohort.meta.iloc[train].subject.tolist()))})
                json_write(out/'scheduling'/f'S{person:04d}.json',{'worker_sha256':sha256(__file__),
                    'selection_sha256':selection_sha,'protocol_sha256':identity['sha256'],
                    'note':'Canonical frozen LOSO fit; an independent process changes scheduling only.'})
        if selection_sha and sha256(out/'selection.json')!=selection_sha:raise AssertionError('Selection changed.')


if __name__=='__main__':main()
