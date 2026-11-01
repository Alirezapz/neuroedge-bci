"""Locally frozen, source-only development followed by cross-dataset testing."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
import yaml

from .config import digest, json_write, sha256
from .frontier_protocol import binary_probabilities, predictions as original_predictions, score, subject_scores

ROOT = Path(__file__).resolve().parents[2]
CRITICAL = ['context_protocol.py','context_data.py','context_models.py',
            'context_neural.py','context_study.py','data.py','models.py',
            'frontier_data.py','frontier_protocol.py','config.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def predictions(meta,probabilities):
    result=original_predictions(meta,probabilities)
    for column in meta.columns:
        if column not in result:result[column]=meta.reset_index(drop=True)[column]
    return result


def grid():
    result=[]
    def add(**spec):
        spec['arm']='pre_cue' if spec.get('context',False) else 'zero_calibration'
        spec['name']=spec['family']+'_'+digest(spec)[:12]
        result.append(spec)
    for mode in ['active','erd','hybrid']:
        for classifier in ['logistic','svm']:
            for C in [.1,1.]:
                for symmetry in ['none','equivariant']:
                    add(family='power',mode=mode,context=mode!='active',classifier=classifier,C=C,symmetry=symmetry)
        add(family='power',mode=mode,context=mode!='active',classifier='logistic',C=1.,symmetry='flip')
    for family in ['fbcsp','tangent']:
        for mode in ['active','erd','hybrid']:
            for C in [.1,1.]:
                for symmetry in ['none','equivariant']:
                    add(family=family,mode=mode,context=mode!='active',classifier='logistic',C=C,symmetry=symmetry,shrinkage=.1)
    add(family='csp_reference',context=False,symmetry='none')
    add(family='neural',architecture='eegnet_reference',context=False,symmetry='none',epochs=25,dro=False)
    for symmetry in ['none','flip']:
        add(family='neural',architecture='eegnet',context=False,symmetry=symmetry,epochs=35,dro=False)
    for context in [False,True]:
        for symmetry in ['none','flip','equivariant']:
            add(family='neural',architecture='filterbank_variance',context=context,symmetry=symmetry,epochs=35,dro=False)
    for symmetry in ['none','equivariant']:
        add(family='neural',architecture='filterbank_variance',context=True,symmetry=symmetry,epochs=35,dro=True)
    assert len(result)==63 and len({s['name'] for s in result})==63
    return result


def load_plan(path):
    path=Path(path).resolve()
    plan=yaml.safe_load(path.read_text())
    plan['_root']=str(path.parent.parent)
    source=set(plan['source_subjects']);selection=set(plan['selection_subjects'])
    if source & selection or not source or not selection or len(source)!=len(plan['source_subjects']) or len(selection)!=len(plan['selection_subjects']):
        raise ValueError('Source and reused selection participants must be disjoint.')
    if plan['data']['task_window']!=[.5,4.] or plan['data']['pre_cue_window']!=[-1.75,-.25] or plan['data']['sfreq']!=128:
        raise ValueError('This implementation has fixed task/context windows and sampling rate.')
    if plan['data']['runs']!=[4,8,12] or plan['external']['cue_offset_seconds']!=2. or plan['external']['sessions']!=['T','E']:
        raise ValueError('Unsupported task mapping or external timing.')
    if (source|selection)&set(plan['data']['prior_structural_exclusions']) or any(s<1 or s>109 for s in source|selection):
        raise ValueError('Unexpected PhysioNet participant set.')
    if plan['external']['dataset']!='BNCI2014_001' or plan['external']['subjects']!=list(range(1,10)):
        raise ValueError('This protocol requires all nine fixed independent dataset participants.')
    return plan


def freeze(plan):
    root=Path(plan['_root']);out=root/plan['output_dir'];out.mkdir(parents=True,exist_ok=True)
    clean={k:v for k,v in plan.items() if not k.startswith('_')}
    source={name:sha256(root/'src/neuroedge'/name) for name in CRITICAL}
    identity={'plan':clean,'candidates':grid(),'scientific_source_sha256':source}
    identity['sha256']=digest(identity)
    path=out/'protocol.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['sha256']!=identity['sha256']:
            raise RuntimeError('Frozen context-study methods changed. Preserve this run and use a new output name.')
    else:
        identity['frozen_utc']=now();json_write(path,identity)
        json_write(out/'candidates.json',grid())
    return out,identity


def splits(plan):
    people=np.asarray(sorted(plan['source_subjects']))
    result=[]
    for fold,(train,test) in enumerate(KFold(plan['search']['folds'],shuffle=True,
                                 random_state=plan['search']['split_seed']).split(people)):
        result.append({'fold':fold,'train_subjects':people[train].tolist(),
                       'validation_subjects':people[test].tolist()})
    return result


def interval(values,plan):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(plan['evaluation']['bootstrap_seed'])
    means=np.mean(values[rng.integers(0,len(values),size=(plan['evaluation']['bootstrap_iterations'],len(values)))],axis=1)
    return [float(v) for v in np.quantile(means,[.025,.975])]
