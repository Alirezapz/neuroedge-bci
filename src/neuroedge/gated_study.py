"""Source-only quality-gate follow-up; parent methods and results are immutable."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch
import yaml

from .config import digest,json_write,sha256
from .context_protocol import freeze as parent_freeze,load_plan as parent_plan,splits,now,predictions,score,interval
from .context_data import load_cohort
from .context_models import fit_temperature,apply_temperature
from .context_study import summarize
from .gated_neural import fit_models,predict_saved


def load_plan(path):
    path=Path(path).resolve();p=yaml.safe_load(path.read_text());p['_root']=str(path.parent.parent)
    parent=parent_plan(Path(p['_root'])/p['parent_plan']);p['_parent']=parent
    if p['variants']!=['fixed_gate','quality_gate']:raise ValueError('This comparison requires both prespecified gate variants.')
    p['_settings']={'neural':parent['neural'],'gate_hidden_units':p['gate_hidden_units'],'gate_initial_bias':p['gate_initial_bias']}
    return p


def specs(plan):
    return [{'name':'gated_'+variant,'family':'gated_neural','architecture':'filterbank_variance','variant':variant,
             'context':True,'symmetry':'equivariant','epochs':plan['epochs'],'dro':False} for variant in plan['variants']]


def freeze(plan):
    root=Path(plan['_root']);parent,identity=parent_freeze(plan['_parent']);out=root/plan['output_dir'];out.mkdir(parents=True,exist_ok=True)
    current={'plan':{k:v for k,v in plan.items() if not k.startswith('_')},'parent_protocol_sha256':identity['sha256'],
             'sources':{n:sha256(root/'src/neuroedge'/n) for n in ['gated_neural.py','gated_study.py']},'candidates':specs(plan)}
    current['sha256']=digest(current);path=out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())['sha256']!=current['sha256']:raise RuntimeError('Frozen gated methods differ; use a new output.')
    else:
        if list((root/plan['_parent']['external']['raw_dir']).glob('*.mat')):raise RuntimeError('Register gated hypotheses before external acquisition.')
        json_write(path,{**current,'frozen_utc':now(),'frozen_before_external_acquisition':True})
    return out,current


def source(plan,variant):
    out,identity=freeze(plan);cohort=load_cohort(plan['_parent'],'development_data')
    spec=next(s for s in specs(plan) if s['variant']==variant);directory=out/'candidates'/variant
    if (directory/'source_complete.json').exists():return
    frames=[];torch.set_num_threads(plan['_parent']['search']['threads'])
    with threadpool_limits(limits=plan['_parent']['search']['threads']):
        for split in splits(plan['_parent']):
            train=np.flatnonzero(cohort.meta.subject.isin(split['train_subjects']));held=np.flatnonzero(cohort.meta.subject.isin(split['validation_subjects']))
            current=directory/'source_folds'/f"fold_{split['fold']}"
            print('Gated source OOF',variant,split['fold'],flush=True)
            fit_models(cohort,train,spec,plan['_settings'],current,[7]);p=predict_saved(cohort,held,spec,current)
            f=predictions(cohort.meta.iloc[held].reset_index(drop=True),p);f['fold']=split['fold'];frames.append(f)
        f=pd.concat(frames,ignore_index=True).sort_values('trial_id').reset_index(drop=True);f.to_csv(directory/'source_predictions.csv',index=False)
        T=fit_temperature(f.prob_right,f.label)
        json_write(directory/'source_metrics.json',{**score(f,np.column_stack([1-f.prob_right,f.prob_right])),'temperature':T})
        train=np.flatnonzero(cohort.meta.subject.isin(plan['_parent']['source_subjects']));held=np.flatnonzero(cohort.meta.subject.isin(plan['_parent']['selection_subjects']))
        fit_models(cohort,train,spec,plan['_settings'],directory/'validation_models',plan['seeds'])
        raw=predict_saved(cohort,held,spec,directory/'validation_models');p=apply_temperature(raw[:,1],T)
        f=predictions(cohort.meta.iloc[held].reset_index(drop=True),p);f['raw_prob_right']=raw[:,1];f.to_csv(directory/'validation_predictions.csv',index=False)
        validated={'name':spec['name'],'variant':variant,'spec':spec,'temperature':T,**score(f,p),
                   'source_prediction_sha256':sha256(directory/'source_predictions.csv'),'validation_prediction_sha256':sha256(directory/'validation_predictions.csv')}
        json_write(directory/'validation_metrics.json',validated)
        fit_models(cohort,np.arange(len(cohort.meta)),spec,plan['_settings'],out/'final_models'/variant,plan['seeds'])
    json_write(directory/'source_complete.json',{'protocol_sha256':identity['sha256'],'validation':validated,
        'final_model_info_sha256':sha256(out/'final_models'/variant/'model_info.json')})
    print('GATED SOURCE COMPLETE',variant,validated,flush=True)


def freeze_selection(plan):
    out,identity=freeze(plan);results=[]
    if list((Path(plan['_root'])/plan['_parent']['external']['raw_dir']).glob('*.mat')) and not (out/'selection.json').exists():
        raise RuntimeError('Do not select gated variants after acquiring external data.')
    for variant in plan['variants']:
        r=json.loads((out/'candidates'/variant/'source_complete.json').read_text())
        if r['protocol_sha256']!=identity['sha256'] or r['final_model_info_sha256']!=sha256(out/'final_models'/variant/'model_info.json'):raise AssertionError('Gated source identity changed.')
        results.append(r['validation'])
    selected=sorted(results,key=lambda r:(-r['balanced_accuracy'],r['log_loss'],r['name']))[0]
    selection={'protocol_sha256':identity['sha256'],'selected':selected,'all_variants_reported_in_transfer':plan['variants'],
               'external_used_for_selection':False}
    p=out/'selection.json'
    if p.exists():
        if json.loads(p.read_text())!=selection:raise AssertionError('Frozen gated selection changed.')
    else:json_write(p,{**selection})
    frozen={'selection_sha256':sha256(p),'model_info_sha256':{v:sha256(out/'final_models'/v/'model_info.json') for v in plan['variants']}}
    frozen_path=out/'final_models_frozen.json'
    if frozen_path.exists():
        previous=json.loads(frozen_path.read_text())
        if any(previous[k]!=v for k,v in frozen.items()):raise AssertionError('Gated final models changed.')
    else:json_write(frozen_path,{**frozen,'frozen_utc':now()})
    return selection


def external(plan):
    out,identity=freeze(plan);selection=json.loads((out/'selection.json').read_text())
    frozen=json.loads((out/'final_models_frozen.json').read_text())
    if frozen['selection_sha256']!=sha256(out/'selection.json'):raise AssertionError('Gated selection changed before external evaluation.')
    for variant,value in frozen['model_info_sha256'].items():
        if sha256(out/'final_models'/variant/'model_info.json')!=value:raise AssertionError('Gated source model changed.')
    cohort=load_cohort(plan['_parent'],'external_data');all_rows=np.arange(len(cohort.meta));lookup={s['variant']:s for s in specs(plan)}
    bootstrap={'evaluation':{'bootstrap_seed':plan['bootstrap_seed'],'bootstrap_iterations':plan['bootstrap_iterations']}}
    receipt={'protocol_sha256':identity['sha256'],'selection_sha256':sha256(out/'selection.json'),'external_used_for_selection':False}
    torch.set_num_threads(plan['_parent']['search']['threads'])
    with threadpool_limits(limits=plan['_parent']['search']['threads']):
        if not (out/'external_transfer/complete.json').exists():
            frames=[]
            for variant,spec in lookup.items():
                metrics=json.loads((out/'candidates'/variant/'validation_metrics.json').read_text())
                raw,gates=predict_saved(cohort,all_rows,spec,out/'final_models'/variant,True);p=apply_temperature(raw[:,1],metrics['temperature'])
                f=predictions(cohort.meta,p);f['model']=variant;f['arm']='pre_cue';frames.append(f)
                pd.DataFrame({'trial_id':cohort.meta.trial_id,'subject':cohort.meta.subject,'mean_context_gate':gates.mean(axis=1),
                              'gate_std_across_features':gates.std(axis=1)}).to_csv(out/f'{variant}_external_gates.csv',index=False)
            summarize(bootstrap,pd.concat(frames,ignore_index=True),out/'external_transfer',receipt)
        variant=selection['selected']['variant'];spec=lookup[variant];T=selection['selected']['temperature'];frames=[]
        for person in sorted(cohort.meta.subject.unique()):
            directory=out/'external_loso'/'folds'/f'S{person:04d}'
            train=np.flatnonzero(cohort.meta.subject!=person);held=np.flatnonzero(cohort.meta.subject==person)
            if (directory/'complete.json').exists():
                r=json.loads((directory/'complete.json').read_text())
                if sha256(directory/'predictions.csv')!=r['prediction_sha256']:raise RuntimeError('Gated LOSO predictions changed.')
                f=pd.read_csv(directory/'predictions.csv')
            else:
                print('Gated external LOSO',person,variant,flush=True)
                fit_models(cohort,train,spec,plan['_settings'],directory/'model',plan['seeds'])
                raw=predict_saved(cohort,held,spec,directory/'model');p=apply_temperature(raw[:,1],T)
                f=predictions(cohort.meta.iloc[held].reset_index(drop=True),p);f['model']='selected_fusion';f['arm']='pre_cue'
                summarize(bootstrap,f,directory,{**receipt,'held_subject':int(person)})
            frames.append(f)
        if not (out/'external_loso/complete.json').exists():summarize(bootstrap,pd.concat(frames,ignore_index=True),out/'external_loso',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['register','source','freeze-selection','external','verify','report'])
    parser.add_argument('--variant',choices=['fixed_gate','quality_gate']);parser.add_argument('--plan',default='configs/gated_context.yaml')
    args=parser.parse_args();plan=load_plan(args.plan)
    if args.command=='register':print(freeze(plan)[0])
    elif args.command=='source':
        if args.variant is None:parser.error('Choose a source variant.')
        source(plan,args.variant)
    elif args.command=='freeze-selection':print(freeze_selection(plan))
    elif args.command=='external':external(plan)
    elif args.command=='verify':
        from .gated_verify import verify
        verify(plan)
    elif args.command=='report':
        from .gated_report import report
        report(plan)


if __name__=='__main__':main()
