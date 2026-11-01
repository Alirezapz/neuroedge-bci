"""Pre-cue contrast, reflection and source-only fusion; frozen external testing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import torch

from .config import digest,environment,json_write,sha256
from .context_data import load_cohort,prepare_development,prepare_external
from .context_models import ClassicalPredictor,apply_temperature,fit_temperature,load_classical
from .context_neural import fit_neural,load_neural
from .context_protocol import binary_probabilities,freeze,grid,interval,load_plan,now,predictions,score,splits,subject_scores


def reference_names():
    specs=grid()
    return [next(s['name'] for s in specs if s['family']=='csp_reference'),
            next(s['name'] for s in specs if s.get('architecture')=='eegnet_reference')]


def seed_policy(spec,plan):
    return plan['neural']['final_seeds'] if spec['family']=='neural' and spec['architecture']!='eegnet_reference' else [plan['search']['seed']]


def fit_base(cohort,rows,spec,plan,directory,seeds):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    meta=cohort.meta.iloc[rows]
    expected={'spec':spec,'train_subjects':sorted(set(meta.subject.tolist())),
              'train_trial_ids_sha256':digest(meta.trial_id.tolist()),'seeds':list(seeds)}
    info_path=directory/'model_info.json'
    if info_path.exists():
        info=json.loads(info_path.read_text())
        if any(info[k]!=v for k,v in expected.items()):raise RuntimeError('Completed context fit has different source membership or settings.')
        for filename,value in info['files'].items():
            if sha256(directory/filename)!=value:raise RuntimeError('Saved context model file changed.')
        return info
    begin=time.monotonic()
    if spec['family']=='neural':
        for seed in seeds:
            current=directory/f'seed_{seed}';current.mkdir(exist_ok=True)
            receipt_path=current/'seed_receipt.json';model_path=current/'model.pt'
            expected_seed={**expected,'seed':int(seed)}
            if model_path.exists() and receipt_path.exists():
                receipt=json.loads(receipt_path.read_text())
                if any(receipt[k]!=v for k,v in expected_seed.items()) or receipt['model_sha256']!=sha256(model_path):
                    raise RuntimeError('Completed neural seed identity differs.')
            else:
                fit_neural(cohort,rows,spec,plan,current,seed)
                json_write(receipt_path,{**expected_seed,'model_sha256':sha256(model_path)})
    else:ClassicalPredictor(spec,plan['search']['seed']).fit(cohort,rows).save(directory/'model.joblib')
    files={str(p.relative_to(directory)):sha256(p) for p in directory.rglob('*') if p.is_file() and p.name!='model_info.json'}
    info={**expected,'files':files,'fit_seconds_this_invocation':time.monotonic()-begin,'completed_utc':now()}
    json_write(info_path,info);return info


def predict_base(cohort,rows,spec,directory):
    directory=Path(directory);info=json.loads((directory/'model_info.json').read_text())
    for filename,value in info['files'].items():
        if sha256(directory/filename)!=value:raise RuntimeError('Model file checksum changed before inference.')
    if spec['family']=='neural':
        outputs=[]
        for seed in info['seeds']:
            model,cp=load_neural(directory/f'seed_{seed}/model.pt')
            if cp['spec']!=spec or cp['train_subjects']!=info['train_subjects'] or cp['seed']!=seed:
                raise RuntimeError('Neural checkpoint metadata differs from fit receipt.')
            outputs.append(model.predict_proba(cohort,rows))
        return binary_probabilities(np.mean(outputs,axis=0))
    return load_classical(directory/'model.joblib').predict_proba(cohort,rows)


def rank(rows):return sorted(rows,key=lambda r:(-r['balanced_accuracy'],r['log_loss'],r['name']))


def screen(plan):
    out,identity=freeze(plan)
    if not (out/'environment.json').exists():json_write(out/'environment.json',environment())
    cohort=prepare_development(plan);source=np.isin(cohort.meta.subject,plan['source_subjects'])
    if set(cohort.meta.loc[source,'subject'])!=set(plan['source_subjects']):raise AssertionError('Missing source participants.')
    split_list=splits(plan);json_write(out/'splits.json',split_list)
    results=[];torch.set_num_threads(plan['search']['threads'])
    with threadpool_limits(limits=plan['search']['threads']):
        for spec in grid():
            directory=out/'screening'/spec['name'];complete=directory/'complete.json'
            if complete.exists():
                result=json.loads(complete.read_text())
                if sha256(directory/'predictions.csv')!=result['prediction_sha256']:raise RuntimeError('Completed discovery predictions changed.')
            else:
                frames=[]
                for split in split_list:
                    train=np.flatnonzero(np.isin(cohort.meta.subject,split['train_subjects']))
                    held=np.flatnonzero(np.isin(cohort.meta.subject,split['validation_subjects']))
                    model_directory=directory/'fits'/f"fold_{split['fold']}"
                    print('Source group fit',spec['name'],'fold',split['fold'],flush=True)
                    fit_base(cohort,train,spec,plan,model_directory,[plan['search']['seed']])
                    p=predict_base(cohort,held,spec,model_directory)
                    frame=predictions(cohort.meta.iloc[held].reset_index(drop=True),p);frame['fold']=split['fold'];frames.append(frame)
                frame=pd.concat(frames,ignore_index=True).sort_values('trial_id').reset_index(drop=True)
                if frame.trial_id.duplicated().any() or set(frame.subject)!=set(plan['source_subjects']):raise AssertionError('OOF identity or coverage differs.')
                frame.to_csv(directory/'predictions.csv',index=False)
                temperature=(1. if spec['name'] in reference_names() else fit_temperature(frame.prob_right,frame.label))
                result={'name':spec['name'],'kind':'single','family':spec['family'],'arm':spec['arm'],'spec':spec,
                        **score(frame,np.column_stack([1-frame.prob_right,frame.prob_right])),
                        'source_oof_temperature':temperature,'temperature_training_trial_ids_sha256':digest(frame.trial_id.tolist()),
                        'prediction_sha256':sha256(directory/'predictions.csv'),'protocol_sha256':identity['sha256']}
                json_write(complete,result)
            results.append(result)
            pd.DataFrame([{k:v for k,v in r.items() if k not in {'spec'}} for r in results]).to_csv(out/'screening_summary.csv',index=False)
            json_write(out/'screening_progress.json',{'finished':len(results),'total':len(grid()),'last':spec['name']})
            print(f"Completed {len(results)}/{len(grid())}: {spec['name']} discovery BA={result['balanced_accuracy']:.4f}",flush=True)
    json_write(out/'screening_complete.json',{'candidates':len(results),'group_fits':len(results)*len(split_list),'protocol_sha256':identity['sha256']})


def _combination(kind,members,arm,OOF,y,C=None):
    rule={'kind':kind,'members':list(members),'arm':arm,'input':'raw_constituent_probabilities'};meta=None
    if kind=='weighted':
        def objective(w):
            p=np.clip(OOF@w,1e-6,1-1e-6)
            return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+.005*np.sum(w*w))
        fit=minimize(objective,np.ones(len(members))/len(members),method='SLSQP',bounds=[(0,1)]*len(members),
                     constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':500,'ftol':1e-12})
        if not fit.success:raise RuntimeError('Source-OOF vote fitting failed.')
        rule['weights']=fit.x.tolist()
    elif kind=='stacking':
        rule['C']=C;meta=LogisticRegression(C=C,max_iter=2000,random_state=7).fit(logit(np.clip(OOF,1e-5,1-1e-5)),y)
    return rule,meta


def ensemble_probability(rule,P,meta=None):
    if rule['kind']=='stacking':return binary_probabilities(meta.predict_proba(logit(np.clip(P,1e-5,1-1e-5))))
    right=P@np.asarray(rule['weights']) if rule['kind']=='weighted' else P.mean(axis=1)
    return binary_probabilities(np.column_stack([1-right,right]))


def select(plan):
    out,identity=freeze(plan)
    if (out/'selection.json').exists():return json.loads((out/'selection.json').read_text())
    if (out/'final_models_frozen.json').exists():raise RuntimeError('External workflow already started without its selection file.')
    cohort=load_cohort(plan,'development_data')
    discovery={p.parent.name:json.loads(p.read_text()) for p in (out/'screening').glob('*/complete.json')}
    if set(discovery)!={s['name'] for s in grid()}:raise RuntimeError('Finish every specified discovery candidate first.')
    lists={};shortlist=set(reference_names())
    for arm in plan['evaluation']['operating_conditions']:
        ranked=rank([r for r in discovery.values() if r['arm']==arm]);lists[arm]=ranked
        shortlist.update(r['name'] for r in ranked[:plan['search']['shortlist_global']])
        for family in sorted({r['family'] for r in ranked}):
            shortlist.add(next(r['name'] for r in ranked if r['family']==family))
    rules=[]
    source_meta=cohort.meta[cohort.meta.subject.isin(plan['source_subjects'])].sort_values('trial_id')
    y=source_meta.label.to_numpy(dtype=int)
    def oof(members):
        return np.column_stack([pd.read_csv(out/'screening'/m/'predictions.csv').set_index('trial_id').loc[source_meta.trial_id].prob_right for m in members])
    for arm,ranked in lists.items():
        top=[r['name'] for r in ranked[:5]]
        diverse=list(dict.fromkeys(next(r['name'] for r in ranked if r['family']==f) for f in sorted({r['family'] for r in ranked})))
        configurations=[('mean',top[:n],None) for n in plan['search']['ensemble_top_counts']]
        configurations += [('mean',diverse,None),('weighted',top,None)]
        configurations += [('stacking',top,C) for C in [.01,.1,1.]]
        if arm=='pre_cue':
            hybrid=list(dict.fromkeys(top[:3]+[r['name'] for r in lists['zero_calibration'][:3]]))
            configurations += [('mean',hybrid,None),('weighted',hybrid,None),('stacking',hybrid,.1)]
        seen=set()
        for kind,members,C in configurations:
            key=digest([kind,members,C])
            if len(members)<2 or key in seen:continue
            seen.add(key);shortlist.update(members)
            rule,meta=_combination(kind,members,arm,oof(members),y,C)
            rules.append((rule,meta))
    train=np.flatnonzero(cohort.meta.subject.isin(plan['source_subjects']));held=np.flatnonzero(cohort.meta.subject.isin(plan['selection_subjects']))
    completed={};torch.set_num_threads(plan['search']['threads'])
    with threadpool_limits(limits=plan['search']['threads']):
        for name in sorted(shortlist):
            directory=out/'validation'/name;complete=directory/'complete.json';spec=discovery[name]['spec']
            if complete.exists():result=json.loads(complete.read_text())
            else:
                print('Reused development selection fit',name,flush=True)
                fit_base(cohort,train,spec,plan,directory,seed_policy(spec,plan))
                raw=predict_base(cohort,held,spec,directory);T=discovery[name]['source_oof_temperature']
                p=apply_temperature(raw[:,1],T)
                frame=predictions(cohort.meta.iloc[held].reset_index(drop=True),p);frame['raw_prob_right']=raw[:,1]
                frame.to_csv(directory/'predictions.csv',index=False)
                result={'name':name,'kind':'single','family':spec['family'],'arm':spec['arm'],'spec':spec,
                        'temperature':T,**score(frame,p),'prediction_sha256':sha256(directory/'predictions.csv')}
                json_write(complete,result)
            if sha256(directory/'predictions.csv')!=result['prediction_sha256']:raise RuntimeError('Validation predictions changed.')
            completed[name]=result
    for rule,meta in rules:
        name='ensemble_'+digest(rule)[:12];directory=out/'validation'/name;directory.mkdir(parents=True,exist_ok=True)
        if meta is not None:
            joblib.dump(meta,directory/'meta.joblib');rule={**rule,'meta_sha256':sha256(directory/'meta.joblib')}
        P=np.column_stack([pd.read_csv(out/'validation'/member/'predictions.csv').raw_prob_right for member in rule['members']])
        p=ensemble_probability(rule,P,meta);frame=predictions(cohort.meta.iloc[held].reset_index(drop=True),p)
        frame.to_csv(directory/'predictions.csv',index=False)
        result={'name':name,'kind':'ensemble','family':'ensemble','arm':rule['arm'],'combination':rule,
                **score(frame,p),'prediction_sha256':sha256(directory/'predictions.csv')}
        json_write(directory/'complete.json',result);completed[name]=result
    pd.DataFrame([{k:v for k,v in r.items() if k not in {'spec','combination'}} for r in completed.values()]).to_csv(out/'validation_summary.csv',index=False)
    selected={arm:rank([r for r in completed.values() if r['arm']==arm])[0] for arm in plan['evaluation']['operating_conditions']}
    selection={'protocol_sha256':identity['sha256'],'selected':selected,'frozen_utc':now(),
               'selection_people_are_previously_inspected_development':True,'external_data_used_for_selection':False,
               'reference_names':reference_names(),'external_protocols':plan['evaluation']['external_protocols']}
    json_write(out/'selection.json',selection)
    print('Frozen source-only selections:',json.dumps(selected,ensure_ascii=False),flush=True)
    return selection


def members(selection):
    names=set(selection['reference_names'])
    for winner in selection['selected'].values():names.update([winner['name']] if winner['kind']=='single' else winner['combination']['members'])
    return sorted(names)


def evaluate_pipelines(plan,out,selection,cohort,rows,components):
    current={}
    for label,name in zip(['csp_reference','eegnet_reference'],selection['reference_names']):
        current[label]={'p':components[name],'arm':'zero_calibration'}
    for arm,winner in selection['selected'].items():
        if winner['kind']=='single':p=apply_temperature(components[winner['name']][:,1],winner['temperature'])
        else:
            rule=winner['combination'];path=out/'validation'/winner['name']/'meta.joblib'
            meta=joblib.load(path) if rule['kind']=='stacking' else None
            if meta is not None and sha256(path)!=rule['meta_sha256']:raise RuntimeError('Frozen stacking model changed.')
            p=ensemble_probability(rule,np.column_stack([components[m][:,1] for m in rule['members']]),meta)
        current['selected_'+arm]={'p':p,'arm':arm}
    frames=[]
    for name,item in current.items():
        frame=predictions(cohort.meta.iloc[rows].reset_index(drop=True),item['p']);frame['model']=name;frame['arm']=item['arm'];frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def summarize(plan,frame,directory,receipt):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    frame.to_csv(directory/'predictions.csv',index=False);summaries=[];people=[]
    for name,current in frame.groupby('model',sort=False):
        p=np.column_stack([1-current.prob_right,current.prob_right]);scores=subject_scores(current.reset_index(drop=True),p)
        scores['model']=name;scores['arm']=current.arm.iloc[0];people.append(scores)
        low,high=interval(scores.balanced_accuracy,plan)
        summaries.append({'model':name,'arm':current.arm.iloc[0],'subjects':len(scores),
                          **score(current.reset_index(drop=True),p),'ci_low':low,'ci_high':high})
    pd.concat(people,ignore_index=True).to_csv(directory/'metrics_by_subject.csv',index=False)
    pd.DataFrame(summaries).to_csv(directory/'summary.csv',index=False)
    json_write(directory/'complete.json',{**receipt,'prediction_sha256':sha256(directory/'predictions.csv'),
                                          'participants':int(frame.subject.nunique()),'trials':int(frame.trial_id.nunique())})
    print(directory.name,flush=True);print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def confirm(plan):
    out,identity=freeze(plan);selection=json.loads((out/'selection.json').read_text());selection_sha=sha256(out/'selection.json')
    cohort=load_cohort(plan,'development_data');specs={s['name']:s for s in grid()};names=members(selection)
    torch.set_num_threads(plan['search']['threads'])
    with threadpool_limits(limits=plan['search']['threads']):
        for name in names:
            print('Final source-only fit',name,flush=True)
            fit_base(cohort,np.arange(len(cohort.meta)),specs[name],plan,out/'final_models'/name,seed_policy(specs[name],plan))
    frozen={'selection_sha256':selection_sha,'protocol_sha256':identity['sha256'],
            'model_info_sha256':{name:sha256(out/'final_models'/name/'model_info.json') for name in names},
            'frozen_before_external_acquisition':True}
    frozen_path=out/'final_models_frozen.json'
    if frozen_path.exists():
        if json.loads(frozen_path.read_text())!=frozen:raise RuntimeError('Final source models changed.')
    else:json_write(frozen_path,{**frozen})
    external=prepare_external(plan);all_rows=np.arange(len(external.meta))
    receipt={'selection_sha256':selection_sha,'protocol_sha256':identity['sha256'],'external_used_for_selection':False}
    with threadpool_limits(limits=plan['search']['threads']):
        if not (out/'external_transfer/complete.json').exists():
            components={name:predict_base(external,all_rows,specs[name],out/'final_models'/name) for name in names}
            frame=evaluate_pipelines(plan,out,selection,external,all_rows,components)
            summarize(plan,frame,out/'external_transfer',{**receipt,'training_condition':'104 PhysioNet people, no BNCI task training'})
        folds=[]
        for person in sorted(external.meta.subject.unique()):
            directory=out/'external_loso'/'folds'/f'S{person:04d}'
            held=np.flatnonzero(external.meta.subject==person);train=np.flatnonzero(external.meta.subject!=person)
            if (directory/'complete.json').exists():
                complete=json.loads((directory/'complete.json').read_text())
                if sha256(directory/'predictions.csv')!=complete['prediction_sha256']:raise RuntimeError('External LOSO predictions changed.')
                frame=pd.read_csv(directory/'predictions.csv')
            else:
                components={}
                for name in names:
                    print(f'Frozen external LOSO held person {person}: {name}',flush=True)
                    fit_base(external,train,specs[name],plan,directory/'models'/name,seed_policy(specs[name],plan))
                    components[name]=predict_base(external,held,specs[name],directory/'models'/name)
                frame=evaluate_pipelines(plan,out,selection,external,held,components)
                summarize(plan,frame,directory,{**receipt,'held_subject':int(person),'train_subjects':sorted(set(external.meta.iloc[train].subject.tolist()))})
            folds.append(frame)
        if not (out/'external_loso/complete.json').exists():
            summarize(plan,pd.concat(folds,ignore_index=True),out/'external_loso',
                      {**receipt,'training_condition':'8 other BNCI people per fold; both sessions of held person excluded; no BNCI hyperparameter or combination selection'})
    if sha256(out/'selection.json')!=selection_sha:raise AssertionError('Selection changed during external evaluation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['all','screen','select','confirm','verify','report'])
    parser.add_argument('--plan',default='configs/context_fusion.yaml');args=parser.parse_args();plan=load_plan(args.plan)
    if args.command in {'all','screen'}:screen(plan)
    if args.command in {'all','select'}:select(plan)
    if args.command in {'all','confirm'}:confirm(plan)
    if args.command in {'all','verify'}:
        from .context_verify import verify
        verify(plan)
    if args.command in {'all','report'}:
        from .context_report import report
        report(plan)


if __name__=='__main__':
    from neuroedge.context_study import main as canonical_main
    canonical_main()
