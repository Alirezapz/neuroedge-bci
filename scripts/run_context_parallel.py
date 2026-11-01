"""Run the frozen studies with disjoint CPU subprocesses and canonical methods.

Concurrency changes scheduling only. Each training process has its own RNG;
the independent scientific verifier checks every saved external fit/prediction.
Register this execution driver before acquiring external EEG.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')


def call(command,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',buffering=1) as stream:
        subprocess.run([sys.executable,'-u',*command],stdout=stream,stderr=subprocess.STDOUT,check=True)


def dispatch(tasks,jobs):
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures={pool.submit(function,*args):label for label,function,args in tasks}
        for future in as_completed(futures):
            future.result();print('Completed scheduled job:',futures[future],flush=True)


def gate_external(path,output):
    for command in ['external','verify']:
        call(['-m','neuroedge.gated_study',command,'--plan',path],Path(output)/'external_execution_parallel.log')


def register_execution(root,out,context,gated,jobs):
    from neuroedge.config import digest,json_write,sha256
    from neuroedge.context_protocol import now
    rule={'driver_sha256':sha256(__file__),'worker_sha256':sha256(Path(__file__).with_name('context_fit_worker.py')),
          'context_protocol_sha256':json.loads((root/context['output_dir']/'protocol.json').read_text())['sha256'],
          'gated_protocol_sha256':json.loads((root/gated['output_dir']/'protocol.json').read_text())['sha256'],
          'intra_op_threads':context['search']['threads'],'wait_policy':os.environ['OMP_WAIT_POLICY'],
          'rule':'Scheduling only: canonical selected model fits, transfer evaluation and per-person LOSO; separate Python subprocess RNGs; no new hyperparameters, seed, epoch, target data use or selection rule.'}
    rule['sha256']=digest(rule);path=out/'execution_protocol.json'
    if path.exists():
        if json.loads(path.read_text())['sha256']!=rule['sha256']:raise RuntimeError('Execution driver changed; preserve its recorded source rather than overriding its identity.')
    else:
        if list((root/context['external']['raw_dir']).glob('*.mat')):raise RuntimeError('Register this schedule before external EEG acquisition.')
        json_write(path,{**rule,'registered_utc':now(),'registered_before_external_acquisition':True})
    journal=out/'execution_sessions.json';sessions=json.loads(journal.read_text()) if journal.exists() else []
    sessions.append({'started_utc':now(),'parallel_subprocesses':jobs,'execution_protocol_sha256':rule['sha256']})
    json_write(journal,sessions)


def transfer(plan):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from neuroedge.config import json_write,sha256
    from neuroedge.context_data import prepare_external
    from neuroedge.context_protocol import freeze,grid
    from neuroedge.context_study import members,predict_base,evaluate_pipelines,summarize
    out,identity=freeze(plan);selection=json.loads((out/'selection.json').read_text());names=members(selection)
    selection_sha=sha256(out/'selection.json')
    frozen={'selection_sha256':selection_sha,'protocol_sha256':identity['sha256'],
        'model_info_sha256':{name:sha256(out/'final_models'/name/'model_info.json') for name in names},
        'frozen_before_external_acquisition':True}
    path=out/'final_models_frozen.json'
    if path.exists():
        if json.loads(path.read_text())!=frozen:raise RuntimeError('Frozen source models changed.')
    else:
        if list((Path(plan['_root'])/plan['external']['raw_dir']).glob('*.mat')):raise RuntimeError('Source model freeze must precede first external acquisition.')
        json_write(path,frozen)
    external=prepare_external(plan);rows=np.arange(len(external.meta));specs={s['name']:s for s in grid()}
    torch.set_num_threads(plan['search']['threads'])
    if not (out/'external_transfer/complete.json').exists():
        with threadpool_limits(limits=plan['search']['threads']):
            components={name:predict_base(external,rows,specs[name],out/'final_models'/name) for name in names}
            frame=evaluate_pipelines(plan,out,selection,external,rows,components)
            summarize(plan,frame,out/'external_transfer',{'selection_sha256':selection_sha,'protocol_sha256':identity['sha256'],
                'external_used_for_selection':False,'training_condition':'104 PhysioNet people, no BNCI task training'})
    if sha256(out/'selection.json')!=selection_sha:raise AssertionError('Selection changed.')


def aggregate(plan):
    import pandas as pd
    from neuroedge.config import sha256
    from neuroedge.context_data import load_cohort
    from neuroedge.context_protocol import freeze
    from neuroedge.context_study import summarize
    out,identity=freeze(plan);external=load_cohort(plan,'external_data');frames=[]
    for person in sorted(external.meta.subject.unique()):
        directory=out/'external_loso/folds'/f'S{person:04d}'
        receipt=json.loads((directory/'complete.json').read_text())
        if sha256(directory/'predictions.csv')!=receipt['prediction_sha256']:raise RuntimeError('Fold predictions changed.')
        frames.append(pd.read_csv(directory/'predictions.csv'))
    if not (out/'external_loso/complete.json').exists():
        summarize(plan,pd.concat(frames,ignore_index=True),out/'external_loso',{
            'selection_sha256':sha256(out/'selection.json'),'protocol_sha256':identity['sha256'],'external_used_for_selection':False,
            'training_condition':'8 other BNCI people per fold; both sessions of held person excluded; no BNCI hyperparameter or combination selection'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-plan',default='configs/context_fusion.yaml')
    parser.add_argument('--gated-plan',default='configs/gated_context.yaml')
    parser.add_argument('--personal-plan',default='configs/personalization.yaml')
    parser.add_argument('--gated-personal-plan',default='configs/personalization_gated.yaml')
    parser.add_argument('--summary-output',default='runs/research_summary')
    parser.add_argument('--jobs',type=int,choices=[1,2,3],default=3)
    args=parser.parse_args();paths=[str(Path(p).resolve()) for p in [args.context_plan,args.gated_plan,args.personal_plan,args.gated_personal_plan]]
    from neuroedge.context_protocol import load_plan,freeze
    from neuroedge.context_study import screen,select,members
    from neuroedge.gated_study import load_plan as gate_plan,freeze as gate_freeze,source,freeze_selection
    from neuroedge.personalization import load_plan as personal_plan,register,bind_selection,experiment
    from context_continue import restore
    context=load_plan(paths[0]);gated=gate_plan(paths[1]);personal=personal_plan(paths[2]);personal_gated=personal_plan(paths[3])
    root=Path(context['_root']);out=root/context['output_dir'];summary=root/args.summary_output
    if (root/gated['parent_plan']).resolve()!=Path(paths[0]) or (root/personal['parent_plan']).resolve()!=Path(paths[0]) or personal['parent_output']!=context['output_dir']:
        raise RuntimeError('Child plans must use the intended parent.')
    if (root/personal_gated['parent_plan']).resolve()!=Path(paths[0]) or personal_gated['parent_output']!=gated['output_dir']:
        raise RuntimeError('Gate calibration must use this context data plan and gate output.')
    freeze(context);gate_freeze(gated);register(personal);register(personal_gated)
    binder=str(Path(__file__).with_name('bind_research_selection.py'))
    subprocess.run([sys.executable,binder,'--context-plan',paths[0],'--gated-plan',paths[1],'--output',args.summary_output,'--register-only'],check=True)
    register_execution(root,summary,context,gated,args.jobs)
    restore(context,'development_data');restore(context,'external_data')
    screen(context);selection=select(context);bind_selection(personal)
    for variant in gated['variants']:source(gated,variant)
    freeze_selection(gated);bind_selection(personal_gated)
    subprocess.run([sys.executable,binder,'--context-plan',paths[0],'--gated-plan',paths[1],'--output',args.summary_output],check=True)
    worker=str(Path(__file__).with_name('context_fit_worker.py'))
    if not (out/'final_models_frozen.json').exists():
        tasks=[(name,call,([worker,'final','--plan',paths[0],'--names',name,'--resume'],out/'execution_logs'/('source_'+name+'.log'))) for name in members(selection)]
        dispatch(tasks,args.jobs)
    print('All source fits finished. Starting the frozen external transfer.',flush=True)
    transfer(context)
    tasks=[('gated external and verification',gate_external,(paths[1],root/gated['output_dir']))]
    for person in [1000+s for s in context['external']['subjects']]:
        directory=out/'external_loso/folds'/f'S{person:04d}'
        if not (directory/'complete.json').exists():
            tasks.append((f'context LOSO {person}',call,([worker,'loso','--plan',paths[0],'--subjects',str(person),'--resume'],out/'execution_logs'/f'LOSO_{person}.log')))
    dispatch(tasks,args.jobs);aggregate(context)
    from neuroedge.context_verify import verify
    from neuroedge.context_report import report
    verify(context);report(context)
    from neuroedge.gated_report import report as gate_report
    gate_report(gated)
    from neuroedge.personalization_verify import verify as verify_personal
    from neuroedge.personalization_report import report as report_personal
    for plan in [personal,personal_gated]:
        experiment(plan);verify_personal(plan);report_personal(plan)
    from neuroedge.research_report import report as overall_report
    overall_report(context,gated,personal,personal_gated,args.summary_output)
    print('All registered experiments, independent checks and reports completed.',flush=True)


if __name__=='__main__':main()
