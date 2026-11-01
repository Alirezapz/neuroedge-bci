"""Resume all source-only studies, then the prespecified external protocols.

Child studies and calibration are registered/bound BEFORE external acquisition.
The two external neural jobs use separate spawned processes, never shared RNG.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

# Execution scheduling only: retain the requested thread count and numerical
# kernels. A synthetic training probe matched state hashes across wait policies.
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')


def parent_external(path):
    from neuroedge.context_protocol import load_plan
    from neuroedge.context_study import confirm
    from neuroedge.context_verify import verify
    from neuroedge.context_report import report
    plan=load_plan(path);out=Path(plan['_root'])/plan['output_dir'];out.mkdir(parents=True,exist_ok=True)
    with (out/'external_execution.log').open('a',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        confirm(plan);verify(plan);report(plan)
    return 'context external protocols verified'


def gated_external(path):
    from neuroedge.gated_study import load_plan,external
    from neuroedge.gated_verify import verify
    from neuroedge.gated_report import report
    plan=load_plan(path);parent=Path(plan['_parent']['_root'])/plan['_parent']['output_dir']
    out=Path(plan['_root'])/plan['output_dir'];out.mkdir(parents=True,exist_ok=True)
    with (out/'external_execution.log').open('a',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        while not (parent/'external_transfer/complete.json').exists():time.sleep(2)
        external(plan);verify(plan)
        while not (parent/'verification.json').exists():time.sleep(2)
        report(plan)
    return 'gated external protocols verified'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-plan',default='configs/context_fusion.yaml')
    parser.add_argument('--gated-plan',default='configs/gated_context.yaml')
    parser.add_argument('--personal-plan',default='configs/personalization.yaml')
    parser.add_argument('--gated-personal-plan',default='configs/personalization_gated.yaml')
    parser.add_argument('--summary-output',default='runs/research_summary')
    args=parser.parse_args();paths=[str(Path(p).resolve()) for p in [args.context_plan,args.gated_plan,args.personal_plan,args.gated_personal_plan]]
    from neuroedge.context_protocol import load_plan,freeze
    from neuroedge.context_study import screen,select
    from neuroedge.gated_study import load_plan as gated_plan,freeze as gated_freeze,source,freeze_selection
    from neuroedge.personalization import load_plan as personal_plan,register,bind_selection,experiment
    from neuroedge.personalization_verify import verify as verify_personal
    from neuroedge.personalization_report import report as report_personal
    from context_continue import restore
    plan=load_plan(paths[0]);gated=gated_plan(paths[1]);personal=personal_plan(paths[2]);personal_gated=personal_plan(paths[3])
    root=Path(plan['_root'])
    if (root/gated['parent_plan']).resolve()!=Path(paths[0]) or (root/personal['parent_plan']).resolve()!=Path(paths[0]) or personal['parent_output']!=plan['output_dir']:
        raise RuntimeError('All child plans must reference the same intended context run.')
    if (root/personal_gated['parent_plan']).resolve()!=Path(paths[0]) or personal_gated['parent_output']!=gated['output_dir']:
        raise RuntimeError('The gate calibration plan must reference the intended gated output and context data plan.')
    freeze(plan);register(personal);gated_freeze(gated);register(personal_gated)
    subprocess.run([sys.executable,str(Path(__file__).with_name('bind_research_selection.py')),
        '--context-plan',paths[0],'--gated-plan',paths[1],'--output',args.summary_output,'--register-only'],check=True)
    restore(plan,'development_data');restore(plan,'external_data')
    screen(plan);select(plan);bind_selection(personal)
    for variant in gated['variants']:source(gated,variant)
    freeze_selection(gated);bind_selection(personal_gated)
    subprocess.run([sys.executable,str(Path(__file__).with_name('bind_research_selection.py')),
        '--context-plan',paths[0],'--gated-plan',paths[1],'--output',args.summary_output],check=True)
    print('All source choices and gated models fixed; starting independent external protocols.',flush=True)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(parent_external,paths[0]),pool.submit(gated_external,paths[1])]
        try:
            for job in as_completed(jobs):print(job.result(),flush=True)
        except BaseException:
            for process in (getattr(pool,'_processes',None) or {}).values():process.terminate()
            raise
    experiment(personal);verify_personal(personal);report_personal(personal)
    experiment(personal_gated);verify_personal(personal_gated);report_personal(personal_gated)
    from neuroedge.research_report import report
    report(plan,gated,personal,personal_gated,args.summary_output)
    print('All registered studies, calibration and independent checks completed.',flush=True)


if __name__=='__main__':main()
