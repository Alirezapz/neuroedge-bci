"""Complete source development, frozen external studies and personal kernels.

Use fresh named configurations in a fresh project directory for reproduction.
Reusing these same nine target people is not a new independent confirmation.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in [('context','context_fusion'),('gated','gated_context'),('personal','personalization'),
                         ('gated-personal','personalization_gated'),('residual','residual_personal')]:
        parser.add_argument('--'+name+'-plan',default='configs/'+default+'.yaml')
    parser.add_argument('--summary-output',default='runs/research_summary')
    parser.add_argument('--jobs',type=int,choices=[1,2,3],default=3)
    parser.add_argument('--external-jobs',type=int,choices=[1,2,3,4,5],default=3)
    args=parser.parse_args()
    from neuroedge.context_protocol import load_plan,freeze
    from neuroedge.context_study import screen,select
    from neuroedge.gated_study import load_plan as gate_plan,freeze as gate_freeze
    from neuroedge.personalization import load_plan as personal_plan,register
    from neuroedge.residual_study import load_plan as residual_plan,source,experiment
    from context_continue import restore
    from register_residual_plan import register_math
    context=load_plan(args.context_plan);gated=gate_plan(args.gated_plan)
    personal=personal_plan(args.personal_plan);personal_gated=personal_plan(args.gated_personal_plan)
    residual=residual_plan(args.residual_plan);root=Path(context['_root'])
    if (root/residual['parent_plan']).resolve()!=Path(args.context_plan).resolve() or residual['parent_output']!=context['output_dir']:
        raise RuntimeError('Residual calibration must use the intended context source and predictions.')
    freeze(context);gate_freeze(gated);register(personal);register(personal_gated)
    restore(context,'development_data');screen(context);select(context)
    register_math(residual);source(residual)
    command=[sys.executable,'-u',str(Path(__file__).with_name('run_context_cpu.py')),
        '--context-plan',args.context_plan,'--gated-plan',args.gated_plan,'--personal-plan',args.personal_plan,
        '--gated-personal-plan',args.gated_personal_plan,'--summary-output',args.summary_output,
        '--jobs',str(args.jobs),'--external-jobs',str(args.external_jobs)]
    subprocess.run(command,check=True)
    experiment(residual)
    from neuroedge.residual_verify import verify
    from neuroedge.residual_report import report
    verify(residual,personal['output_dir']);report(residual)
    from neuroedge.research_report import attach_residual
    attach_residual(context,residual,args.summary_output)
    print('All experiments, including source-prior personal kernels, completed and verified.',flush=True)


if __name__=='__main__':main()
