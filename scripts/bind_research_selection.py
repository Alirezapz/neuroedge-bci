"""Freeze one overall source-selected pipeline/condition before external data."""
import argparse
import json
from pathlib import Path
import time

from neuroedge.config import digest,json_write,sha256
from neuroedge.context_protocol import load_plan,now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-plan',default='configs/context_fusion.yaml')
    parser.add_argument('--gated-plan',default='configs/gated_context.yaml')
    parser.add_argument('--output',default='runs/research_summary')
    parser.add_argument('--register-only',action='store_true')
    args=parser.parse_args()
    from neuroedge.gated_study import load_plan as gated_plan
    context=load_plan(args.context_plan);gated=gated_plan(args.gated_plan);root=Path(context['_root'])
    parent=root/context['output_dir'];child=root/gated['output_dir'];out=root/args.output;out.mkdir(parents=True,exist_ok=True)
    protocol={'rule':'For trial-only, inherit the parent source winner. For pre-cue, compare the parent source winner and the gate-study source winner using the SAME reused selection people: highest balanced accuracy, then lowest log loss, then stable name. No external selection.',
              'parent_protocol_sha256':json.loads((parent/'protocol.json').read_text())['sha256'],
              'gated_protocol_sha256':json.loads((child/'protocol.json').read_text())['sha256'],
              'binder_sha256':sha256(__file__)}
    protocol['sha256']=digest(protocol);path=out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())['sha256']!=protocol['sha256']:raise RuntimeError('Overall source-selection rule changed.')
    else:
        if list((root/context['external']['raw_dir']).glob('*.mat')):raise RuntimeError('Register the overall selection rule before external acquisition.')
        json_write(path,{**protocol,'registered_utc':now(),'registered_before_external_acquisition':True})
    if args.register_only:return
    while not ((parent/'selection.json').exists() and (child/'selection.json').exists()):time.sleep(2)
    p=json.loads((parent/'selection.json').read_text());g=json.loads((child/'selection.json').read_text())
    candidates=[{'origin':'context','model':'selected_pre_cue','result':p['selected']['pre_cue']},
                {'origin':'gated','model':g['selected']['variant'],'result':g['selected']}]
    winner=sorted(candidates,key=lambda r:(-r['result']['balanced_accuracy'],r['result']['log_loss'],r['result']['name']))[0]
    selection={'parent_selection_sha256':sha256(parent/'selection.json'),'gated_selection_sha256':sha256(child/'selection.json'),
               'zero_calibration':{'origin':'context','model':'selected_zero_calibration','result':p['selected']['zero_calibration']},
               'pre_cue':winner,'external_used_for_selection':False}
    target=out/'selection.json'
    if target.exists():
        if any(json.loads(target.read_text())[k]!=v for k,v in selection.items()):raise RuntimeError('Overall frozen selection changed.')
    else:
        if list((root/context['external']['raw_dir']).glob('*.mat')):raise RuntimeError('Do not choose an overall winner after external acquisition.')
        json_write(target,{**selection,'frozen_utc':now(),'frozen_before_external_acquisition':True})
    print('Overall source selection frozen:',winner['origin'],winner['model'],flush=True)


if __name__=='__main__':main()
