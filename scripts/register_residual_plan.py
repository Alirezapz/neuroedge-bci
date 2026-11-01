"""Register the residual mathematical specification before external acquisition."""
import argparse
import json
from pathlib import Path
from neuroedge.config import digest,json_write,sha256
from neuroedge.context_protocol import now
from neuroedge.residual_study import load_plan


def register_math(plan):
    root=Path(plan['_root']);out=root/plan['output_dir'];parent=root/plan['parent_output']
    clean={k:v for k,v in plan.items() if not k.startswith('_')}
    expected={'plan':clean,'parent_protocol_sha256':json.loads((parent/'protocol.json').read_text())['sha256'],
              'parent_selection_sha256':sha256(parent/'selection.json')}
    path=out/'specification.json'
    if path.exists():
        value=json.loads(path.read_text());claimed=value.pop('sha256')
        if digest(value)!=claimed or any(value[k]!=v for k,v in expected.items()):raise RuntimeError('Residual mathematical specification changed.')
        return path
    if list((root/plan['_parent']['external']['raw_dir']).glob('*.mat')):raise RuntimeError('Register the mathematical method before first external acquisition; use a fresh reproduction directory.')
    value={**expected,'mathematical_specification_registered_utc':now(),'mathematical_specification_before_external_acquisition':True,
        'scope':'Additional sequential source-development hypothesis. Implementation will be frozen before its source fitting and supplementary evaluation. No external outcome chooses its kernel, budget, C or representation.'}
    value['sha256']=digest(value);json_write(path,value);return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',default='configs/residual_personal.yaml')
    args=parser.parse_args();print(register_math(load_plan(args.plan)))
