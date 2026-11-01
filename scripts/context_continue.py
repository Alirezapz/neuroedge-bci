"""Recover omitted context-study arrays only if every frozen checksum matches."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from neuroedge.config import json_write,sha256
from neuroedge.context_protocol import freeze,load_plan
from neuroedge.context_data import prepare_development,prepare_external
from neuroedge.context_study import screen,select,confirm


def restore(plan,role):
    root=Path(plan['_root']);out=root/plan['output_dir'];path=out/role/'manifest.json'
    if not path.exists():return
    saved=json.loads(path.read_text())
    if all((root/item['path']).exists() for item in saved['files'].values()):return
    temporary=copy.deepcopy(plan);temporary['output_dir']=plan['output_dir']+'_cache_recovery'
    temporary['data']['cache_dir']=plan['data']['cache_dir']+'_cache_recovery'
    if role=='development_data':rebuilt=prepare_development(temporary)
    else:
        if not (out/'selection.json').exists() or not (out/'final_models_frozen.json').exists():
            raise RuntimeError('Cannot reconstruct external data without the original frozen selection.')
        frozen=json.loads((out/'final_models_frozen.json').read_text())
        if frozen['selection_sha256']!=sha256(out/'selection.json'):raise RuntimeError('Original frozen selection differs.')
        json_write(root/temporary['output_dir']/'final_models_frozen.json',{
            'recovery_of_original_frozen_run':plan['output_dir'],'original_selection_sha256':frozen['selection_sha256'],
            'note':'Permission to reconstruct already-acquired arrays; not a new model fit or selection.'})
        rebuilt=prepare_external(temporary)
    for name,item in saved['files'].items():
        other=rebuilt.manifest['files'][name]
        if other['sha256']!=item['sha256']:
            raise RuntimeError(f'Reconstructed {role}/{name} differs. Preserve the archive and use a fresh named reproduction; never bypass data identities.')
    for name,item in saved['files'].items():
        target=root/item['path'];source=root/rebuilt.manifest['files'][name]['path']
        if target.exists():
            if sha256(target)!=item['sha256']:raise RuntimeError('Existing frozen array differs.')
        elif source!=target:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    print('Exact context arrays restored:',role,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',default='configs/context_fusion.yaml')
    args=parser.parse_args();plan=load_plan(args.plan);freeze(plan)
    restore(plan,'development_data');restore(plan,'external_data')
    screen(plan);select(plan);confirm(plan)
    from neuroedge.context_verify import verify
    from neuroedge.context_report import report
    verify(plan);report(plan)


if __name__=='__main__':main()
