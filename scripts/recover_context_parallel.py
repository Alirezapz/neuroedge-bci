"""Rebuild omitted arrays in isolated processes; require every frozen byte hash.

This changes data-preparation scheduling only. Canonical scientific extraction
functions and the original manifests are retained unchanged.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import os
import tempfile
import time
import zipfile


def rebuild(job):
    import numpy as np
    from neuroedge.config import sha256,json_write
    from neuroedge.context_data import _physio_subject,bnci_session
    plan,role,original=job;root=Path(plan['_root'])
    paths=[root/item['path'] for item in original['sources']]
    started=time.monotonic()
    while not all(p.exists() for p in paths):
        if time.monotonic()-started>900:raise RuntimeError('Required raw EEG did not arrive: '+str(paths))
        time.sleep(1)
    for path,item in zip(paths,original['sources']):
        if sha256(path)!=item['sha256']:raise RuntimeError('Raw EEG differs from original acquisition: '+str(path))
    if role=='development_data':
        cache=root/plan['data']['cache_dir']/'subjects';cache.mkdir(parents=True,exist_ok=True)
        target=cache/f"S{original['subject']:03d}.npz";metadata=target.with_suffix('.json')
        if target.exists() and metadata.exists():
            saved=json.loads(metadata.read_text())
            actual=sha256(target)
            if actual!=saved['sha256'] or not zipfile.is_zipfile(target):
                quarantine=root/'data/context_recovery_quarantine';quarantine.mkdir(parents=True,exist_ok=True)
                json_write(root/'tests/recovery_quarantine'/f"S{original['subject']:03d}.json",{
                    'scope':'Corrupt temporary reconstruction cache only; frozen raw/array/model identities retained.',
                    'observed_sha256':actual,'expected_temporary_sha256':saved['sha256'],'raw_sources':original['sources']})
                os.replace(target,quarantine/target.name);os.replace(metadata,quarantine/metadata.name)
        if target.exists() and metadata.exists():receipt=_physio_subject(plan,original['subject'],cache)
        else:
            with tempfile.TemporaryDirectory(prefix='complete_source_',dir=cache) as staging:
                staging=Path(staging);receipt=_physio_subject(plan,original['subject'],staging)
                os.replace(staging/target.name,target);os.replace(staging/metadata.name,metadata)
        receipt['cache_path']=str(target.relative_to(root))
    else:
        path=paths[0];subject=original['subject']-1000;session=path.stem[-1]
        arrays,records=bnci_session(path,subject,session)
        cache=root/plan['data']['cache_dir']/'external_subjects';cache.mkdir(parents=True,exist_ok=True)
        target=cache/(path.stem+'.npz')
        with tempfile.TemporaryDirectory(prefix='complete_external_',dir=cache) as staging:
            stage=Path(staging)/target.name;np.savez_compressed(stage,**arrays);os.replace(stage,target)
        receipt={'subject':original['subject'],'cache_path':str(target.relative_to(root)),
                 'sha256':sha256(target),'records':records,'sources':original['sources']}
    if receipt['records']!=original['records']:raise RuntimeError('Rebuilt trial records differ from the frozen acquisition.')
    return receipt


def main():
    from neuroedge.config import json_write,sha256
    from neuroedge.context_data import _assemble,load_cohort
    from neuroedge.context_protocol import load_plan,freeze,now
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='configs/context_fusion.yaml')
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=3)
    args=parser.parse_args();plan=load_plan(args.plan);out,identity=freeze(plan);root=Path(plan['_root'])
    frozen=json.loads((out/'final_models_frozen.json').read_text())
    if frozen['selection_sha256']!=sha256(out/'selection.json'):raise RuntimeError('Original source selection changed.')
    temporary=copy.deepcopy(plan);temporary['output_dir']+='_cache_recovery';temporary['data']['cache_dir']+='_cache_recovery'
    roles=['development_data','external_data'];audit={'status':'in_progress','started_utc':now(),
        'protocol_sha256':identity['sha256'],'execution_worker_sha256':sha256(__file__),
        'workers':args.workers,'method':'Unchanged canonical extraction, parallel sessions/people, strict original array and trial checksums.','roles':{}}
    for role in roles:
        manifest=json.loads((out/role/'manifest.json').read_text())
        if all((root/item['path']).exists() for item in manifest['files'].values()):
            load_cohort(plan,role);audit['roles'][role]={'status':'already_exact'};continue
        original=json.loads((out/role/'source_receipts.json').read_text());receipts={}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(rebuild,(temporary,role,receipt)):i for i,receipt in enumerate(original)}
            for future in as_completed(jobs):
                index=jobs[future];receipts[index]=future.result()
                print('Recovered',role,len(receipts),'/',len(jobs),receipts[index]['sources'][0]['path'],flush=True)
        for index in range(len(original)):
            for attempt in range(3):
                cached=root/receipts[index]['cache_path']
                if sha256(cached)==receipts[index]['sha256'] and zipfile.is_zipfile(cached):break
                print('Repairing damaged temporary cache from original raw EEG:',cached.name,flush=True)
                receipts[index]=rebuild((temporary,role,original[index]))
            else:raise RuntimeError('Temporary cache did not remain intact; frozen originals were not changed.')
        rebuilt=_assemble(temporary,[receipts[i] for i in range(len(original))],role)
        for name,item in manifest['files'].items():
            if rebuilt.manifest['files'][name]['sha256']!=item['sha256']:
                raise RuntimeError('Reconstructed frozen bytes differ: '+role+'/'+name+'; never bypass this check.')
        for name,item in manifest['files'].items():
            source=root/rebuilt.manifest['files'][name]['path'];target=root/item['path']
            if target.exists():
                if sha256(target)!=item['sha256']:raise RuntimeError('Existing frozen destination changed.')
            else:
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        load_cohort(plan,role)
        audit['roles'][role]={'status':'exact','files':{name:item['sha256'] for name,item in manifest['files'].items()}}
        json_write(root/'tests/context_recovery_20261101.json',audit)
        print('All five frozen hashes exactly restored:',role,flush=True)
    audit['status']='PASS';audit['completed_utc']=now();json_write(root/'tests/context_recovery_20261101.json',audit)
    print('Exact original development and external arrays restored.',flush=True)


if __name__=='__main__':main()
