"""Preserve fold-training receipts/histories without packaging large search fits."""
from pathlib import Path
import argparse
import json
import shutil

from neuroedge.config import digest,json_write,sha256
from neuroedge.frontier import discovery_splits
from neuroedge.frontier_data import load_cohort
from neuroedge.frontier_features import candidates
from neuroedge.frontier_protocol import load_plan,freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='configs/frontier.yaml')
    args=parser.parse_args()
    plan=load_plan(args.plan);out,identity=freeze(plan)
    source=load_cohort(plan,'development_data')
    run_key=digest({k:v for k,v in plan.items() if not k.startswith('_')})[:16]
    cache=Path(plan['_root'])/plan['dataset']['cache_dir']/'screen_models'/run_key
    copied=0;models=0;inner=0
    for spec in candidates():
        directory=out/'screening'/spec['name']
        if not (directory/'complete.json').exists():
            raise RuntimeError('Finish every candidate before collecting final evidence.')
        for split in discovery_splits(plan):
            fitted=cache/spec['name']/f"fold_{split['fold']}"
            info=json.loads((fitted/'model_info.json').read_text())
            expected=source.meta[source.meta.subject.isin(split['train_subjects'])]
            if set(info['train_subjects'])!=set(split['train_subjects']):
                raise AssertionError('Screening fit participant membership differs.')
            if info['train_trial_ids_sha256']!=digest(expected.trial_id.tolist()):
                raise AssertionError('Screening source trial identities differ.')
            for filename,expected_sha in info['files'].items():
                if sha256(fitted/filename)!=expected_sha:
                    raise AssertionError('Screening model artifact hash changed.')
                models+=1
            for path in fitted.rglob('*'):
                if not path.is_file() or path.suffix not in {'.json','.csv'}:
                    continue
                if path.name=='inner_split.json':
                    saved=json.loads(path.read_text())
                    train=set(saved['train_subjects']);val=set(saved['validation_subjects'])
                    if train&val or train|val!=set(split['train_subjects']):
                        raise AssertionError('Inner neural source-only split differs.')
                    if (train|val)&set(split['validation_subjects']):
                        raise AssertionError('Outer discovery labels used for epoch selection.')
                    inner+=1
                target=directory/'fit_evidence'/f"fold_{split['fold']}"/path.relative_to(fitted)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target);copied+=1
    json_write(out/'screening_fit_evidence.json',{'status':'PASS','fold_evaluations':len(candidates())*plan['search']['folds'],
        'training_artifact_hashes_checked':models,'source_only_inner_splits_checked':inner,
        'metadata_history_files_preserved':copied,'protocol_sha256':identity['sha256'],
        'collector_source_sha256':sha256(__file__),
        'scope':'All fold-training participant/trial receipts and model hashes checked; metadata and neural histories preserved. Large transient fitted screening models remain rebuildable data-cache artifacts, not in the source archive.'})
    acquired=[]
    for subject in plan['discovery_subjects']+plan['validation_subjects']+plan['confirmation_subjects']:
        for run in [1]+plan['dataset']['runs']:
            path=Path(plan['_root'])/plan['dataset']['raw_dir']/f'S{subject:03d}/S{subject:03d}R{run:02d}.edf'
            sidecar=path.with_suffix('.source.json')
            if not path.exists() or not sidecar.exists():continue
            provenance=json.loads(sidecar.read_text())
            if provenance['sha256']!=sha256(path) or provenance['bytes']!=path.stat().st_size:
                raise AssertionError('Acquisition-source bytes/hash changed.')
            acquired.append({'subject':subject,'run':run,'path':str(path.relative_to(Path(plan['_root']))),**provenance})
    json_write(out/'acquisition_sources.json',{'dataset':'EEGMMIDB 1.0.0, PhysioNet DOI10.13026/C28G6P',
        'files':acquired,'n_files':len(acquired),'collector_source_sha256':sha256(__file__),
        'note':'Exact acquired URL, bytes and SHA256. Only already-acquired planned files are listed; no download is initiated by this collector.'})
    print('SCREENING FIT EVIDENCE PASS',models,'model artifacts,',inner,'inner source-only splits,',copied,'metadata/history files')


if __name__=='__main__':main()
