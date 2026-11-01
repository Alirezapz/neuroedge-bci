"""Record exact new sensor-input hashes before a named covariance evaluation."""
import argparse
import json
from pathlib import Path
from neuroedge.config import digest,json_write,sha256
from neuroedge.covariance_study import load_plan,register,inputs

p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/covariance_lab_corrected.yaml');args=p.parse_args()
plan=load_plan(args.plan);root=Path(plan['_root']);out,identity=register(plan);meta,paths=inputs(plan)
value={'protocol_sha256':identity['sha256'],'covariances':{str(n):{'sha256':sha256(path),'path':str(Path(path).relative_to(root))} for n,path in paths.items()},
    'trial_ids_sha256':digest(meta.trial_id.tolist()),'labels_sha256':digest(meta.label.tolist()),
    'cov22_receipt':json.loads((root/plan['cache_dir']/'cov22.json').read_text())}
path=out/'input_manifest.json'
if path.exists():
    if json.loads(path.read_text())['identity']!=value:raise RuntimeError('Frozen sensor inputs differ')
else:
    if list((out/'models').glob('*/*.joblib')):raise RuntimeError('Freeze inputs before final E scoring')
    json_write(path,{'identity':value,'frozen_before_final_E_scoring':True})
print('PASS: exact16/22-sensor inputs frozen')
