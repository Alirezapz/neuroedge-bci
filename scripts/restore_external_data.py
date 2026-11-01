"""Restore archived BNCI inputs and require all original array fingerprints."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
from neuroedge.context_protocol import load_plan
from neuroedge.context_data import _download_external, bnci_session, _assemble
from neuroedge.config import sha256, json_write

root = Path(__file__).resolve().parents[1]
plan = load_plan(root / 'configs/context_fusion.yaml')
manifest_path = root / plan['output_dir'] / 'external_data/manifest.json'
expected = json.loads(manifest_path.read_text())
original_receipts = json.loads(manifest_path.with_name('source_receipts.json').read_text())
known = {s['path']: s['sha256'] for r in original_receipts for s in r['sources']}
requests = [(s,t) for s in plan['external']['subjects'] for t in plan['external']['sessions']]
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs = {pool.submit(_download_external,plan,*r):r for r in requests}
    for job in as_completed(jobs):
        path = job.result()
        assert sha256(path) == known[str(path.relative_to(root))], 'Raw acquisition changed'
        print('Acquired and verified', path.name, flush=True)
cache = root / plan['data']['cache_dir'] / 'external_subjects'
cache.mkdir(parents=True, exist_ok=True)
receipts = []
for subject, session in requests:
    path = root / plan['external']['raw_dir'] / f'A{subject:02d}{session}.mat'
    arrays, records = bnci_session(path,subject,session)
    target = cache / (path.stem+'.npz')
    np.savez_compressed(target,**arrays)
    receipts.append({'subject':1000+subject,'cache_path':str(target.relative_to(root)),
        'sha256':sha256(target),'records':records,
        'sources':[{'path':str(path.relative_to(root)),**json.loads(path.with_suffix('.source.json').read_text())}]})
    print('Prepared',path.stem,flush=True)
_assemble(plan,receipts,'external_data')
actual=json.loads(manifest_path.read_text())
assert actual == expected, 'Restored data differ from original manifest; stop before experiments'
json_write(root/'tests/external_restore_english.json',{'status':'PASS','all_five_original_input_hashes_match':True,
    'raw_files_verified':len(requests),'manifest_sha256':sha256(manifest_path)})
print('PASS: all five archived input fingerprints match',flush=True)
