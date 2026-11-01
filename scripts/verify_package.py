"""Check archive CRC, complete retained fits and immutable neural seed files."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile


def verify(path):
    path=Path(path);checked=seeds=0
    with zipfile.ZipFile(path) as archive:
        names=set(archive.namelist())
        if len(names)!=len(archive.namelist()):raise RuntimeError('Duplicate archive entry.')
        for name in names:
            parts=PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or not parts.parts or parts.parts[0]!='neuroedge':raise RuntimeError('Unexpected archive path.')
        bad=archive.testzip()
        if bad:raise RuntimeError('CRC failure: '+bad)
        for name in names:
            if not name.endswith('/model_info.json'):continue
            info=json.loads(archive.read(name));directory=name.rsplit('/',1)[0]
            # Temporary source screening/validation weights are intentionally
            # omitted; their complete prediction/specification records remain.
            if any(part in name for part in ['/screening/','/validation/','/validation_models/']):continue
            if isinstance(info.get('files'),dict):
                for filename,expected in info['files'].items():
                    if isinstance(expected,str):
                        actual=hashlib.sha256(archive.read(directory+'/'+filename)).hexdigest()
                        if actual!=expected:raise RuntimeError('Model hash differs: '+name+'/'+filename)
                checked+=1
        for name in names:
            if not name.endswith('/seed_receipt.json'):continue
            model=name.rsplit('/',1)[0]+'/model.pt'
            if model not in names and any(p in name for p in ['/screening/','/validation/']):continue
            expected=json.loads(archive.read(name))['model_sha256']
            if hashlib.sha256(archive.read(model)).hexdigest()!=expected:raise RuntimeError('Neural seed hash differs: '+name)
            seeds+=1
        session_checked=0
        receipt_name='neuroedge/runs/session_adaptation/fit_receipts.json'
        if receipt_name in names:
            for receipt in json.loads(archive.read(receipt_name)):
                model='neuroedge/runs/session_adaptation/'+receipt['model_path']
                if hashlib.sha256(archive.read(model)).hexdigest()!=receipt['model_sha256']:
                    raise RuntimeError('Session model differs: '+model)
                session_checked+=1
            if session_checked!=540:raise RuntimeError('Session model coverage differs')
        status=json.loads(archive.read('neuroedge/PACKAGE_STATUS.json'))
        covariance_checked=0
        receipt_name='neuroedge/runs/covariance_lab_corrected/fit_receipts.json'
        if receipt_name in names:
            for receipt in json.loads(archive.read(receipt_name)):
                model='neuroedge/runs/covariance_lab_corrected/'+receipt['model_path']
                if hashlib.sha256(archive.read(model)).hexdigest()!=receipt['sha256']:
                    raise RuntimeError('Covariance model differs: '+model)
                covariance_checked+=1
            if covariance_checked!=972:raise RuntimeError('Covariance model coverage differs')
    result={'status':'PASS','path':str(path.resolve()),'bytes':path.stat().st_size,'files':len(names),
        'retained_model_file_sets_checked':checked,'retained_neural_seed_hashes_checked':seeds,
        'session_calibration_artifacts_checked':session_checked,
        'covariance_lab_artifacts_checked':covariance_checked,
        'context_complete_folds':sum('/context_fusion/external_loso/folds/' in n and n.endswith('/complete.json') for n in names),
        'gate_complete_folds':sum('/gated_context/external_loso/folds/' in n and n.endswith('/complete.json') for n in names),
        'run_status':{name:value['status'] for name,value in status['runs'].items() if 'status' in value}}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive');args=parser.parse_args()
    print(json.dumps(verify(args.archive)),flush=True)
