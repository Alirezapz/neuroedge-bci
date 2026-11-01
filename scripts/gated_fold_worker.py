"""Precompute a frozen gated fold; publish atomically without replacing a writer.

The Linux-only publication step refuses an already-started main destination.
All scientific fitting/prediction functions and settings remain canonical.
"""
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')


def publish_without_replacement(source,destination):
    """One atomic RENAME_NOREPLACE; never overwrite an existing destination."""
    libc=ctypes.CDLL(None,use_errno=True)
    if not hasattr(libc,'renameat2'):return False,'atomic_publication_unavailable'
    rename=libc.renameat2;rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    code=rename(-100,os.fsencode(source),-100,os.fsencode(destination),1)
    if code==0:return True,'published'
    error=ctypes.get_errno()
    if error in [errno.EEXIST,errno.ENOTEMPTY]:return False,'main_destination_already_exists'
    raise OSError(error,os.strerror(error),str(destination))


def main():
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from neuroedge.config import json_write,sha256
    from neuroedge.context_data import load_cohort
    from neuroedge.context_models import apply_temperature
    from neuroedge.context_protocol import now,predictions
    from neuroedge.context_study import summarize
    from neuroedge.gated_study import freeze,load_plan
    from neuroedge.gated_neural import fit_models,predict_saved
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',default='configs/gated_context.yaml')
    parser.add_argument('--subject',type=int,required=True);args=parser.parse_args();plan=load_plan(args.plan);out,identity=freeze(plan)
    selection=json.loads((out/'selection.json').read_text());winner=selection['selected'];target=out/'external_loso/folds'/f'S{args.subject:04d}'
    if target.exists():print('Main gate destination already started; skipping auxiliary work.',flush=True);return
    cohort=load_cohort(plan['_parent'],'external_data')
    if args.subject not in set(cohort.meta.subject):raise RuntimeError('Unknown held person.')
    train=np.flatnonzero(cohort.meta.subject!=args.subject);held=np.flatnonzero(cohort.meta.subject==args.subject)
    temporary_parent=Path(plan['_root'])/'data/gated_auxiliary';temporary_parent.mkdir(parents=True,exist_ok=True);target.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(plan['_parent']['search']['threads'])
    with tempfile.TemporaryDirectory(prefix=f'S{args.subject:04d}_',dir=temporary_parent) as name:
        temporary=Path(name)/'fold';temporary.mkdir()
        with threadpool_limits(limits=plan['_parent']['search']['threads']):
            fit_models(cohort,train,winner['spec'],plan['_settings'],temporary/'model',plan['seeds'])
            raw=predict_saved(cohort,held,winner['spec'],temporary/'model');p=apply_temperature(raw[:,1],winner['temperature'])
        frame=predictions(cohort.meta.iloc[held].reset_index(drop=True),p);frame['model']='selected_fusion';frame['arm']='pre_cue'
        bootstrap={'evaluation':{'bootstrap_seed':plan['bootstrap_seed'],'bootstrap_iterations':plan['bootstrap_iterations']}}
        summarize(bootstrap,frame,temporary,{'protocol_sha256':identity['sha256'],'selection_sha256':sha256(out/'selection.json'),
            'external_used_for_selection':False,'held_subject':args.subject})
        published,status=publish_without_replacement(temporary,target)
    json_write(out/'scheduling'/f'auxiliary_{args.subject}.json',{'worker_sha256':sha256(__file__),'completed_utc':now(),
        'protocol_sha256':identity['sha256'],'selection_sha256':sha256(out/'selection.json'),'published':published,'status':status,
        'scope':'Same frozen fold; scheduling only. Existing main destination always wins.'})
    print('Gate auxiliary fold',args.subject,status,flush=True)


if __name__=='__main__':main()
