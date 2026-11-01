"""Restore an explicit 22-EEG-sensor covariance input; no EOG or hidden sensors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .config import sha256,json_write
from .context_data import BNCI_CHANNELS,load_cohort
from .context_protocol import load_plan
from .frontier_data import BANDS,empirical_covariances


def parent_cohort(plan):return load_cohort(load_plan(Path(plan['_root'])/plan['parent_plan']),'external_data')


def prepare22(plan):
    import mne
    root=Path(plan['_root']);parent=parent_cohort(plan);cache=root/plan['cache_dir'];cache.mkdir(parents=True,exist_ok=True)
    path=cache/'cov22.npy';receipt=cache/'cov22.json'
    if path.exists() and receipt.exists():
        r=json.loads(receipt.read_text())
        if r['sha256']!=sha256(path) or r['code_sha256']!=sha256(__file__):raise RuntimeError('Cov22 cache changed')
        return
    original=json.loads((root/plan['parent_output']/'external_data/source_receipts.json').read_text())
    raw_sources={s['path']:s['sha256'] for r in original for s in r['sources']}
    array=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(parent.meta),3,8,22,22));ids=[];offset=0
    for subject in range(1,10):
        for session in ['T','E']:
            raw_path=root/'data/external/BNCI2014_001'/f'A{subject:02d}{session}.mat'
            if sha256(raw_path)!=raw_sources[str(raw_path.relative_to(root))]:raise AssertionError('Raw file changed')
            runs=np.atleast_1d(loadmat(raw_path,struct_as_record=False,squeeze_me=True)['data']);blocks=[]
            for ri,run in enumerate(runs):
                trials=np.atleast_1d(run.trial).astype(int)
                if not len(trials):continue
                labels=np.atleast_1d(run.y).astype(int);keep=np.flatnonzero(np.isin(labels,[1,2]));onsets=(trials[keep]-1)/250+2
                X=np.asarray(run.X,dtype=np.float64)
                if X.shape[1]!=25 or not np.isfinite(X).all() or float(run.fs)!=250:raise ValueError('Invalid recording')
                raw=mne.io.RawArray(X[:,:22].T*1e-6,mne.create_info(BNCI_CHANNELS,250,'eeg'),verbose='ERROR');bands=[]
                for low,high in BANDS:
                    r=raw.copy().filter(low,high,method='iir',iir_params={'order':4,'ftype':'butter','output':'sos'},phase='zero',verbose='ERROR');r.resample(128,verbose='ERROR')
                    v=r.get_data(units='uV');epochs=np.stack([v[:,round((t+.5)*128):round((t+.5)*128)+448] for t in onsets]).astype(np.float32)
                    epochs-=epochs.mean(axis=1,keepdims=True)
                    bands.append(np.stack([empirical_covariances(epochs),empirical_covariances(epochs[...,:224]),empirical_covariances(epochs[...,224:])],axis=1))
                block=np.stack(bands,axis=2).astype(np.float32);blocks.append(block)
                ids.extend(f'BNCI_A{subject:02d}_{session}_R{ri:02d}_T{i:03d}' for i in keep)
            current=np.concatenate(blocks);array[offset:offset+len(current)]=current;offset+=len(current)
            print('22-sensor covariance',subject,session,flush=True)
    array.flush();np.testing.assert_array_equal(ids,parent.meta.trial_id.to_numpy())
    if offset!=len(parent.meta):raise AssertionError('Cov22 coverage differs')
    json_write(receipt,{'sha256':sha256(path),'code_sha256':sha256(__file__),'shape':list(array.shape),
        'channels':BNCI_CHANNELS,'reference':'own22 common average','raw_sources':raw_sources,
        'parent_manifest_sha256':sha256(root/plan['parent_output']/'external_data/manifest.json')})


def covariances(plan,sensors):
    if sensors==16:return parent_cohort(plan).task_cov
    if sensors!=22:raise ValueError('Unexpected sensor count')
    root=Path(plan['_root']);r=json.loads((root/plan['cache_dir']/'cov22.json').read_text());p=root/plan['cache_dir']/'cov22.npy'
    if r['sha256']!=sha256(p) or r['code_sha256']!=sha256(__file__):raise AssertionError('Cov22 identity changed')
    return np.load(p,mmap_mode='r')
