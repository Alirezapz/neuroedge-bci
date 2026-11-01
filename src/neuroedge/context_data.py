"""Six-band task/context pairs, selected sensors only, no target-batch estimates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter,resample_poly,sosfilt,sosfilt_zi
from math import gcd

from .config import digest,json_write,sha256
from .data import CHANNELS_16
from .frontier_data import BANDS,acquire,empirical_covariances,validate_annotations

CHANNELS=list(CHANNELS_16)
MIRROR=np.asarray([CHANNELS.index({'FC3':'FC4','FC4':'FC3','C3':'C4','C4':'C3',
 'CP3':'CP4','CP4':'CP3','FC1':'FC2','FC2':'FC1','C1':'C2','C2':'C1',
 'CP1':'CP2','CP2':'CP1','C5':'C6','C6':'C5'}.get(c,c)) for c in CHANNELS])
BNCI_CHANNELS=['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4',
               'C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
ARRAYS=['active','baseline','task_cov','baseline_cov']


def paired_epochs(raw,cues):
    """Offline task filter; strictly past-only filter and resampling for context.

    Two forward Butterworth passes match the task filter's squared magnitude,
    while context has causal phase. No task/future sample can affect context.
    """
    active=[];baseline=[]
    native=raw.get_data(units='uV');fs=int(round(raw.info['sfreq']))
    for low,high in BANDS:
        filtered=raw.copy().filter(low,high,method='iir',iir_params={'order':4,'ftype':'butter','output':'sos'},
                                   phase='zero',verbose='ERROR')
        filtered.resample(128,verbose='ERROR')
        values=filtered.get_data(units='uV')
        sos=butter(4,[low,high],btype='bandpass',fs=fs,output='sos')
        past=native
        for _ in range(2):
            zi=sosfilt_zi(sos)[:,None,:]*past[None,:,0,None]
            past,_=sosfilt(sos,past,axis=-1,zi=zi)
        a=[];b=[]
        for cue in cues:
            start=round((cue+.5)*128);before=round((cue-1.75)*fs)
            x=values[:,start:start+448]
            z=resample_poly(past[:,before:before+round(1.5*fs)],128//gcd(128,fs),fs//gcd(128,fs),axis=-1,padtype='line')
            if x.shape!=(16,448) or z.shape!=(16,192) or before<0:
                raise ValueError('Incomplete fixed task/pre-cue pair.')
            a.append(x.astype(np.float32));b.append(z.astype(np.float32))
        active.append(np.stack(a));baseline.append(np.stack(b))
    A=np.stack(active,axis=1);B=np.stack(baseline,axis=1)
    # No access to omitted electrodes for the common average reference.
    A-=A.mean(axis=2,keepdims=True);B-=B.mean(axis=2,keepdims=True)
    def cov(x):
        return empirical_covariances(x.reshape(-1,16,x.shape[-1])).reshape(len(x),8,16,16)
    task=np.stack([cov(A),cov(A[...,:224]),cov(A[...,224:])],axis=1).astype(np.float32)
    return {'active':A,'baseline':B,'task_cov':task,
            'baseline_cov':cov(B).astype(np.float32)}


def _physio_subject(plan,subject,cache):
    import mne
    from mne.datasets.eegbci import standardize
    root=Path(plan['_root']);paths=[root/plan['data']['raw_dir']/f'S{subject:03d}/S{subject:03d}R{r:02d}.edf' for r in plan['data']['runs']]
    sources=[{'path':str(p.relative_to(root)),'sha256':sha256(p)} for p in paths]
    signature=digest({'sources':sources,'code':sha256(__file__),'channels':CHANNELS})
    target=cache/f'S{subject:03d}.npz';receipt_path=target.with_suffix('.json')
    if target.exists() and receipt_path.exists():
        saved=json.loads(receipt_path.read_text())
        if saved['signature']!=signature or saved['sha256']!=sha256(target):
            raise RuntimeError('Context cache changed; do not overwrite a frozen run.')
        return saved
    blocks=[];records=[]
    for path,run in zip(paths,plan['data']['runs']):
        raw=mne.io.read_raw_edf(path,preload=True,verbose='ERROR');standardize(raw);raw.pick('eeg')
        if len(raw.ch_names)!=64 or not np.isfinite(raw.get_data()).all():raise ValueError('Invalid source EEG.')
        cues=validate_annotations(raw,run,{'minimum_task_duration':4.,'minimum_trials_per_class_per_run':2})
        rest=[(float(t),float(t+d)) for t,d,label in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description) if label=='T0']
        for cue,label in cues:
            if not any(cue-1.75>=start-1/raw.info['sfreq'] and cue-.25<=end+1/raw.info['sfreq'] for start,end in rest):
                raise ValueError(f'Pre-cue context is not wholly within annotated rest: S{subject:03d} R{run:02d}.')
        raw.pick(CHANNELS);raw.reorder_channels(CHANNELS)
        blocks.append(paired_epochs(raw,[t for t,y in cues]))
        records.extend({'trial_id':f'S{subject:03d}_R{run:02d}_T{i:03d}','subject':subject,'run':run,
                        'session':1,'onset':t,'label':y,'dataset':'EEGMMIDB','artifact_flag':False,'artifact_flag_available':False}
                       for i,(t,y) in enumerate(cues))
    np.savez_compressed(target,**{name:np.concatenate([b[name] for b in blocks]) for name in ARRAYS})
    receipt={'subject':subject,'signature':signature,'sha256':sha256(target),'sources':sources,'records':records}
    json_write(receipt_path,receipt);return receipt


def _assemble(plan,receipts,role):
    root=Path(plan['_root']);out=root/plan['output_dir']/role;out.mkdir(parents=True,exist_ok=True)
    dest=root/plan['data']['cache_dir']/role;dest.mkdir(parents=True,exist_ok=True)
    total=sum(len(r['records']) for r in receipts)
    shapes={'active':(total,8,16,448),'baseline':(total,8,16,192),
            'task_cov':(total,3,8,16,16),'baseline_cov':(total,8,16,16)}
    arrays={n:np.lib.format.open_memmap(dest/(n+'.npy'),mode='w+',dtype=np.float32,shape=s) for n,s in shapes.items()}
    offset=0;records=[]
    for receipt in receipts:
        with np.load(root/receipt['cache_path']) as saved:
            n=len(receipt['records'])
            for name in ARRAYS:arrays[name][offset:offset+n]=saved[name]
            offset+=n;records.extend(receipt['records'])
    for values in arrays.values():values.flush()
    pd.DataFrame(records).to_csv(dest/'trials.csv',index=False)
    files={name:{'path':str((dest/(name+'.npy')).relative_to(root)),'sha256':sha256(dest/(name+'.npy'))} for name in ARRAYS}
    files['trials']={'path':str((dest/'trials.csv').relative_to(root)),'sha256':sha256(dest/'trials.csv')}
    manifest={'role':role,'channels':CHANNELS,'files':files,'n_trials':total,
              'subjects':sorted({r['subject'] for r in records}),'data_source_sha256':sha256(__file__)}
    json_write(out/'manifest.json',manifest);json_write(out/'source_receipts.json',receipts)
    return load_cohort(plan,role)


def load_cohort(plan,role):
    root=Path(plan['_root']);p=root/plan['output_dir']/role/'manifest.json'
    manifest=json.loads(p.read_text())
    if manifest['channels']!=CHANNELS or manifest['data_source_sha256']!=sha256(__file__):
        raise RuntimeError('Data preparation identity differs.')
    for item in manifest['files'].values():
        if sha256(root/item['path'])!=item['sha256']:raise RuntimeError('Context data checksum differs.')
    return ContextCohort(root,manifest)


class ContextCohort:
    def __init__(self,root,manifest):
        self.root=Path(root);self.manifest=manifest
        for name in ARRAYS:setattr(self,name,np.load(self.root/manifest['files'][name]['path'],mmap_mode='r'))
        self.meta=pd.read_csv(self.root/manifest['files']['trials']['path'])
        self.names=list(manifest['channels'])


def prepare_development(plan):
    root=Path(plan['_root']);role='development_data'
    if (root/plan['output_dir']/role/'manifest.json').exists():return load_cohort(plan,role)
    cache=root/plan['data']['cache_dir']/'subjects';cache.mkdir(parents=True,exist_ok=True)
    receipts=[];people=plan['source_subjects']+plan['selection_subjects']
    acquire({'_root':plan['_root'],'dataset':plan['data']},people)
    for i,subject in enumerate(people,1):
        receipt=_physio_subject(plan,subject,cache)
        receipt['cache_path']=str((cache/f'S{subject:03d}.npz').relative_to(root));receipts.append(receipt)
        print(f'Context pairs S{subject:03d} ({i}/{len(people)})',flush=True)
    return _assemble(plan,receipts,role)


def _download_external(plan,subject,session):
    root=Path(plan['_root']);cfg=plan['external'];filename=f'A{subject:02d}{session}.mat'
    path=root/cfg['raw_dir']/filename;path.parent.mkdir(parents=True,exist_ok=True);sidecar=path.with_suffix('.source.json')
    if path.exists() and sidecar.exists():
        saved=json.loads(sidecar.read_text())
        if sha256(path)!=saved['sha256']:raise RuntimeError('External raw checksum changed.')
        return path
    url=cfg['base_url']+'/'+filename
    for attempt in range(cfg['download_attempts']):
        temporary=path.with_suffix('.part')
        try:
            with urllib.request.urlopen(url,timeout=cfg['download_timeout']) as response,temporary.open('wb') as f:
                while chunk:=response.read(1024*1024):f.write(chunk)
            with temporary.open('rb') as f:
                if not f.read(20).startswith(b'MATLAB 5.0 MAT-file'):raise ValueError('Not a MATLAB file.')
            temporary.replace(path);json_write(sidecar,{'url':url,'sha256':sha256(path),'bytes':path.stat().st_size})
            return path
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt+1==cfg['download_attempts']:raise
            time.sleep(2**attempt)


def bnci_session(path,subject,session):
    """BNCI MATLAB trial indices denote trial start, cue is two seconds later."""
    import mne
    blocks=[];records=[]
    runs=np.atleast_1d(loadmat(path,struct_as_record=False,squeeze_me=True)['data'])
    for run_index,run in enumerate(runs):
        trials=np.atleast_1d(run.trial).astype(int)
        if not len(trials):continue  # separate EOG calibration, never a training example
        y=np.atleast_1d(run.y).astype(int)
        if len(y)!=len(trials) or set(y)!={1,2,3,4}:raise ValueError('Missing external labels or unexpected task mapping.')
        X=np.asarray(run.X,dtype=np.float64)
        if X.ndim!=2 or X.shape[1]!=25 or float(run.fs)!=250 or not np.isfinite(X).all():raise ValueError('Invalid BNCI recording structure.')
        keep=np.flatnonzero(np.isin(y,[1,2]));onsets=(trials[keep]-1)/250+2.
        flags=np.atleast_1d(getattr(run,'artifacts',np.asarray([],dtype=bool))).astype(bool)
        flags_available=len(flags)>0
        if not flags_available:flags=np.zeros(len(y),dtype=bool)
        if len(flags)!=len(y):raise ValueError('Artifact flag identity mismatch.')
        indices=[BNCI_CHANNELS.index(c) for c in CHANNELS]
        raw=mne.io.RawArray(X[:,indices].T*1e-6,mne.create_info(CHANNELS,250,'eeg'),verbose='ERROR')
        blocks.append(paired_epochs(raw,onsets))
        records.extend({'trial_id':f'BNCI_A{subject:02d}_{session}_R{run_index:02d}_T{i:03d}',
                        'subject':1000+subject,'session':1 if session=='T' else 2,'run':run_index,'onset':t,
                        'label':int(y[i]-1),'dataset':'BNCI2014_001','artifact_flag':bool(flags[i]),'artifact_flag_available':flags_available}
                       for i,t in zip(keep,onsets))
    if len(records)!=144:raise ValueError('Expected every one of the 144 left/right trials per external session.')
    return {name:np.concatenate([b[name] for b in blocks]) for name in ARRAYS},records


def prepare_external(plan):
    root=Path(plan['_root']);out=root/plan['output_dir']
    if not (out/'final_models_frozen.json').exists():raise RuntimeError('Freeze all final source models before external EEG acquisition.')
    if (out/'external_data/manifest.json').exists():return load_cohort(plan,'external_data')
    requests=[(s,t) for s in plan['external']['subjects'] for t in plan['external']['sessions']]
    with ThreadPoolExecutor(max_workers=plan['external']['download_workers']) as pool:
        jobs=[pool.submit(_download_external,plan,*r) for r in requests]
        for i,job in enumerate(as_completed(jobs),1):job.result();print(f'External MATLAB ready {i}/{len(jobs)}',flush=True)
    cache=root/plan['data']['cache_dir']/'external_subjects';cache.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for subject,session in requests:
        path=root/plan['external']['raw_dir']/f'A{subject:02d}{session}.mat'
        arrays,records=bnci_session(path,subject,session)
        target=cache/(path.stem+'.npz');np.savez_compressed(target,**arrays)
        receipts.append({'subject':1000+subject,'cache_path':str(target.relative_to(root)),
                         'sha256':sha256(target),'records':records,
                         'sources':[{'path':str(path.relative_to(root)),**json.loads(path.with_suffix('.source.json').read_text())}]})
        print('External context pairs',path.stem,flush=True)
    return _assemble(plan,receipts,'external_data')
