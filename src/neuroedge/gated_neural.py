"""Matched fixed versus quality-conditioned fusion of task and ERD features.

Both networks retain task features, share normalization across concatenated
task/rest time points, and enforce reflection equivariance. The fixed control
has the same gate architecture/parameter count but receives constant quality
features; its learned gate is population fixed. No target-batch adaptation.
"""
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .config import digest,json_write,sha256
from .context_data import MIRROR
from .context_neural import inputs,probabilities
from .context_protocol import binary_probabilities,now


class GatedContext(nn.Module):
    def __init__(self,spec,settings):
        super().__init__();self.variant=spec['variant'];self.m=settings['neural']['spatial_filters_per_band']
        channels=6*self.m;hidden=settings['gate_hidden_units']
        self.spatial=nn.Conv2d(6,channels,(16,1),groups=6,bias=False)
        self.norm=nn.BatchNorm2d(channels,momentum=.1)
        self.gate=nn.Sequential(nn.LayerNorm(2*channels),nn.Linear(2*channels,hidden),nn.SiLU(),
                                nn.Linear(hidden,channels),nn.Sigmoid())
        nn.init.constant_(self.gate[-2].bias,settings['gate_initial_bias'])
        self.drop=nn.Dropout(settings['neural']['dropout']);self.classifier=nn.Linear(channels*4*2,2)
        self.register_buffer('mirror_indices',torch.tensor(MIRROR,dtype=torch.long))

    def _features(self,A,B):
        joined=F.silu(self.norm(self.spatial(torch.cat([A,B],dim=-1)))).squeeze(2)
        a=joined[...,:448];b=joined[...,448:]
        active=torch.log(torch.clamp(a.reshape(len(a),a.shape[1],4,112).var(dim=-1,unbiased=False),min=1e-6))
        baseline=torch.log(torch.clamp(b.var(dim=-1,unbiased=False),min=1e-6))
        local=torch.log(torch.clamp(b.reshape(len(b),b.shape[1],4,48).var(dim=-1,unbiased=False),min=1e-6))
        quality=torch.cat([baseline-baseline.mean(dim=1,keepdim=True),local.std(dim=-1,unbiased=False)],dim=1)
        if self.variant=='fixed_gate':quality=torch.zeros_like(quality)
        gate=self.gate(quality)
        contrast=active-gate[:,:,None]*baseline[:,:,None]
        return torch.cat([active.flatten(1),contrast.flatten(1)],dim=1),gate

    def forward(self,A,B):
        n=len(A);A=torch.cat([A,A[:,:,self.mirror_indices]],dim=0);B=torch.cat([B,B[:,:,self.mirror_indices]],dim=0)
        features,gate=self._features(A,B);raw=self.classifier(self.drop(features))
        return .5*(raw[:n]+raw[n:,[1,0]])

    @torch.no_grad()
    def constrain(self):
        for weight,axes,limit in [(self.spatial.weight,(1,2,3),2.),(self.classifier.weight,(1,),.5)]:
            norm=torch.linalg.vector_norm(weight,dim=axes,keepdim=True)
            weight.mul_(torch.clamp(limit/(norm+1e-8),max=1))


def fit_models(cohort,rows,spec,settings,directory,seeds):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    meta=cohort.meta.iloc[rows];expected={'spec':spec,'train_subjects':sorted(set(meta.subject.tolist())),
        'train_trial_ids_sha256':digest(meta.trial_id.tolist()),'seeds':list(seeds)}
    if (directory/'model_info.json').exists():
        info=json_read(directory/'model_info.json')
        if any(info[k]!=v for k,v in expected.items()):raise RuntimeError('Gated source membership changed.')
        check_files(directory,info);return info
    A,B=inputs(cohort,rows,spec);y=meta.label.to_numpy(dtype=np.int64)
    for seed in seeds:
        current=directory/f'seed_{seed}';current.mkdir(exist_ok=True);model_file=current/'model.pt';receipt=current/'seed_receipt.json'
        if model_file.exists() and receipt.exists():
            saved=json_read(receipt)
            if saved['train_trial_ids_sha256']!=expected['train_trial_ids_sha256'] or saved['model_sha256']!=sha256(model_file):raise RuntimeError('Gated seed changed.')
            continue
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(seed);model=GatedContext(spec,settings)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['neural']['learning_rate'],weight_decay=settings['neural']['weight_decay'])
        history=[];begin=time.monotonic()
        for epoch in range(1,spec['epochs']+1):
            model.train();order=rng.permutation(len(A));losses=[]
            for offset in range(0,len(order),settings['neural']['batch_size']):
                ids=order[offset:offset+settings['neural']['batch_size']]
                a=torch.from_numpy(A[ids]);b=torch.from_numpy(B[ids]);labels=torch.from_numpy(y[ids])
                optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(a,b),labels)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();model.constrain();losses.append(float(loss.detach()))
            history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-begin})
            if epoch%10==0 or epoch==spec['epochs']:print(f"Gated {spec['variant']} seed={seed} epoch={epoch}/{spec['epochs']} loss={history[-1]['training_loss']:.4f}",flush=True)
        pd.DataFrame(history).to_csv(current/'history.csv',index=False)
        torch.save({'spec':spec,'settings':settings,'state_dict':model.state_dict(),'train_subjects':expected['train_subjects'],
                    'train_trial_ids_sha256':expected['train_trial_ids_sha256'],'seed':int(seed)},model_file)
        json_write(receipt,{**expected,'seed':int(seed),'model_sha256':sha256(model_file)})
    info={**expected,'completed_utc':now(),'files':{str(p.relative_to(directory)):sha256(p) for p in directory.rglob('*') if p.is_file() and p.name!='model_info.json'}}
    json_write(directory/'model_info.json',info);return info


def json_read(path):
    import json
    return json.loads(Path(path).read_text())


def check_files(directory,info):
    for name,value in info['files'].items():
        if sha256(Path(directory)/name)!=value:raise RuntimeError('Gated model file checksum differs.')


def predict_saved(cohort,rows,spec,directory,return_gates=False):
    directory=Path(directory);info=json_read(directory/'model_info.json');check_files(directory,info)
    A,B=inputs(cohort,rows,spec);outputs=[];gates=[]
    for seed in info['seeds']:
        cp=torch.load(directory/f'seed_{seed}/model.pt',map_location='cpu',weights_only=True)
        if cp['spec']!=spec or cp['train_trial_ids_sha256']!=info['train_trial_ids_sha256'] or cp['train_subjects']!=info['train_subjects']:raise RuntimeError('Gated checkpoint provenance differs.')
        model=GatedContext(spec,cp['settings']);model.load_state_dict(cp['state_dict']);model.eval()
        outputs.append(probabilities(model,A,B,spec))
        if return_gates:
            with torch.no_grad():
                g=[]
                for start in range(0,len(A),128):g.append(model._features(torch.from_numpy(A[start:start+128]),torch.from_numpy(B[start:start+128]))[1].numpy())
                gates.append(np.concatenate(g))
    p=binary_probabilities(np.mean(outputs,axis=0))
    return (p,np.mean(gates,axis=0)) if return_gates else p
