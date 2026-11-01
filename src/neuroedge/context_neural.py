"""Independent FBCNet-inspired task/context adaptation and symmetry ablations.

Not an exact reproduction of the published nine-band FBCNet. Uses six fixed
bands, eight spatial filters/band, four variance segments, and paired pre-cue
log-variance subtraction in the context condition. No target-batch adaptation.
"""
from __future__ import annotations

import copy
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .context_data import MIRROR
from .context_protocol import binary_probabilities
from .models import EEGNet,TrainOnlyScaler


class FilterBankVariance(nn.Module):
    def __init__(self,plan,context=False,equivariant=False):
        super().__init__();cfg=plan['neural'];self.context=context;self.equivariant=equivariant
        m=cfg['spatial_filters_per_band'];self.segments=cfg['time_segments']
        self.spatial=nn.Conv2d(6,6*m,(16,1),groups=6,bias=False)
        self.norm=nn.BatchNorm2d(6*m,momentum=.1)
        self.drop=nn.Dropout(cfg['dropout']);self.classifier=nn.Linear(6*m*self.segments,2)
        self.register_buffer('mirror_indices',torch.tensor(MIRROR,dtype=torch.long))

    def _representation(self,A,B):
        x=F.silu(self.norm(self.spatial(A))).squeeze(2)
        x=x.reshape(len(x),x.shape[1],self.segments,-1)
        feature=torch.log(torch.clamp(x.var(dim=-1,unbiased=False),min=1e-6))
        if self.context:
            # Use the SAME source-trained spatial/nonlinear map for both inputs.
            z=F.silu(self.norm(self.spatial(B))).squeeze(2)
            before=torch.log(torch.clamp(z.var(dim=-1,unbiased=False),min=1e-6))
            feature=feature-before[:,:,None]
        return feature.flatten(1)

    def forward(self,A,B):
        if not self.equivariant:
            return self.classifier(self.drop(self._representation(A,B)))
        original=len(A)
        A=torch.cat([A,A[:,:,self.mirror_indices]],dim=0)
        B=torch.cat([B,B[:,:,self.mirror_indices]],dim=0)
        raw=self.classifier(self.drop(self._representation(A,B)))
        return .5*(raw[:original]+raw[original:,torch.tensor([1,0])])

    @torch.no_grad()
    def constrain(self):
        for weights,axes,limit in [(self.spatial.weight,(1,2,3),2.),(self.classifier.weight,(1,),.5)]:
            norms=torch.linalg.vector_norm(weights,dim=axes,keepdim=True)
            weights.mul_(torch.clamp(limit/(norms+1e-8),max=1))


def make_model(spec,plan):
    if spec['architecture']=='filterbank_variance':
        return FilterBankVariance(plan,spec['context'],spec['symmetry']=='equivariant')
    model=EEGNet(16,384 if spec['architecture']=='eegnet_reference' else 448,dropout=.5)
    if spec['architecture']!='eegnet_reference':
        for module in model.modules():
            if isinstance(module,nn.BatchNorm2d):module.momentum=.1
    return model


def inputs(cohort,rows,spec):
    if spec['architecture']=='filterbank_variance':
        A=np.asarray(cohort.active[rows,2:],dtype=np.float32)
        B=np.asarray(cohort.baseline[rows,2:],dtype=np.float32)
        scale=np.sqrt(np.mean(A*A,axis=(1,2,3),keepdims=True,dtype=np.float64)).astype(np.float32)
        # The SAME scalar scales task and its baseline; preserves their ratio.
        A=A/np.maximum(scale,1e-6);B=B/np.maximum(scale,1e-6)
    else:
        reference=spec['architecture']=='eegnet_reference'
        A=np.asarray(cohort.active[rows,1 if reference else 0],dtype=np.float32)
        if reference:A=A[...,64:]
        else:A=A/np.maximum(np.sqrt(np.mean(A*A,axis=(1,2),keepdims=True,dtype=np.float64)),1e-6)
        A=np.asarray(A,dtype=np.float32);B=np.empty((len(A),0),dtype=np.float32)
    return np.ascontiguousarray(A),np.ascontiguousarray(B)


@torch.no_grad()
def probabilities(model,A,B,spec,batch_size=128):
    model.eval();result=[]
    for start in range(0,len(A),batch_size):
        a=torch.from_numpy(A[start:start+batch_size]);b=torch.from_numpy(B[start:start+batch_size])
        logits=model(a,b) if spec['architecture']=='filterbank_variance' else model(a[:,None])
        result.append(torch.softmax(logits,dim=1).numpy())
    return binary_probabilities(np.concatenate(result))


class NeuralPredictor:
    def __init__(self,spec,plan,model,scaler=None):
        self.spec=copy.deepcopy(spec);self.plan=copy.deepcopy(plan);self.model=model.eval();self.scaler=scaler

    def predict_proba(self,cohort,rows):
        A,B=inputs(cohort,rows,self.spec)
        if self.scaler is not None:A=self.scaler.transform(A)
        return probabilities(self.model,A,B,self.spec)

    def save(self,path,subjects,seed):
        torch.save({'spec':self.spec,'neural_settings':self.plan['neural'],'state_dict':self.model.state_dict(),
                    'scaler_mean':None if self.scaler is None else torch.from_numpy(self.scaler.mean_),
                    'scaler_scale':None if self.scaler is None else torch.from_numpy(self.scaler.scale_),
                    'train_subjects':list(map(int,subjects)),'seed':int(seed),'epochs':self.spec['epochs']},path)


def load_neural(path):
    cp=torch.load(path,map_location='cpu',weights_only=True);plan={'neural':cp['neural_settings']}
    model=make_model(cp['spec'],plan);model.load_state_dict(cp['state_dict']);scaler=None
    if cp['scaler_mean'] is not None:
        scaler=TrainOnlyScaler();scaler.mean_=cp['scaler_mean'].numpy();scaler.scale_=cp['scaler_scale'].numpy()
    return NeuralPredictor(cp['spec'],plan,model.eval(),scaler),cp


def fit_neural(cohort,rows,spec,plan,directory,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    A,B=inputs(cohort,rows,spec);y=cohort.meta.iloc[rows].label.to_numpy(dtype=np.int64)
    subjects=cohort.meta.iloc[rows].subject.to_numpy();people=np.unique(subjects)
    source_group=np.searchsorted(people,subjects)
    scaler=None;reference=spec['architecture']=='eegnet_reference'
    if spec['architecture']!='filterbank_variance':
        scaler=TrainOnlyScaler().fit(A);A=scaler.transform(A)
    model=make_model(spec,plan)
    optimizer=(torch.optim.Adam(model.parameters(),lr=plan['neural']['learning_rate']) if reference else
               torch.optim.AdamW(model.parameters(),lr=plan['neural']['learning_rate'],weight_decay=plan['neural']['weight_decay']))
    q=np.ones(len(people))/len(people);history=[];start_time=time.monotonic()
    for epoch in range(1,spec['epochs']+1):
        model.train();order=rng.permutation(len(A));losses=[]
        for offset in range(0,len(order),plan['neural']['batch_size']):
            indices=order[offset:offset+plan['neural']['batch_size']]
            a=torch.from_numpy(A[indices]);b=torch.from_numpy(B[indices]);labels=torch.from_numpy(y[indices])
            if spec['symmetry']=='flip':
                chosen=torch.from_numpy(rng.random(len(indices))<.5)
                if spec['architecture']=='filterbank_variance':
                    a[chosen]=a[chosen][:,:,MIRROR];b[chosen]=b[chosen][:,:,MIRROR]
                else:
                    # Mirror in raw-normalized coordinates, then apply source scaler.
                    mirrored=(a[chosen].numpy()*scaler.scale_+scaler.mean_)[:,MIRROR]
                    a[chosen]=torch.from_numpy(scaler.transform(mirrored))
                labels[chosen]=1-labels[chosen]
            optimizer.zero_grad(set_to_none=True)
            logits=model(a,b) if spec['architecture']=='filterbank_variance' else model(a[:,None])
            individual=F.cross_entropy(logits,labels,reduction='none')
            if spec['dro']:
                groups=source_group[indices];present=np.unique(groups)
                group_losses=torch.stack([individual[torch.from_numpy(groups==g)].mean() for g in present])
                q[present]*=np.exp(np.clip(plan['neural']['dro_eta']*group_losses.detach().numpy(),-20,20));q/=q.sum()
                weights=torch.from_numpy((q[present]/q[present].sum()).astype(np.float32))
                loss=torch.sum(weights*group_losses)
            else:loss=individual.mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();model.constrain()
            losses.append(float(loss.detach()))
        history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-start_time})
        if epoch%10==0 or epoch==spec['epochs']:
            print(f"  {spec['name']} seed={seed} epoch={epoch}/{spec['epochs']} loss={history[-1]['training_loss']:.4f}",flush=True)
    pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
    if spec['dro']:
        pd.DataFrame({'subject':people,'source_group_weight':q}).to_csv(directory/'source_group_weights.csv',index=False)
    result=NeuralPredictor(spec,plan,model.eval(),scaler);result.save(directory/'model.pt',people,seed)
    return result
