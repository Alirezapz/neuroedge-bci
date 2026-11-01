"""Compact neural adaptations with strictly source-only checkpoint selection.

ShallowLogPower and TemporalAttention are independent compact adaptations
inspired by Schirrmeister 2017 and Song 2023, not exact published models.
Every discovery OOF prediction comes from a full source-only refit: inner
source participants choose epochs, then all source participants are refitted.
"""
from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .models import EEGNet, TrainOnlyScaler


class ShallowLogPower(nn.Module):
    def __init__(self, channels, samples, dropout=.5):
        super().__init__()
        self.temporal = nn.Conv2d(1,24,(1,25),padding=(0,12),bias=False)
        self.spatial = nn.Conv2d(24,48,(channels,1),groups=24,bias=False)
        self.norm = nn.BatchNorm2d(48,momentum=.1)
        self.pool = nn.AvgPool2d((1,48),stride=(1,16))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(48*((samples-48)//16+1),2)

    def forward(self,X):
        X = self.norm(self.spatial(self.temporal(X)))
        X = torch.log(torch.clamp(self.pool(X.square()),min=1e-6))
        return self.classifier(self.drop(X.flatten(1)))


class TemporalAttention(nn.Module):
    def __init__(self, channels, samples, dropout=.5):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(1,16,(1,25),padding=(0,12),bias=False),
            nn.Conv2d(16,32,(channels,1),groups=16,bias=False),
            nn.BatchNorm2d(32,momentum=.1),nn.GELU(),
            nn.AvgPool2d((1,32),stride=(1,12)),nn.Dropout(dropout),
            nn.Conv2d(32,40,(1,1)))
        layer = nn.TransformerEncoderLayer(d_model=40,nhead=4,dim_feedforward=80,
                                          dropout=.2,activation="gelu",batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        tokens = (samples-32)//12+1
        self.position = nn.Parameter(torch.zeros(1,tokens,40))
        self.classifier = nn.Sequential(nn.LayerNorm(40),nn.Linear(40,2))

    def forward(self,X):
        X = self.patch(X).squeeze(2).transpose(1,2)
        return self.classifier(self.encoder(X+self.position).mean(dim=1))


def make_model(spec, samples):
    architecture = spec["architecture"]
    if architecture in {"eegnet","eegnet_baseline"}:
        model = EEGNet(spec["channels"],samples,dropout=spec["dropout"])
        if architecture == "eegnet":
            for module in model.modules():
                if isinstance(module,nn.BatchNorm2d):
                    module.momentum = .1
        return model
    if architecture == "shallow":
        return ShallowLogPower(spec["channels"],samples,spec["dropout"])
    if architecture == "attention":
        return TemporalAttention(spec["channels"],samples,spec["dropout"])
    raise ValueError(architecture)


def trial_normalize(X, spec):
    X = np.asarray(X,dtype=np.float32)
    if spec["architecture"] != "eegnet_baseline":
        X = X / np.maximum(np.sqrt(np.mean(X.astype(np.float64)**2,axis=(1,2),keepdims=True)),1e-6)
    return np.ascontiguousarray(X,dtype=np.float32)


@torch.no_grad()
def probabilities(model, X, batch_size=128):
    model.eval()
    result=np.concatenate([torch.softmax(model(torch.from_numpy(X[i:i+batch_size,None])),dim=1).numpy()
                           for i in range(0,len(X),batch_size)])
    from .frontier_protocol import binary_probabilities
    return binary_probabilities(result)


def segment_recombine(X,y,rng):
    result = X.copy()
    slices = np.array_split(np.arange(X.shape[-1]),7)
    for label in [0,1]:
        rows = np.flatnonzero(y==label)
        if len(rows)==0:
            continue
        for segment in slices:
            source = rng.choice(rows,size=len(rows),replace=True)
            result[rows[:,None,None],np.arange(X.shape[1])[None,:,None],segment[None,None,:]] = \
                X[source[:,None,None],np.arange(X.shape[1])[None,:,None],segment[None,None,:]]
    return result


def _train(X,y,spec,plan,seed,epochs, validation=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = make_model(spec,X.shape[-1])
    baseline = spec["architecture"] == "eegnet_baseline"
    optimizer = (torch.optim.Adam(model.parameters(),lr=plan["neural"]["learning_rate"])
                 if baseline else torch.optim.AdamW(model.parameters(),lr=plan["neural"]["learning_rate"],
                                                    weight_decay=plan["neural"]["weight_decay"]))
    history, best_loss, best_epoch, stale = [], float("inf"), epochs, 0
    batch_size = plan["neural"]["batch_size"]
    for epoch in range(1,epochs+1):
        model.train()
        order = rng.permutation(len(X))
        epoch_X = segment_recombine(X,y,rng) if spec["augmentation"]=="segment_recombine" else X
        losses=[]
        for start in range(0,len(X),batch_size):
            indices = order[start:start+batch_size]
            xb = torch.from_numpy(epoch_X[indices,None])
            yb = torch.from_numpy(y[indices].astype(np.int64))
            optimizer.zero_grad(set_to_none=True)
            if spec["augmentation"]=="mixup":
                lam = float(rng.beta(plan["neural"]["mixup_alpha"],plan["neural"]["mixup_alpha"]))
                permutation = torch.from_numpy(rng.permutation(len(indices)))
                logits = model(lam*xb+(1-lam)*xb[permutation])
                loss = lam*F.cross_entropy(logits,yb)+(1-lam)*F.cross_entropy(logits,yb[permutation])
            else:
                loss = F.cross_entropy(model(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
            if hasattr(model,"constrain"):
                model.constrain()
            losses.append(float(loss.detach()))
        row={"epoch":epoch,"train_loss":float(np.mean(losses))}
        if validation is not None:
            val_X,val_y = validation
            p = probabilities(model,val_X)
            val_loss = float(-np.log(np.clip(p[np.arange(len(p)),val_y],1e-7,1)).mean())
            row["validation_loss"] = val_loss
            row["validation_accuracy"] = float(np.mean(p.argmax(1)==val_y))
            if val_loss < best_loss-1e-6:
                best_loss,best_epoch,stale = val_loss,epoch,0
            else:
                stale += 1
            if epoch>=plan["neural"]["minimum_epochs"] and stale>=plan["neural"]["patience"]:
                history.append(row)
                break
        history.append(row)
        if epoch%10==0:
            print(f"  {spec['name']} seed={seed} epoch={epoch}/{epochs} "
                  f"loss={row['train_loss']:.4f} "
                  f"inner_val_loss={row.get('validation_loss',float('nan')):.4f}",flush=True)
    return model.eval(),history,int(best_epoch)


class NeuralPredictor:
    def __init__(self,spec,model,scaler,epochs,seed):
        self.spec,self.model,self.scaler,self.epochs,self.seed=spec,model,scaler,epochs,seed

    def predict_proba(self,X):
        return probabilities(self.model,self.scaler.transform(trial_normalize(X,self.spec)))

    def save(self,path,train_subjects):
        torch.save({"spec":self.spec,"samples":self.model.spec["samples"] if isinstance(self.model,EEGNet)
                    else int(self.samples),"state_dict":self.model.state_dict(),
                    "scaler_mean":torch.from_numpy(self.scaler.mean_),
                    "scaler_scale":torch.from_numpy(self.scaler.scale_),
                    "epochs":self.epochs,"seed":self.seed,"train_subjects":list(map(int,train_subjects))},path)


def load_neural(path):
    cp = torch.load(path,map_location="cpu",weights_only=True)
    model = make_model(cp["spec"],cp["samples"])
    model.load_state_dict(cp["state_dict"])
    scaler=TrainOnlyScaler()
    scaler.mean_=cp["scaler_mean"].numpy()
    scaler.scale_=cp["scaler_scale"].numpy()
    predictor=NeuralPredictor(cp["spec"],model.eval(),scaler,cp["epochs"],cp["seed"])
    predictor.samples=cp["samples"]
    return predictor,cp


def fit_neural(X,y,subjects,spec,plan,directory,*,epochs=None,seed=7):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    normalized=trial_normalize(X,spec)
    if spec.get("fixed_epochs"):
        epochs=spec["fixed_epochs"]
    if epochs is None:
        people=np.unique(subjects).copy()
        np.random.default_rng(seed).shuffle(people)
        internal_validation=people[:max(1,int(np.ceil(len(people)*.2)))].tolist()
        validation=np.isin(subjects,internal_validation)
        if len(people)<3:
            raise ValueError("Need at least three source participants for inner epoch selection.")
        inner_scaler=TrainOnlyScaler().fit(normalized[~validation])
        _,history,epochs=_train(inner_scaler.transform(normalized[~validation]),y[~validation],spec,plan,
            seed,plan["neural"]["epochs"],
            validation=(inner_scaler.transform(normalized[validation]),y[validation]))
        pd.DataFrame(history).to_csv(directory/"inner_epoch_selection.csv",index=False)
        from .config import json_write
        json_write(directory/"inner_split.json",{"train_subjects":sorted(set(subjects[~validation].tolist())),
            "validation_subjects":sorted(internal_validation),"selected_epoch":int(epochs),
            "outer_validation_used":False})
    scaler=TrainOnlyScaler().fit(normalized)
    model,history,_=_train(scaler.transform(normalized),y,spec,plan,seed,int(epochs))
    pd.DataFrame(history).to_csv(directory/"refit_history.csv",index=False)
    result=NeuralPredictor(spec,model,scaler,int(epochs),seed)
    result.samples=X.shape[-1]
    result.save(directory/"model.pt",np.unique(subjects))
    return result
