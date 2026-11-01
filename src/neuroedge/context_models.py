"""Matched active/ERD features, reflection controls and source-OOF calibration."""
from __future__ import annotations

import copy
from pathlib import Path
import joblib
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pyriemann.tangentspace import TangentSpace

from .context_data import MIRROR
from .context_protocol import binary_probabilities
from .frontier_data import shrink_covariances


def mirror_cov(C):
    return C[...,MIRROR,:][...,:,MIRROR]


def inverse_sqrt(C):
    values,vectors=np.linalg.eigh(np.asarray(C,dtype=np.float64))
    if (values<=0).any():raise ValueError('Context covariance is not SPD.')
    return (vectors*values[...,None,:]**-.5)@np.swapaxes(vectors,-1,-2)


def covariance_inputs(cohort,rows):
    return (np.asarray(cohort.task_cov[rows,:,2:],dtype=np.float64),
            np.asarray(cohort.baseline_cov[rows,2:],dtype=np.float64))


def feature_mode(active,baseline,mode):
    erd=active-baseline
    if mode=='active':return active
    if mode=='erd':return erd
    if mode=='hybrid':return np.concatenate([active,erd],axis=1)
    raise ValueError(mode)


class ClassicalPredictor:
    def __init__(self,spec,seed=7):
        self.spec=copy.deepcopy(spec);self.seed=seed

    def _features(self,A,B,fit=False,y=None):
        family=self.spec['family'];mode=self.spec['mode']
        if family=='power':
            active=np.log(np.maximum(np.diagonal(A,axis1=-2,axis2=-1),1e-8)).reshape(len(A),-1)
            base=np.log(np.maximum(np.diagonal(B,axis1=-2,axis2=-1),1e-8))
            before=np.repeat(base[:,None],3,axis=1).reshape(len(A),-1)
            return feature_mode(active,before,mode)
        if family=='fbcsp':
            if fit:
                self.filters=[]
                for band in range(6):
                    classes=[shrink_covariances(A[y==label,0,band],self.spec['shrinkage']).mean(axis=0) for label in [0,1]]
                    eigenvalues,vectors=eigh(classes[1],classes[0]+classes[1])
                    self.filters.append(vectors[:,[0,1,-2,-1]].T)
            active=[];before=[]
            for band,W in enumerate(self.filters):
                energy=np.einsum('kc,ntcd,kd->ntk',W,A[:,:,band],W,optimize=True)
                rest=np.einsum('kc,ncd,kd->nk',W,B[:,band],W,optimize=True)
                active.append(np.log(np.maximum(energy,1e-8)).reshape(len(A),-1))
                before.append(np.repeat(np.log(np.maximum(rest,1e-8))[:,None],3,axis=1).reshape(len(A),-1))
            return feature_mode(np.concatenate(active,axis=1),np.concatenate(before,axis=1),mode)
        if family=='tangent':
            blocks=[]
            if fit:self.tangents=[]
            kinds=['active'] if mode=='active' else ['erd'] if mode=='erd' else ['active','erd']
            for kind in kinds:
                for band in range(6):
                    task=shrink_covariances(A[:,0,band],self.spec['shrinkage'])
                    if kind=='erd':
                        rest=shrink_covariances(B[:,band],self.spec['shrinkage'])
                        W=inverse_sqrt(rest);task=W@task@W
                    task=task/np.trace(task,axis1=-2,axis2=-1)[:,None,None]
                    if fit:
                        transform=TangentSpace(metric='logeuclid',tsupdate=False).fit(task)
                        self.tangents.append(transform)
                    transform=self.tangents[len(blocks)]
                    blocks.append(transform.transform(task))
            return np.concatenate(blocks,axis=1)
        raise ValueError(family)

    def fit(self,cohort,rows):
        y=cohort.meta.iloc[rows].label.to_numpy(dtype=int)
        self.train_subjects=sorted(set(cohort.meta.iloc[rows].subject.tolist()))
        if self.spec['family']=='csp_reference':
            from mne.decoding import CSP
            X=np.asarray(cohort.active[rows,1,:,64:],dtype=np.float64)
            self.reference=make_pipeline(CSP(n_components=4,reg=.1,log=True,norm_trace=False),
                LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')).fit(X,y)
            return self
        A,B=covariance_inputs(cohort,rows)
        if self.spec['symmetry']!='none':
            A=np.concatenate([A,mirror_cov(A)]);B=np.concatenate([B,mirror_cov(B)]);y=np.r_[y,1-y]
        features=self._features(A,B,fit=True,y=y)
        classifier=(LogisticRegression(C=self.spec['C'],max_iter=2500,random_state=self.seed)
                    if self.spec['classifier']=='logistic' else
                    SVC(C=self.spec['C'],kernel='rbf',probability=True,random_state=self.seed))
        self.classifier=make_pipeline(StandardScaler(),classifier).fit(features,y)
        return self

    def predict_proba(self,cohort,rows):
        if self.spec['family']=='csp_reference':
            return binary_probabilities(self.reference.predict_proba(np.asarray(cohort.active[rows,1,:,64:],dtype=np.float64)))
        A,B=covariance_inputs(cohort,rows)
        p=self.classifier.predict_proba(self._features(A,B))
        if self.spec['symmetry']=='equivariant':
            mirrored=self.classifier.predict_proba(self._features(mirror_cov(A),mirror_cov(B)))
            p=.5*(p+mirrored[:,::-1])
        return binary_probabilities(p)

    def save(self,path):joblib.dump(self,path)


def fit_temperature(probabilities,y):
    p=np.clip(np.asarray(probabilities),1e-6,1-1e-6);z=logit(p)
    y=np.asarray(y,dtype=int)
    def objective(log_temperature):
        prediction=expit(z/np.exp(log_temperature))
        return float(-np.mean(y*np.log(np.clip(prediction,1e-12,1))+(1-y)*np.log(np.clip(1-prediction,1e-12,1))))
    fit=minimize_scalar(objective,bounds=(-3.,3.),method='bounded',options={'xatol':1e-10})
    if not fit.success:raise RuntimeError('Source-OOF temperature fit failed.')
    return float(np.exp(fit.x))


def apply_temperature(probabilities,temperature):
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Invalid temperature.')
    p=expit(logit(np.clip(np.asarray(probabilities),1e-6,1-1e-6))/temperature)
    return binary_probabilities(np.column_stack([1-p,p]))


def load_classical(path):return joblib.load(path)
