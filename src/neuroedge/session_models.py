"""Support-only feature maps and source-anchored adaptation.

Predictors accept covariance arrays and fixed source scores, never test labels.
No transform estimates any statistic from the prediction batch.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .personal_kernels import fit_residual, predict_residual
from .personalization import fit_intercept, adapt

BIAS = {'probability_clip': 1e-6, 'prior_standard_deviation': 4., 'bounds': [-16.,16.]}
RESIDUAL = {'probability_clip':1e-6, 'intercept_prior_standard_deviation':4.,
            'kernel_jitter':1e-6, 'max_iterations':1000}
TUNED = ['spectral_logistic','tangent_logistic','tangent_rbf_svm',
         'anchored_tangent','anchored_dual']
BASE = ['source_ensemble','source_eegnet','intercept','csp_lda','fbcsp_lda']
METHODS = BASE + TUNED + ['donor_selected_fusion']


def symmetric_function(C, function):
    values,vectors=np.linalg.eigh(np.asarray(C,dtype=np.float64))
    return (vectors*function(np.maximum(values,1e-12))[...,None,:])@np.swapaxes(vectors,-1,-2)


def spd(C):
    C=np.asarray(C,dtype=np.float64)
    if not np.isfinite(C).all():raise ValueError('Nonfinite covariance')
    trace=np.trace(C,axis1=-2,axis2=-1)
    if (trace<=0).any():raise ValueError('Nonpositive covariance trace')
    return .9*C/(trace[...,None,None]/C.shape[-1])+.1*np.eye(C.shape[-1])


def spectral(C):
    power=np.log(np.maximum(np.diagonal(C,axis1=-2,axis2=-1),1e-12))
    power-=power.mean(axis=(-1,-2),keepdims=True)
    return power.reshape(len(C),-1)


class LocalFeatures:
    """Support arithmetic-mean whitening, tangent and shrunk-variance scaling."""
    def fit(self,C):
        self.whitening=symmetric_function(spd(C[:,0,2:]).mean(axis=0),lambda x:x**-.5)
        blocks=self.raw(C)
        self.means={k:v.mean(axis=0) for k,v in blocks.items()}
        self.scales={}
        for k,v in blocks.items():
            variance=v.var(axis=0)
            self.scales[k]=np.sqrt(np.maximum(.5*variance+.5*variance.mean(),1e-8))
        return self

    def raw(self,C):
        cov=spd(C[:,0,2:]);W=self.whitening
        logged=symmetric_function(W@cov@W,np.log)
        i,j=np.triu_indices(C.shape[-1])
        tangent=(logged[...,i,j]*np.where(i==j,1.,np.sqrt(2))).reshape(len(C),-1)
        return {'tangent':tangent,'spectral':spectral(C)}

    def transform(self,C):
        return {k:(v-self.means[k])/self.scales[k] for k,v in self.raw(C).items()}


def linear_kernel(a,b,kind):
    if kind=='dual':return .5*(linear_kernel(a,b,'tangent')+linear_kernel(a,b,'spectral'))
    return a[kind]@b[kind].T/a[kind].shape[1]


class CSPFeatures:
    def __init__(self,filterbank):self.filterbank=filterbank

    def fit(self,C,y):
        self.bands=list(range(2,8)) if self.filterbank else [1]
        self.filters=[]
        for band in self.bands:
            classes=[spd(C[y==label,0,band]).mean(axis=0) for label in [0,1]]
            _,vectors=eigh(classes[1],classes[0]+classes[1])
            self.filters.append(vectors[:,[0,1,-2,-1]].T)
        return self

    def transform(self,C):
        output=[]
        for band,W in zip(self.bands,self.filters):
            covariance=C[:,:,band] if self.filterbank else C[:,0:1,band]
            energy=np.einsum('kc,ntcd,kd->ntk',W,covariance,W,optimize=True)
            output.append(np.log(np.maximum(energy,1e-12)).reshape(len(C),-1))
        return np.concatenate(output,axis=1)


class SessionModel:
    def __init__(self,method,C=None):self.method=method;self.C=C

    def fit(self,cov,source,y):
        self.single_class=len(np.unique(y))<2
        self.prior=float(np.mean(y));method=self.method
        if method in ['source_ensemble','source_eegnet']:return self
        if method=='intercept':
            self.intercept,self.status=fit_intercept(source,y,BIAS);return self
        if self.single_class:return self
        if method in ['csp_lda','fbcsp_lda']:
            self.features=CSPFeatures(method=='fbcsp_lda').fit(cov,y)
            self.classifier=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(self.features.transform(cov),y)
        else:
            self.features=LocalFeatures().fit(cov);values=self.features.transform(cov)
            if method.startswith('anchored_'):
                self.kind=method.removeprefix('anchored_');self.support_features=values
                self.residual=fit_residual(linear_kernel(values,values,self.kind),source,y,self.C,RESIDUAL)
            else:
                self.kind='spectral' if method=='spectral_logistic' else 'tangent'
                self.classifier=(SVC(C=self.C,kernel='rbf',gamma='scale') if method=='tangent_rbf_svm' else
                    LogisticRegression(C=self.C,class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=7))
                self.classifier.fit(values[self.kind],y)
        return self

    def predict(self,cov,source):
        if self.method in ['source_ensemble','source_eegnet']:return np.asarray(source).copy()
        if self.method=='intercept':return adapt(source,self.intercept,BIAS)[:,1]
        if self.single_class:
            return np.asarray(source).copy() if self.method.startswith('anchored_') else np.full(len(cov),self.prior)
        values=self.features.transform(cov)
        if self.method.startswith('anchored_'):
            return predict_residual(linear_kernel(values,self.support_features,self.kind),source,self.residual,RESIDUAL)
        if self.method in ['csp_lda','fbcsp_lda']:return self.classifier.predict_proba(values)[:,1]
        if self.method=='tangent_rbf_svm':return expit(self.classifier.decision_function(values[self.kind]))
        return self.classifier.predict_proba(values[self.kind])[:,1]
