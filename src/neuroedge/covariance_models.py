"""Bounded covariance candidate grid; labels are accepted only at fitting."""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from .config import digest
from .session_models import symmetric_function,RESIDUAL
from .personal_kernels import fit_residual,predict_residual


def candidates():
    result=[]
    def add(family,**p):
        s={'family':family,**p};s['name']=family+'_'+digest(s)[:10];result.append(s)
    for shrink in [.01,.1]:
        for components in [4,8]:add('fbcsp_lda',shrink=shrink,components=components)
        for C in [.1,1.,10.]:
            for gamma in [.25,1.,4.]:
                add('fbcsp_svm',shrink=shrink,components=4,C=C,gamma=gamma)
                add('tangent_svm',shrink=shrink,C=C,gamma=gamma)
        for C in [.01,.1,1.]:add('tangent_logistic',shrink=shrink,C=C)
    for C in [.1,1.,10.]:
        for keep in [12,24,48]:add('selected_fbcsp',shrink=.05,components=8,C=C,gamma=1.,keep=keep)
        for gamma in [.25,1.,4.]:add('joint_svm',shrink=.05,components=4,C=C,gamma=gamma)
        add('power_means',shrink=.05,C=C)
    for C in [.01,.1,1.,10.]:add('anchored_multiview',shrink=.05,components=4,C=C)
    assert len({s['name'] for s in result})==len(result)
    return result


def normalize(C,shrink):
    C=np.asarray(C,dtype=np.float64);trace=np.trace(C,axis1=-2,axis2=-1)
    if not np.isfinite(C).all() or (trace<=0).any():raise ValueError('Invalid covariance')
    n=C.shape[-1]
    return (1-shrink)*C/(trace[...,None,None]/n)+shrink*np.eye(n)


class BatchTransport:
    def fit(self,C):
        self.support_root=symmetric_function(normalize(C,.01).mean(axis=0),np.sqrt)
        return self

    def transform(self,C):
        cov=normalize(C,.01)
        inverse=symmetric_function(cov.mean(axis=0),lambda x:x**-.5)
        W=self.support_root@inverse
        return W@cov@np.swapaxes(W,-1,-2)


class FeatureMap:
    def __init__(self,spec):self.spec=spec

    def fit(self,C,y):
        cov=normalize(C,self.spec['shrink']);family=self.spec['family'];self.filters=[]
        if family in ['fbcsp_lda','fbcsp_svm','selected_fbcsp','joint_svm','anchored_multiview']:
            k=self.spec['components']//2
            for band in range(2,8):
                means=[cov[y==c,0,band].mean(axis=0) for c in [0,1]]
                _,V=eigh(means[1],means[0]+means[1]);self.filters.append(V[:,np.r_[np.arange(k),np.arange(C.shape[-1]-k,C.shape[-1])]].T)
        if family in ['tangent_logistic','tangent_svm','joint_svm','anchored_multiview']:
            self.W=symmetric_function(cov[:,0,2:].mean(axis=0),lambda x:x**-.5)
        if family=='power_means':
            self.prototypes=[]
            for label in [0,1]:
                A=cov[y==label,0,2:];logged=symmetric_function(A,np.log)
                values,vectors=np.linalg.eigh(logged.mean(axis=0))
                # A matrix logarithm is symmetric, not positive definite.
                # Its negative eigenvalues must survive matrix exponentiation.
                geometric=(vectors*np.exp(values)[...,None,:])@np.swapaxes(vectors,-1,-2)
                means=[symmetric_function(np.mean(symmetric_function(A,lambda x:1/x),axis=0),lambda x:1/x),
                       geometric,A.mean(axis=0)]
                self.prototypes.append(np.stack([symmetric_function(m,np.log) for m in means]))
        raw=self.raw(C);self.means={k:v.mean(axis=0) for k,v in raw.items()};self.scales={}
        for k,v in raw.items():
            var=v.var(axis=0);self.scales[k]=np.sqrt(np.maximum(.9*var+.1*var.mean(),1e-8))
        return self

    def raw(self,C):
        cov=normalize(C,self.spec['shrink']);blocks={};family=self.spec['family']
        if self.filters:
            values=[]
            for band,W in zip(range(2,8),self.filters):
                energy=np.einsum('kc,ntcd,kd->ntk',W,cov[:,:,band],W,optimize=True)
                values.append(np.log(np.maximum(energy,1e-12)).reshape(len(C),-1))
            blocks['csp']=np.concatenate(values,axis=1)
        if family in ['tangent_logistic','tangent_svm','joint_svm','anchored_multiview']:
            logged=symmetric_function(self.W@cov[:,0,2:]@self.W,np.log)
            i,j=np.triu_indices(C.shape[-1]);blocks['tangent']=(logged[...,i,j]*np.where(i==j,1.,np.sqrt(2.))).reshape(len(C),-1)
        if family=='power_means':
            logged=symmetric_function(cov[:,0,2:],np.log)
            blocks['means']=np.concatenate([np.sum((logged[:,None]-p[None])**2,axis=(-2,-1)).reshape(len(C),-1) for p in self.prototypes],axis=1)
        return blocks

    def transform(self,C):return {k:(v-self.means[k])/self.scales[k] for k,v in self.raw(C).items()}


def design(values,family):
    if family=='joint_svm':return np.concatenate([v/np.sqrt(v.shape[1]) for v in values.values()],axis=1)
    return next(iter(values.values()))


def multiview_kernel(a,b):
    x=a['tangent'];y=b['tangent'];d=np.maximum(np.sum(x*x,1)[:,None]+np.sum(y*y,1)[None,:]-2*x@y.T,0)/x.shape[1]
    return .5*(np.exp(-d)+a['csp']@b['csp'].T/a['csp'].shape[1])


class CovarianceModel:
    def __init__(self,spec,condition):self.spec=spec.copy();self.condition=condition

    def fit(self,C,y,source):
        self.single_class=len(np.unique(y))<2;self.prior=float(np.mean(y))
        if self.single_class:return self
        self.transport=BatchTransport().fit(C) if self.condition=='batch_unlabeled' else None
        # Support is unchanged in both arms; prediction batch is transported.
        self.features=FeatureMap(self.spec).fit(C,y);values=self.features.transform(C);family=self.spec['family']
        if family=='anchored_multiview':
            self.support_features=values;self.residual=fit_residual(multiview_kernel(values,values),source,y,self.spec['C'],RESIDUAL)
        else:
            X=design(values,family);self.keep=None
            if family=='selected_fbcsp':
                f,_=f_classif(X,y);self.keep=np.argsort(-np.nan_to_num(f,nan=0.,posinf=1e15),kind='stable')[:self.spec['keep']];X=X[:,self.keep]
            if family=='fbcsp_lda':self.classifier=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
            elif family in ['tangent_logistic','power_means']:self.classifier=LogisticRegression(C=self.spec['C'],class_weight='balanced',max_iter=2000,random_state=7)
            else:self.classifier=SVC(C=self.spec['C'],gamma=self.spec['gamma']/(X.shape[1]*max(float(X.var()),1e-12)),kernel='rbf',class_weight='balanced')
            self.classifier.fit(X,y)
        return self

    def predict(self,C,source):
        if self.single_class:return np.asarray(source).copy() if self.spec['family']=='anchored_multiview' else np.full(len(C),self.prior)
        if self.transport is not None:C=self.transport.transform(C)
        values=self.features.transform(C);family=self.spec['family']
        if family=='anchored_multiview':return predict_residual(multiview_kernel(values,self.support_features),source,self.residual,RESIDUAL)
        X=design(values,family)
        if self.keep is not None:X=X[:,self.keep]
        if isinstance(self.classifier,SVC):return expit(self.classifier.decision_function(X))
        return self.classifier.predict_proba(X)[:,1]
