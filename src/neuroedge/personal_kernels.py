"""Task-only features and source-prior residual kernel logistic calibration."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit


def features(cohort,rows):
    cov=np.asarray(cohort.task_cov[rows],dtype=np.float64)
    diagonal=np.diagonal(cov,axis1=-2,axis2=-1)
    power=np.log(np.maximum(diagonal,1e-12))
    power-=power.mean(axis=(-2,-1),keepdims=True)
    spectral=power.reshape(len(rows),-1)
    current=cov[:,0];trace=np.trace(current,axis1=-2,axis2=-1)
    if not np.isfinite(cov).all() or (trace<=0).any():raise ValueError('Invalid task covariance.')
    current=current/(trace[...,None,None]/16)
    current=.9*current+.1*np.eye(16)
    values,vectors=np.linalg.eigh(current)
    logged=(vectors*np.log(np.maximum(values,1e-10))[...,None,:])@np.swapaxes(vectors,-2,-1)
    i,j=np.triu_indices(16);geometry=logged[...,i,j]*np.where(i==j,1.,np.sqrt(2.))
    return {'spectral':spectral,'geometry':geometry.reshape(len(rows),-1)}


def fit_scaler(values):
    result={}
    for name,x in values.items():
        scale=x.std(axis=0);scale=np.where(scale<1e-6,1.,scale)
        result[name+'_mean']=x.mean(axis=0);result[name+'_scale']=scale
    return result


def transform(values,scaler):
    return {name:(x-scaler[name+'_mean'])/scaler[name+'_scale'] for name,x in values.items()}


def kernel(a,b,kind):
    if kind=='hybrid':return .5*(kernel(a,b,'spectral')+kernel(a,b,'geometry'))
    x=a[kind];y=b[kind]
    distance=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(y*y,axis=1)[None,:]-2*x@y.T,0.)/x.shape[1]
    return np.exp(-distance)


def subset(values,indices):return {name:x[indices] for name,x in values.items()}


def objective(theta,design,z,y,C,sigma):
    n=len(y);u=theta[:-1];b=theta[-1];current=z+design@u+b
    weights=np.where(y==0,.5/np.sum(y==0),.5/np.sum(y==1))
    loss=np.sum(weights*(np.logaddexp(0,current)-y*current))
    loss+=u@u/(2*n*C)+b*b/(2*n*sigma*sigma)
    residual=weights*(expit(current)-y)
    gradient=np.r_[design.T@residual+u/(n*C),residual.sum()+b/(n*sigma*sigma)]
    return float(loss),gradient


def fit_residual(K,p,y,C,config):
    p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=int);n=len(y)
    if set(y)!={0,1}:return {'alpha':np.zeros(n),'intercept':0.,'status':'single_class_keep_source'}
    K=.5*(K+K.T);L=np.linalg.cholesky(K+config['kernel_jitter']*np.eye(n))
    inverse=np.linalg.solve(L.T,np.eye(n));design=K@inverse
    z=logit(np.clip(p,config['probability_clip'],1-config['probability_clip']))
    fit=minimize(objective,np.zeros(n+1),args=(design,z,y,C,config['intercept_prior_standard_deviation']),
        jac=True,method='L-BFGS-B',options={'maxiter':config['max_iterations'],'ftol':1e-12,'gtol':1e-8})
    if not fit.success:raise RuntimeError('Residual calibration failed: '+fit.message)
    return {'alpha':inverse@fit.x[:-1],'intercept':float(fit.x[-1]),'status':'support_only_fit'}


def predict_residual(K,p,fit,config):
    p=np.asarray(p,dtype=float)
    if fit['status']=='single_class_keep_source':return p.copy()
    z=logit(np.clip(p,config['probability_clip'],1-config['probability_clip']))
    return expit(z+K@fit['alpha']+fit['intercept'])
