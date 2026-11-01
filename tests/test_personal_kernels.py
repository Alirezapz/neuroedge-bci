"""Synthetic software checks, never research performance."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from neuroedge.personal_kernels import (features,fit_scaler,transform,subset,kernel,
    objective,fit_residual,predict_residual)
from neuroedge.residual_study import predict_groups

CONFIG={'probability_clip':1e-6,'intercept_prior_standard_deviation':4.,'kernel_jitter':1e-6,'max_iterations':1000}


def test_task_features_and_source_only_standardization():
    rng=np.random.default_rng(8);x=rng.normal(size=(10,3,8,16,20));cov=x@np.swapaxes(x,-2,-1)/20
    cohort=SimpleNamespace(task_cov=cov,baseline_cov=np.zeros((10,8,16,16)))
    f=features(cohort,np.arange(10));assert f['spectral'].shape==(10,384) and f['geometry'].shape==(10,1088)
    scaler=fit_scaler(subset(f,np.arange(6)));cohort.baseline_cov[:]=1e9
    again=features(cohort,np.arange(10))
    for k in f:np.testing.assert_array_equal(f[k],again[k])
    for kind in ['spectral','geometry','hybrid']:
        a=transform(f,scaler);K=kernel(a,a,kind)
        np.testing.assert_allclose(np.diag(K),1.,atol=1e-12)
        assert np.linalg.eigvalsh(K).min()>-1e-9
    altered={k:v.copy() for k,v in f.items()}
    for x in altered.values():x[6:]*=1e6
    recomputed=fit_scaler(subset(altered,np.arange(6)))
    for k in scaler:np.testing.assert_array_equal(scaler[k],recomputed[k])


def test_kernel_logistic_analytic_gradient():
    rng=np.random.default_rng(7);D=rng.normal(size=(5,5));z=rng.normal(size=5);y=np.array([0,1,0,1,0]);theta=rng.normal(size=6)
    error=check_grad(lambda t:objective(t,D,z,y,.3,4.)[0],lambda t:objective(t,D,z,y,.3,4.)[1],theta)
    assert error<1e-6


def test_residual_support_fit_and_single_class_retention():
    x=np.r_[np.ones((4,2))*(-1),np.ones((4,2))];v={'spectral':x,'geometry':x}
    K=kernel(v,v,'hybrid');prior=np.full(8,.75);labels=np.r_[np.zeros(4),np.ones(4)].astype(int)
    fit=fit_residual(K,prior,labels,1.,CONFIG);p=predict_residual(K,prior,fit,CONFIG)
    assert p[:4].mean()<p[4:].mean() and np.isfinite(p).all()
    single=fit_residual(K,prior,np.zeros(8,dtype=int),1.,CONFIG)
    np.testing.assert_array_equal(predict_residual(K,prior,single,CONFIG),prior)


def test_future_labels_never_change_personal_fit():
    rng=np.random.default_rng(11);n=16
    base=pd.DataFrame({'trial_id':[f'FIXTURE_{i}' for i in range(n)],'subject':np.repeat([1001,1002],8),
        'session':np.tile(np.repeat([1,2],4),2),'run':1,'onset':np.arange(n),
        'label':np.tile([0,1],8),'prob_right':rng.uniform(.2,.8,n)})
    values={'spectral':rng.normal(size=(n,5)),'geometry':rng.normal(size=(n,7))}
    plan={'calibration_session':1,'test_session':2,'adaptation':CONFIG}
    a,receipts=predict_groups(plan,base,values,'hybrid',1.,4)
    altered=base.copy();altered.loc[altered.session==2,'label']=1-altered.loc[altered.session==2,'label']
    b,other=predict_groups(plan,altered,values,'hybrid',1.,4)
    assert receipts==other
    np.testing.assert_array_equal(a.prob_right,b.prob_right)
