"""Synthetic software checks; no fixture scores are scientific evidence."""
import numpy as np
import pandas as pd
import pytest
from neuroedge.session_models import LocalFeatures,SessionModel,linear_kernel
from neuroedge.session_study import support_test


def fixture_cov(seed,n=14):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,3,8,16,20))
    return np.einsum('...ct,...dt->...cd',X,X)/20


@pytest.mark.parametrize('method',['tangent_logistic','tangent_rbf_svm','anchored_tangent','anchored_dual','fbcsp_lda'])
def test_predictions_do_not_depend_on_other_test_trials(method):
    C=fixture_cov(21);T=fixture_cov(22,5);y=np.arange(len(C))%2
    fitted=SessionModel(method,.1).fit(C,np.linspace(.2,.8,len(C)),y)
    batch=fitted.predict(T,np.linspace(.1,.9,len(T)))
    single=fitted.predict(T[:1],np.array([.1]))
    np.testing.assert_allclose(single,batch[:1],atol=1e-10,rtol=1e-10)


def test_local_transform_is_orthogonal_sensor_invariant():
    # Tangent inner products are preserved under orthogonal sensor rotations.
    C=fixture_cov(23);rng=np.random.default_rng(24);Q=np.linalg.qr(rng.normal(size=(16,16)))[0]
    A=LocalFeatures().fit(C).raw(C)['tangent']
    B=LocalFeatures().fit(Q@C@Q.T).raw(Q@C@Q.T)['tangent']
    np.testing.assert_allclose(A@A.T,B@B.T,atol=1e-10,rtol=1e-10)


def test_dual_kernel_is_positive_semidefinite():
    C=fixture_cov(25);F=LocalFeatures().fit(C).transform(C);K=linear_kernel(F,F,'dual')
    assert np.linalg.eigvalsh(K).min()>-1e-10


def test_single_class_fallback_never_looks_ahead():
    C=fixture_cov(26,4);p=np.array([.1,.4,.6,.9])
    for method in ['anchored_dual','anchored_tangent','intercept']:
        fitted=SessionModel(method,.1).fit(C,p,np.zeros(4,dtype=int))
        np.testing.assert_allclose(fitted.predict(C,p),p,atol=1e-12)


def test_support_has_exact_budget_and_donor_validation_uses_later_runs():
    frame=pd.DataFrame({'subject':1001,'session':np.repeat([1,2],144),
        'run':np.tile(np.repeat(np.arange(6),24),2),'onset':np.tile(np.arange(144),2),
        'trial_id':[f't{i}' for i in range(288)]})
    support,test=support_test(frame,1001,32)
    assert len(support)==32 and len(test)==144 and set(test.session)=={2}
    support,test=support_test(frame,1001,144,True)
    assert len(support)==96 and test.run.min()>support.run.max() and set(test.session)=={1}
