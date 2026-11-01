"""Synthetic fixtures check software invariants only, never research accuracy."""
import numpy as np
import pandas as pd
import pytest
from neuroedge.covariance_models import candidates,CovarianceModel,BatchTransport,normalize,FeatureMap
from neuroedge.covariance_study import select_specs


def cov(seed,n):
    r=np.random.default_rng(seed);X=r.normal(size=(n,3,8,16,24));return X@np.swapaxes(X,-1,-2)/24


@pytest.mark.parametrize('family',sorted({s['family'] for s in candidates()}))
def test_strict_predictor_does_not_use_other_test_trials(family):
    spec=next(s for s in candidates() if s['family']==family);A=cov(82,18);B=cov(83,5);y=np.arange(18)%2
    model=CovarianceModel(spec,'strict').fit(A,y,np.linspace(.3,.7,18));p=model.predict(B,np.linspace(.2,.8,5));q=model.predict(B[:1],np.array([.2]))
    np.testing.assert_allclose(p[:1],q,atol=1e-10,rtol=1e-10)


def test_unlabelled_transport_matches_support_covariance_center():
    A=cov(84,18);B=cov(85,11);T=BatchTransport().fit(A)
    np.testing.assert_allclose(T.transform(B).mean(axis=0),normalize(A,.01).mean(axis=0),atol=1e-10,rtol=1e-10)


def test_parameter_selection_excludes_all_outer_person_scores():
    specs=candidates()[:2];rows=[]
    for person in [1001,1002,1003]:
        for i,s in enumerate(specs):rows.append({'subject':person,'candidate':s['name'],'family':s['family'],'balanced_accuracy':.6+i*.1,'log_loss':.7})
    frame=pd.DataFrame(rows);before=select_specs(frame,[1001]);frame.loc[frame.subject==1001,'balanced_accuracy']=99
    assert before==select_specs(frame,[1001])


def test_candidate_names_are_unique_and_sensor_independent():
    grid=candidates();assert len(grid)==71 and len({s['name'] for s in grid})==len(grid)
    spec=next(s for s in grid if s['family']=='tangent_svm');A=cov(86,18);B=cov(87,4)
    Q=np.random.default_rng(88).normal(size=(22,16));Q[:16]=np.eye(16)
    # More channels require actual extra covariance dimensions; no hidden access.
    A22=Q@A@Q.T;B22=Q@B@Q.T
    p=CovarianceModel(spec,'strict').fit(A22,np.arange(18)%2,np.full(18,.5)).predict(B22,np.full(4,.5))
    assert p.shape==(4,) and np.isfinite(p).all()


def test_all_power_means_preserve_a_constant_spd_input_below_identity():
    from neuroedge.session_models import symmetric_function
    C=np.broadcast_to(np.diag(np.linspace(.1,2.,16)),(8,3,8,16,16)).copy()
    spec=next(s for s in candidates() if s['family']=='power_means')
    fitted=FeatureMap(spec).fit(C,np.arange(8)%2)
    expected=symmetric_function(normalize(C,spec['shrink'])[0,0,2:],np.log)
    assert np.linalg.eigvalsh(expected).min()<0
    for label in [0,1]:
        for prototype in fitted.prototypes[label]:np.testing.assert_allclose(prototype,expected,atol=1e-12,rtol=0)
