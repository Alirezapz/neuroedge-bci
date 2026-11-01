"""Synthetic fixtures verify software invariants, never EEG research accuracy."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from neuroedge.frontier import discovery_splits, _ensemble_predictions, _logits
from neuroedge.frontier_alignment import inverse_sqrt, RestCalibratedCohort
from neuroedge.frontier_data import empirical_covariances, referenced_covariances, shrink_covariances
from neuroedge.frontier_features import candidates, CovarianceModel, CovarianceFeatures
from neuroedge.frontier_neural import make_model, segment_recombine, trial_normalize, fit_neural, load_neural
from neuroedge.frontier_protocol import load_plan, freeze, score


ROOT=Path(__file__).resolve().parents[1]


def test_candidate_grid_unique_and_both_arms_explicit():
    specs=candidates()
    assert len(specs)==224
    assert len({s['name'] for s in specs})==224
    assert sum(s.get('arm')=='rest_calibrated' for s in specs)==15
    assert {'csp','fbcsp','tangent','fbtangent','mdm','power','neural','raw_csp'}=={s['family'] for s in specs}


def test_discovery_groups_never_contain_validation_or_fresh_confirmation():
    plan=load_plan(ROOT/'configs/frontier.yaml')
    coverage=[]
    for fold in discovery_splits(plan):
        assert not set(fold['train_subjects'])&set(fold['validation_subjects'])
        assert not (set(fold['train_subjects'])|set(fold['validation_subjects'])) & set(plan['validation_subjects']+plan['confirmation_subjects'])
        coverage.extend(fold['validation_subjects'])
    assert sorted(coverage)==plan['discovery_subjects']


def test_old_test_participants_cannot_be_relabelled_as_fresh(tmp_path):
    import yaml
    plan=load_plan(ROOT/'configs/frontier.yaml')
    plan['confirmation_subjects']=[80]
    plan['validation_subjects']=list(range(61,80))
    path=tmp_path/'plan.yaml';path.write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError,match='Previously inspected'):
        load_plan(path)


def test_frozen_plan_rejects_method_change(tmp_path):
    plan=load_plan(ROOT/'configs/frontier.yaml');plan['_root']=str(tmp_path)
    freeze(plan)
    plan['neural']['learning_rate']*=2
    with pytest.raises(RuntimeError,match='Frozen frontier'):
        freeze(plan)


def test_covariance_reference_uses_only_available_sensors():
    X=np.random.default_rng(3).normal(size=(12,64,80))
    selected=[2,4,9,12]
    C=referenced_covariances(empirical_covariances(X),selected)
    observed=X[:,selected];observed-=observed.mean(axis=1,keepdims=True)
    np.testing.assert_allclose(C,empirical_covariances(observed),atol=1e-12)
    X[:,[i for i in range(64) if i not in selected]]*=10000
    np.testing.assert_allclose(C,referenced_covariances(empirical_covariances(X),selected),atol=1e-12)


def test_spd_shrinkage_is_trial_independent():
    X=np.random.default_rng(2).normal(size=(9,8,64))
    X-=X.mean(axis=1,keepdims=True)
    C=empirical_covariances(X)
    S=shrink_covariances(C)
    assert np.linalg.eigvalsh(S).min()>0
    np.testing.assert_allclose(S,np.concatenate([shrink_covariances(c[None]) for c in C]),atol=1e-12)
    A=inverse_sqrt(S[0])
    np.testing.assert_allclose(A@S[0]@A.T,np.eye(8),atol=1e-10)


@pytest.mark.parametrize('family',['csp','fbcsp','tangent','fbtangent','mdm','power'])
def test_classical_inference_does_not_update_from_other_test_examples(family):
    rng=np.random.default_rng(9)
    X=rng.normal(size=(30,3,8,64))
    C=np.stack([empirical_covariances(X[:,i]) for i in range(3)],axis=1)
    y=np.tile([0,1],15)
    spec={'family':family,'components':4,'reg':.1,'normalize':True,'metric':'logeuclid',
          'classifier':'logistic','C':.1}
    model=CovarianceModel(spec).fit(C[:24],y[:24])
    p=model.predict_proba(C[24:])
    single=np.concatenate([model.predict_proba(c[None]) for c in C[24:]])
    np.testing.assert_allclose(p,single,atol=1e-10)
    if family in {'tangent','fbtangent'}:
        assert all(not mapper.tsupdate for mapper in model.features.transformers_)


@pytest.mark.parametrize('architecture',['eegnet','shallow','attention'])
def test_neural_shapes_and_finite_gradients(architecture):
    torch.set_num_threads(2)
    spec={'name':'software_fixture','architecture':architecture,'channels':16,'dropout':.5}
    model=make_model(spec,448)
    X=torch.randn(3,1,16,448)
    logits=model(X)
    assert logits.shape==(3,2)
    logits.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_segment_augmentation_never_uses_other_class_or_input_changes():
    X=np.stack([np.full((3,28),i,dtype=np.float32) for i in range(8)])
    y=np.array([0]*4+[1]*4)
    original=X.copy()
    mixed=segment_recombine(X,y,np.random.default_rng(6))
    np.testing.assert_array_equal(X,original)
    assert set(np.unique(mixed[y==0])).issubset(set(range(4)))
    assert set(np.unique(mixed[y==1])).issubset(set(range(4,8)))


def test_trial_normalization_is_not_evaluation_batch_normalization():
    X=np.random.default_rng(7).normal(size=(6,16,448)).astype('float32')
    spec={'architecture':'shallow'}
    all_X=trial_normalize(X,spec)
    separate=np.concatenate([trial_normalize(x[None],spec) for x in X])
    np.testing.assert_allclose(all_X,separate,atol=1e-7)


def test_neural_serialization_preserves_probabilities_and_source_scaler(tmp_path):
    torch.set_num_threads(2)
    X=np.random.default_rng(11).normal(size=(12,16,384)).astype('float32')
    y=np.tile([0,1],6);subjects=np.repeat([1,2,3],4)
    plan=load_plan(ROOT/'configs/frontier.yaml')
    spec={'name':'software_fixture','family':'neural','architecture':'shallow','channels':16,
          'dropout':.5,'augmentation':'none'}
    trained=fit_neural(X,y,subjects,spec,plan,tmp_path,epochs=1)
    restored,cp=load_neural(tmp_path/'model.pt')
    np.testing.assert_allclose(trained.predict_proba(X),restored.predict_proba(X),atol=1e-7)
    assert cp['train_subjects']==[1,2,3]
    np.testing.assert_allclose(trained.scaler.mean_,trial_normalize(X,spec).mean(axis=(0,2),keepdims=True),atol=1e-7)


def test_ensemble_rules_are_explicit_and_probabilities_valid():
    P=np.array([[.1,.8],[.9,.2]])
    np.testing.assert_allclose(_ensemble_predictions({'kind':'average'},P),[[.55,.45],[.45,.55]])
    np.testing.assert_allclose(_ensemble_predictions({'kind':'weighted','weights':[1,0]},P)[:,1],P[:,0])
    assert np.isfinite(_logits(np.array([0.,1.]))).all()


def test_rest_alignment_depends_only_on_separate_reference_and_own_participant():
    from types import SimpleNamespace
    import pandas as pd
    from neuroedge.data import CHANNELS_16
    rng=np.random.default_rng(17)
    rest=np.stack([empirical_covariances(rng.normal(size=(8,16,80))) for _ in range(2)])
    task=empirical_covariances(rng.normal(size=(4,16,80)))[:,None]
    base=SimpleNamespace(manifest={'subjects':[1,2]},meta=pd.DataFrame({'subject':[1,1,2,2]}),names=CHANNELS_16,
                         covariance_view=lambda spec,rows:task[rows])
    aligned=RestCalibratedCohort(base,rest)
    spec={'channels':16,'bands':[1],'arm':'rest_calibrated'}
    all_C=aligned.covariance_view(spec,np.arange(4))
    separate=np.concatenate([aligned.covariance_view(spec,np.array([i])) for i in range(4)])
    np.testing.assert_allclose(all_C,separate,atol=1e-12)
    changed=rest.copy();changed[1]*=1000
    altered=RestCalibratedCohort(base,changed)
    np.testing.assert_allclose(all_C[:2],altered.covariance_view(spec,np.array([0,1])),atol=1e-12)
    zero={**spec,'arm':'zero_calibration'}
    np.testing.assert_array_equal(aligned.covariance_view(zero,np.arange(4)),task)


def test_binary_probability_roundoff_survives_csv_metric_recomputation():
    import pandas as pd
    from neuroedge.frontier_protocol import predictions,score
    meta=pd.DataFrame({'trial_id':['a','b'],'subject':[1,1],'run':[4,4],'onset':[1.,5.], 'label':[0,1]})
    p=np.array([[.7,.30000004],[.40000004,.6]],dtype=np.float32)
    frame=predictions(meta,p)
    before=score(meta,p)
    after=score(frame,np.column_stack([1-frame.prob_right,frame.prob_right]))
    for key in before:
        np.testing.assert_allclose(before[key],after[key],atol=1e-14,rtol=0)
