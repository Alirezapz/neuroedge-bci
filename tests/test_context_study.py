"""Synthetic software fixtures only; no research performance is asserted."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from neuroedge.context_data import ARRAYS,CHANNELS,MIRROR,bnci_session,paired_epochs,prepare_external
from neuroedge.context_models import ClassicalPredictor,apply_temperature,fit_temperature,inverse_sqrt,mirror_cov
from neuroedge.context_neural import fit_neural,load_neural,make_model
from neuroedge.context_protocol import ROOT,freeze,grid,load_plan,splits
from neuroedge.frontier_data import empirical_covariances


def fixture_cohort(people=range(1,7),trials=8):
    rng=np.random.default_rng(71);people=list(people);n=len(people)*trials
    active=rng.normal(size=(n,8,16,448)).astype(np.float32)
    baseline=rng.normal(size=(n,8,16,192)).astype(np.float32)
    active-=active.mean(axis=2,keepdims=True);baseline-=baseline.mean(axis=2,keepdims=True)
    def cov(x):return empirical_covariances(x.reshape(-1,16,x.shape[-1])).reshape(n,8,16,16).astype(np.float32)
    task_cov=np.stack([cov(active),cov(active[...,:224]),cov(active[...,224:])],axis=1)
    meta=pd.DataFrame({'subject':np.repeat(people,trials),'label':np.tile(np.arange(trials)%2,len(people)),
                       'trial_id':[f'FIXTURE_{s:04d}_{i:03d}' for s in people for i in range(trials)],
                       'run':1,'session':1,'onset':4.2,'dataset':'SYNTHETIC_SOFTWARE_FIXTURE',
                       'artifact_flag':False,'artifact_flag_available':False})
    return SimpleNamespace(active=active,baseline=baseline,task_cov=task_cov,baseline_cov=cov(baseline),
                           meta=meta,names=CHANNELS,manifest={'subjects':people})


def test_context_grid_and_role_separation():
    specs=grid();assert len(specs)==63 and len({s['name'] for s in specs})==63
    assert all((s['arm']=='pre_cue')==s['context'] for s in specs)
    plan=load_plan(ROOT/'configs/context_fusion.yaml')
    held=[]
    for split in splits(plan):
        assert not set(split['train_subjects'])&set(split['validation_subjects'])
        assert not (set(split['train_subjects'])|set(split['validation_subjects']))&set(plan['selection_subjects'])
        held+=split['validation_subjects']
    assert sorted(held)==sorted(plan['source_subjects'])
    assert plan['external']['subjects']==list(range(1,10))


def test_context_method_freeze_rejects_changes(tmp_path):
    plan=load_plan(ROOT/'configs/context_fusion.yaml');plan['output_dir']=str(tmp_path/'frozen')
    freeze(plan);plan['neural']['learning_rate']*=2
    with pytest.raises(RuntimeError,match='Frozen context-study'):freeze(plan)


def test_reflection_and_batched_spd_whitening():
    assert np.array_equal(MIRROR[MIRROR],np.arange(16))
    assert CHANNELS[MIRROR[CHANNELS.index('C3')]]=='C4'
    rng=np.random.default_rng(3);A=rng.normal(size=(5,16,16));C=A@np.swapaxes(A,-1,-2)+np.eye(16)
    W=inverse_sqrt(C)
    np.testing.assert_allclose(W@C@W,np.repeat(np.eye(16)[None],5,axis=0),atol=1e-12)
    np.testing.assert_array_equal(mirror_cov(mirror_cov(C)),C)


@pytest.mark.parametrize('family',['power','fbcsp','tangent'])
@pytest.mark.parametrize('mode',['active','erd','hybrid'])
def test_context_features_reflection_and_no_test_batch_fit(family,mode):
    cohort=fixture_cohort();spec=next(s for s in grid() if s['family']==family and s['mode']==mode and s['symmetry']=='equivariant' and s['C']==.1 and s['classifier']=='logistic')
    train=np.arange(32);held=np.arange(32,48)
    model=ClassicalPredictor(spec).fit(cohort,train);p=model.predict_proba(cohort,held)
    reflected=copy.copy(cohort);reflected.task_cov=mirror_cov(cohort.task_cov);reflected.baseline_cov=mirror_cov(cohort.baseline_cov)
    flipped=model.predict_proba(reflected,held)
    np.testing.assert_allclose(flipped,p[:,::-1],atol=1e-10)
    for index in [32,47]:np.testing.assert_allclose(model.predict_proba(cohort,[index]),p[index-32:index-31],atol=1e-10)
    assert model.train_subjects==[1,2,3,4]


@pytest.mark.parametrize('context',[False,True])
def test_context_neural_reflection_law_and_finite_gradients(context):
    plan=load_plan(ROOT/'configs/context_fusion.yaml')
    spec=next(s for s in grid() if s.get('architecture')=='filterbank_variance' and s['context']==context and s['symmetry']=='equivariant' and not s['dro'])
    model=make_model(spec,plan).eval();A=torch.randn(3,6,16,448);B=torch.randn(3,6,16,192)
    output=model(A,B);mirrored=model(A[:,:,MIRROR],B[:,:,MIRROR])
    torch.testing.assert_close(output,mirrored[:,[1,0]],atol=1e-6,rtol=1e-6)
    output.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_context_serialization_source_group_weighting(tmp_path):
    cohort=fixture_cohort();plan=load_plan(ROOT/'configs/context_fusion.yaml')
    spec=copy.deepcopy(next(s for s in grid() if s.get('dro') and s['symmetry']=='equivariant'));spec['epochs']=2
    predictor=fit_neural(cohort,np.arange(32),spec,plan,tmp_path,7);expected=predictor.predict_proba(cohort,np.arange(32,48))
    restored,cp=load_neural(tmp_path/'model.pt');actual=restored.predict_proba(cohort,np.arange(32,48))
    np.testing.assert_allclose(actual,expected,atol=1e-7);assert cp['train_subjects']==[1,2,3,4]
    weights=pd.read_csv(tmp_path/'source_group_weights.csv');assert set(weights.subject)=={1,2,3,4}
    assert weights.source_group_weight.sum()==pytest.approx(1)


def test_source_temperature_keeps_decisions_and_reflection():
    y=np.asarray([0,1]*20);p=np.linspace(.001,.999,40);T=fit_temperature(p,y)
    result=apply_temperature(p,T);flipped=apply_temperature(1-p,T)
    np.testing.assert_array_equal(result.argmax(axis=1),(p>.5).astype(int))
    np.testing.assert_allclose(flipped,result[:,::-1],atol=1e-10)
    loss=lambda q:float(-np.log(q[np.arange(len(y)),y]).mean())
    assert loss(result)<=loss(apply_temperature(p,1))+1e-9


def test_pre_cue_extraction_cannot_see_task_or_future_samples():
    import mne
    rng=np.random.default_rng(88);values=rng.normal(size=(16,1600))*1e-6
    raw=mne.io.RawArray(values,mne.create_info(CHANNELS,160,'eeg'),verbose='ERROR')
    before=paired_epochs(raw,[4.2])
    changed=values.copy();changed[:,int(4.2*160):]*=1000
    other=paired_epochs(mne.io.RawArray(changed,mne.create_info(CHANNELS,160,'eeg'),verbose='ERROR'),[4.2])
    assert before['active'].shape==(1,8,16,448)
    np.testing.assert_array_equal(before['baseline'],other['baseline'])
    np.testing.assert_array_equal(before['baseline_cov'],other['baseline_cov'])
    assert not np.allclose(before['active'],other['active'])


def test_external_mapping_timing_and_eog_exclusion(monkeypatch):
    import neuroedge.context_data as data
    # A synthetic MATLAB STRUCTURE fixture, not a recorded EEG study.
    X=np.tile(np.arange(25,dtype=float),(97000,1));y=np.tile([1,2,3,4],12)
    run=SimpleNamespace(X=X,fs=250,trial=1001+np.arange(48)*2000,y=y,artifacts=np.zeros(48))
    empty=SimpleNamespace(trial=np.asarray([]));captured=[]
    monkeypatch.setattr(data,'loadmat',lambda *a,**k:{'data':np.asarray([empty]+[run]*6,dtype=object)})
    def pairs(raw,cues):
        captured.append((raw.get_data()[:,0],cues))
        n=len(cues)
        return {'active':np.zeros((n,8,16,448),dtype=np.float32),'baseline':np.zeros((n,8,16,192),dtype=np.float32),
                'task_cov':np.zeros((n,3,8,16,16),dtype=np.float32),'baseline_cov':np.zeros((n,8,16,16),dtype=np.float32)}
    monkeypatch.setattr(data,'paired_epochs',pairs)
    arrays,records=bnci_session('synthetic_structure.mat',1,'T')
    assert len(records)==144 and set(r['label'] for r in records)=={0,1}
    assert captured[0][1][0]==6.0 and records[0]['subject']==1001
    np.testing.assert_allclose(captured[0][0],[data.BNCI_CHANNELS.index(c)*1e-6 for c in CHANNELS])
    assert set(arrays)==set(ARRAYS)


def test_external_acquisition_is_blocked_before_final_source_freeze(tmp_path):
    plan=load_plan(ROOT/'configs/context_fusion.yaml');plan['_root']=str(tmp_path)
    with pytest.raises(RuntimeError,match='Freeze all final source models'):prepare_external(plan)


def test_extreme_prediction_csv_roundtrip_preserves_log_loss(tmp_path):
    from neuroedge.context_verify import _read_predictions
    from neuroedge.context_protocol import score
    right=np.asarray([.9999999999999953,1.484373528710602e-14,.12345678901234567,.9999999999274614])
    frame=pd.DataFrame({'subject':[1,1,1,1],'label':[0,1,1,0],'prob_right':right})
    p=np.column_stack([1-right,right]);path=tmp_path/'extreme_probabilities.csv';frame.to_csv(path,index=False)
    restored=_read_predictions(path)
    np.testing.assert_array_equal(restored.prob_right,right)
    actual=score(restored,np.column_stack([1-restored.prob_right,restored.prob_right]))
    expected=score(frame,p)
    for key in expected:np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=0)


def test_context_full_pipeline_and_tampering_rejection(tmp_path,monkeypatch):
    import neuroedge.context_protocol as protocol
    import neuroedge.context_study as study
    import neuroedge.context_verify as verification
    import neuroedge.context_report as presentation
    # All results of this isolated run remain in pytest's temporary directory.
    # Synthetic fixture scores are never research results or shipped run data.
    chosen=[copy.deepcopy(next(s for s in grid() if s['family']=='csp_reference')),
            copy.deepcopy(next(s for s in grid() if s.get('architecture')=='eegnet_reference')),
            copy.deepcopy(next(s for s in grid() if s['family']=='power' and s['mode']=='active' and s['symmetry']=='none')),
            copy.deepcopy(next(s for s in grid() if s.get('architecture')=='filterbank_variance' and s['context'] and s['symmetry']=='equivariant' and not s['dro']))]
    for spec in chosen:
        if spec['family']=='neural':spec['epochs']=1
    for module in [protocol,study,verification,presentation]:monkeypatch.setattr(module,'grid',lambda:copy.deepcopy(chosen))
    plan=load_plan(ROOT/'configs/context_fusion.yaml');plan['source_subjects']=[1,2,3,4];plan['selection_subjects']=[5,6]
    plan['output_dir']=str(tmp_path/'synthetic_workflow');plan['neural']['final_seeds']=[7];plan['search']['threads']=1
    plan['evaluation']['bootstrap_iterations']=20
    development=fixture_cohort();external=fixture_cohort(range(1001,1010))
    monkeypatch.setattr(study,'prepare_development',lambda p:development)
    monkeypatch.setattr(study,'load_cohort',lambda p,r:development if r=='development_data' else external)
    monkeypatch.setattr(study,'prepare_external',lambda p:external)
    monkeypatch.setattr(verification,'load_cohort',lambda p,r:development if r=='development_data' else external)
    study.screen(plan);study.select(plan);study.confirm(plan)
    result=verification.verify(plan);assert result['status']=='PASS'
    presentation.report(plan)
    out=Path(plan['output_dir']);assert (out/'RESULTS.md').exists() and (out/'results.html').exists()
    # The alternate CPU scheduler must reproduce the canonical frozen transfer
    # and LOSO aggregation from the SAME fitted models. All arrays stay here.
    import importlib.util
    import neuroedge.context_data as data
    for index,filename in enumerate(['run_context_parallel.py','run_context_cpu.py']):
        module_spec=importlib.util.spec_from_file_location('neuroedge_parallel_fixture_'+str(index),ROOT/'scripts'/filename)
        scheduler=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(scheduler)
        monkeypatch.setattr(data,'prepare_external',lambda p:external)
        monkeypatch.setattr(data,'load_cohort',lambda p,r:external)
        original_transfer=pd.read_csv(out/'external_transfer/predictions.csv')
        (out/'external_transfer').rename(out/('previous_transfer_'+str(index)))
        scheduler.transfer(plan)
        pd.testing.assert_frame_equal(original_transfer,pd.read_csv(out/'external_transfer/predictions.csv'),check_exact=False,atol=1e-14,rtol=1e-14)
        original_loso=pd.read_csv(out/'external_loso/predictions.csv')
        for filename in ['predictions.csv','summary.csv','metrics_by_subject.csv','complete.json']:
            (out/'external_loso'/filename).rename(out/'external_loso'/('previous_'+str(index)+'_'+filename))
        scheduler.aggregate(plan)
        pd.testing.assert_frame_equal(original_loso,pd.read_csv(out/'external_loso/predictions.csv'),check_exact=False,atol=1e-14,rtol=1e-14)
        assert verification.verify(plan)['status']=='PASS'
    fold=next((out/'external_loso/folds').glob('*/predictions.csv'));original=fold.read_bytes()
    fold.write_bytes(original+b'\n')
    with pytest.raises(RuntimeError,match='Fold predictions changed'):scheduler.aggregate(plan)
    fold.write_bytes(original)
    prediction=out/'external_transfer/predictions.csv'
    with prediction.open('a') as f:f.write('\n')
    with pytest.raises(AssertionError,match='Final predictions changed'):verification.verify(plan)


def test_cpu_dispatcher_terminates_children_after_failed_job(tmp_path):
    import importlib.util
    import subprocess
    import time
    module_spec=importlib.util.spec_from_file_location('neuroedge_cancellation_fixture',ROOT/'scripts/run_context_cpu.py')
    scheduler=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(scheduler)
    tasks=[('fails',scheduler.call,(['-c','import time; time.sleep(.05); raise SystemExit(7)'],tmp_path/'failure.log')),
           ('running',scheduler.call,(['-c','import time; time.sleep(30)'],tmp_path/'running.log')),
           ('queued',scheduler.call,(['-c','import time; time.sleep(30)'],tmp_path/'queued.log'))]
    start=time.monotonic()
    with pytest.raises(subprocess.CalledProcessError):scheduler.dispatch(tasks,2)
    assert time.monotonic()-start<12
    assert not scheduler.ACTIVE and scheduler.STOP.is_set()
