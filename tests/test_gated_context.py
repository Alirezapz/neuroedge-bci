"""Synthetic software checks for the matched quality-gate hypothesis."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

from neuroedge.context_data import MIRROR
from neuroedge.gated_neural import GatedContext,fit_models,predict_saved
from neuroedge.gated_study import load_plan,specs,freeze

ROOT=Path(__file__).resolve().parents[1]


def test_gated_parameter_match_reflection_and_quality_dependence():
    plan=load_plan(ROOT/'configs/gated_context.yaml');A=torch.randn(4,6,16,448);B=torch.randn(4,6,16,192)
    models=[GatedContext(s,plan['_settings']).eval() for s in specs(plan)]
    assert sum(p.numel() for p in models[0].parameters())==sum(p.numel() for p in models[1].parameters())
    for model in models:
        output=model(A,B);other=model(A[:,:,MIRROR],B[:,:,MIRROR])
        torch.testing.assert_close(output,other[:,[1,0]],atol=1e-6,rtol=1e-6)
        output.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    _,fixed=models[0]._features(A,B);_,dynamic=models[1]._features(A,B)
    torch.testing.assert_close(fixed,fixed[:1].expand_as(fixed),atol=0,rtol=0)
    assert float(dynamic.std(dim=0).mean().detach())>0


def test_gated_source_roundtrip_and_batch_independence(tmp_path):
    plan=load_plan(ROOT/'configs/gated_context.yaml');spec=specs(plan)[1];spec['epochs']=2
    rng=np.random.default_rng(51);A=rng.normal(size=(24,8,16,448)).astype(np.float32);B=rng.normal(size=(24,8,16,192)).astype(np.float32)
    meta=pd.DataFrame({'subject':np.repeat([1,2,3],8),'label':np.tile([0,1],12),'trial_id':[f'SYNTHETIC_GATE_{i}' for i in range(24)]})
    cohort=SimpleNamespace(active=A,baseline=B,meta=meta)
    info=fit_models(cohort,np.arange(16),spec,plan['_settings'],tmp_path,[7]);assert info['train_subjects']==[1,2]
    p=predict_saved(cohort,np.arange(16,24),spec,tmp_path)
    np.testing.assert_allclose(predict_saved(cohort,np.asarray([17]),spec,tmp_path),p[1:2],atol=2e-5,rtol=2e-5)
    p2,gates=predict_saved(cohort,np.arange(16,24),spec,tmp_path,True);np.testing.assert_allclose(p2,p,atol=1e-7)
    assert gates.shape==(8,48) and np.isfinite(gates).all()


def test_gated_protocol_rejects_changed_quality_network(tmp_path):
    plan=load_plan(ROOT/'configs/gated_context.yaml');plan['output_dir']=str(tmp_path/'frozen')
    plan['_parent']['output_dir']=str(tmp_path/'parent_frozen')
    plan['_parent']['external']['raw_dir']=str(tmp_path/'never_downloaded')
    freeze(plan);plan['gate_hidden_units']=16
    with pytest.raises(RuntimeError,match='Frozen gated methods'):freeze(plan)


def test_gated_end_to_end_and_independent_reinference(tmp_path,monkeypatch):
    import copy,json
    import neuroedge.gated_study as study
    import neuroedge.gated_verify as verification
    from neuroedge.gated_report import report
    from neuroedge.context_study import summarize
    from neuroedge.context_protocol import predictions
    # Small, isolated synthetic arrays. No fixture output enters research runs.
    def cohort(people):
        rng=np.random.default_rng(13);people=list(people);n=len(people)*4
        meta=pd.DataFrame({'subject':np.repeat(people,4),'label':np.tile([0,1,0,1],len(people)),
            'trial_id':[f'SYNTHETIC_GATE_{s}_{i}' for s in people for i in range(4)],'run':1,
            'session':np.tile([1,1,2,2],len(people)),'onset':4.2})
        return SimpleNamespace(active=rng.normal(size=(n,8,16,448)).astype(np.float32),
            baseline=rng.normal(size=(n,8,16,192)).astype(np.float32),meta=meta)
    development=cohort(range(1,7));external=cohort(range(1001,1010))
    plan=load_plan(ROOT/'configs/gated_context.yaml');plan['output_dir']=str(tmp_path/'gated');plan['epochs']=1;plan['seeds']=[7]
    plan['bootstrap_iterations']=20;parent=plan['_parent'];parent['output_dir']=str(tmp_path/'parent')
    parent['source_subjects']=[1,2,3,4];parent['selection_subjects']=[5,6];parent['search']['threads']=1
    parent['external']['raw_dir']=str(tmp_path/'never_downloaded')
    for module in [study,verification]:monkeypatch.setattr(module,'load_cohort',lambda p,r:development if r=='development_data' else external)
    for variant in plan['variants']:study.source(plan,variant)
    study.freeze_selection(plan);study.external(plan)
    assert verification.verify(plan)['status']=='PASS'
    # Regression: allowed reinference roundoff must not be compared to archived
    # log loss at a stricter tolerance than the verified CSV probabilities.
    original_predict=verification.predict_saved
    def slight_reinference_roundoff(*args,**kwargs):
        p=np.asarray(original_predict(*args,**kwargs),dtype=np.float64).copy()
        p[:,1]=np.clip(p[:,1]+np.where(p[:,1]>.5,1.,-1.)*1e-9,1e-12,1-1e-12)
        p[:,0]=1-p[:,1]
        return p
    with monkeypatch.context() as context:
        context.setattr(verification,'predict_saved',slight_reinference_roundoff)
        assert verification.verify(plan)['status']=='PASS'
    # Parent comparison tables are mocked software fixtures, not EEG findings.
    parent_out=Path(parent['output_dir']);(parent_out/'verification.json').write_text(json.dumps({'status':'PASS','scope':'synthetic fixture only'}))
    frame=[]
    for model in ['csp_reference','eegnet_reference','selected_zero_calibration','selected_pre_cue']:
        f=predictions(external.meta,np.full((len(external.meta),2),.5));f['model']=model;f['arm']='pre_cue' if model=='selected_pre_cue' else 'zero_calibration';frame.append(f)
    bootstrap={'evaluation':{'bootstrap_seed':43,'bootstrap_iterations':20}}
    for role in ['external_transfer','external_loso']:summarize(bootstrap,pd.concat(frame,ignore_index=True),parent_out/role,{'synthetic_fixture':True})
    report(plan);assert (Path(plan['output_dir'])/'results.html').exists()
    target=Path(plan['output_dir'])/'external_transfer/predictions.csv'
    with target.open('a') as f:f.write('\n')
    with pytest.raises(AssertionError,match='Gated external predictions changed'):verification.verify(plan)
