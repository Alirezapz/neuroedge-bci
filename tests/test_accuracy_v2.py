"""Synthetic fixtures check implementation invariants, never research accuracy."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from neuroedge.accuracy_v2 import load_plan, validate_plan, UnitTraceCovariances, classical
from neuroedge.learning_diagnostics import recalibrate_bn_training_only
from neuroedge.models import EEGNet
from neuroedge.experiment import probabilities


def test_fresh_confirmation_subjects_are_disjoint():
    plan = load_plan(Path(__file__).resolve().parents[1] / "configs/accuracy_v2.yaml")
    assert not set(plan["development_subjects"]) & set(plan["confirmation_subjects"])
    altered = copy.deepcopy(plan)
    altered["confirmation_subjects"][0] = 1
    with pytest.raises(ValueError, match="leakage"):
        validate_plan(altered)


def test_training_only_bn_calibration_preserves_original_and_weights():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    model = EEGNet(8, 384).eval()
    original = copy.deepcopy(model.state_dict())
    X = np.random.default_rng(1).normal(2, 1, (12, 8, 384)).astype(np.float32)
    calibrated = recalibrate_bn_training_only(model, X, batch_size=4)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    for (name, before), (_, after) in zip(model.named_parameters(), calibrated.named_parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    with torch.no_grad():
        features = model.temporal[1](model.temporal[0](torch.from_numpy(X[:,None])))
    torch.testing.assert_close(calibrated.temporal[2].running_mean, features.mean((0,2,3)), atol=1e-6, rtol=1e-5)
    all_p = probabilities(calibrated, X)
    one_p = np.concatenate([probabilities(calibrated, x[None]) for x in X])
    np.testing.assert_allclose(all_p, one_p, atol=1e-6, rtol=1e-5)


def test_covariance_normalization_is_per_trial_and_positive_definite():
    X = np.random.default_rng(6).normal(size=(12, 8, 384))
    cov = UnitTraceCovariances().fit(X).transform(X)
    np.testing.assert_allclose(np.trace(cov, axis1=1, axis2=2), 1)
    assert np.linalg.eigvalsh(cov).min() > 0
    separate = np.concatenate([UnitTraceCovariances().transform(x[None]) for x in X])
    np.testing.assert_allclose(cov, separate, atol=1e-12)


def test_tangent_pipeline_has_no_test_batch_reference(tmp_path):
    import joblib
    X = np.random.default_rng(2).normal(size=(24, 8, 384))
    y = np.tile([0,1], 12)
    plan = load_plan(Path(__file__).resolve().parents[1] / "configs/accuracy_v2.yaml")
    model = classical("tangent_logistic", plan).fit(X, y)
    assert model.named_steps["tangent"].tsupdate is False
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    batch = restored.predict_proba(X[:4])
    single = np.concatenate([restored.predict_proba(x[None]) for x in X[:4]])
    np.testing.assert_allclose(batch, single, atol=1e-7)


def test_source_size_control_cannot_be_prespecified_after_test_started(tmp_path):
    from neuroedge.accuracy_size_control import freeze_control
    plan=load_plan(Path(__file__).resolve().parents[1]/"configs/accuracy_expanded.yaml")
    plan["_root"]=str(tmp_path)
    directory=tmp_path/plan["output_dir"]
    directory.mkdir(parents=True)
    (directory/"confirmation_started.json").write_text("{}")
    with pytest.raises(RuntimeError,match="retrospectively"):
        freeze_control(plan)
    assert not (directory/"source_size_control_plan.json").exists()

