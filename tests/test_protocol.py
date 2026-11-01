"""Synthetic fixtures here test software invariants, never research performance."""
import numpy as np
import pandas as pd
import pytest
from neuroedge.data import event_label, CHANNELS_8, CHANNELS_16, select_reference
from neuroedge.splits import make_splits, assert_disjoint
from neuroedge.models import TrainOnlyScaler, EEGNet
from neuroedge.deployment import calibration_indices, export_model, ort_session, ort_probabilities
from neuroedge.experiment import probabilities
from neuroedge.report import summarize_predictions, bootstrap_subjects
import torch


def test_run_dependent_labels_and_rest():
    for run in (4, 8, 12):
        assert event_label("T1", run) == 0
        assert event_label("T2", run) == 1
        assert event_label("T0", run) is None
    with pytest.raises(ValueError):
        event_label("T1", 6)  # T1 means both fists in this different task.
    with pytest.raises(ValueError):
        event_label("unknown", 4)


def test_outer_fold_coverage_and_disjoint_validation():
    splits = make_splits(range(1, 21), 5, 20261029, 0.25)
    assert sorted(s for split in splits for s in split["test"]) == list(range(1, 21))
    for split in splits:
        assert_disjoint(split["train"], split["validation"], split["test"])
        assert len(split["train"]) == 12
        assert len(split["validation"]) == 4
        assert len(split["test"]) == 4
    with pytest.raises(ValueError):
        assert_disjoint([1, 2], [2, 3], [4])


def test_unavailable_electrodes_do_not_affect_reduced_reference():
    assert set(CHANNELS_8) <= set(CHANNELS_16)
    names = CHANNELS_16 + [f"unused_{i}" for i in range(48)]
    X = np.random.default_rng(2).normal(size=(3, 64, 64)).astype(np.float32)
    a = select_reference(X, names, 8)
    X[:, 16:] += 1e6
    X[:, 8:16] -= 1e6
    np.testing.assert_array_equal(a, select_reference(X, names, 8))
    np.testing.assert_allclose(a.mean(1), 0, atol=1e-6)


def test_scaler_does_not_refit_on_new_participants():
    train = np.random.default_rng(3).normal(size=(8, 3, 128)).astype(np.float32)
    test = train + 1000
    scaler = TrainOnlyScaler().fit(train)
    before = scaler.mean_.copy(), scaler.scale_.copy()
    transformed = scaler.transform(test)
    np.testing.assert_array_equal(before[0], scaler.mean_)
    np.testing.assert_array_equal(before[1], scaler.scale_)
    assert transformed.mean() > 900
    np.testing.assert_allclose(scaler.transform(train).mean(axis=(0, 2)), 0, atol=1e-6)


def test_calibration_uses_only_training_subjects():
    meta = pd.DataFrame({"subject": np.repeat([1, 2, 3, 4], 10)})
    indices = calibration_indices(meta, [1, 2], 64, 7)
    assert len(indices) == 20
    assert set(meta.iloc[indices].subject) == {1, 2}
    assert len(set(indices)) == len(indices)


@pytest.mark.parametrize("channels", [8, 16, 64])
def test_eegnet_architecture_and_max_norm(channels):
    torch.set_num_threads(1)
    model = EEGNet(channels, 384)
    assert sum(p.numel() for p in model.parameters()) == 1234 + 16 * channels
    assert model.spatial.groups == 8
    assert model(torch.zeros(2, 1, channels, 384)).shape == (2, 2)
    with torch.no_grad():
        model.spatial.weight.fill_(10)
        model.classifier.weight.fill_(10)
    model.constrain()
    assert torch.linalg.vector_norm(model.spatial.weight, dim=(1, 2, 3)).max() <= 1.00001
    assert torch.linalg.vector_norm(model.classifier.weight, dim=1).max() <= 0.25001


def test_onnx_export_agrees_numerically(tmp_path):
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = EEGNet(8, 384).eval()
    X = np.random.default_rng(5).normal(size=(5, 8, 384)).astype(np.float32)
    destination = tmp_path / "model.onnx"
    export_model(model, destination)
    expected = probabilities(model, X)
    observed = ort_probabilities(ort_session(destination), X)
    np.testing.assert_allclose(expected, observed, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(expected.argmax(1), observed.argmax(1))


def test_repeated_seeds_do_not_increase_participant_sample_size():
    rows = []
    for seed in [7, 17, 27]:
        for subject in [1, 2]:
            for i, label in enumerate([0, 0, 1, 1]):
                rows.append({"seed": seed, "subject": subject, "model": "eegnet", "channels": 8,
                             "trial_id": f"S{subject}_T{i}", "label": label, "prediction": label})
    per_seed, per_subject, summary = summarize_predictions(pd.DataFrame(rows))
    assert len(per_seed) == 6
    assert len(per_subject) == 2
    assert summary.iloc[0].subjects == 2
    assert summary.iloc[0].balanced_accuracy == 1
    lo, hi = bootstrap_subjects([0.5, 1.0])
    assert lo == 0.5 and hi == 1.0


def test_duplicate_trial_predictions_are_rejected():
    frame = pd.DataFrame([{"seed": 7, "subject": 1, "model": "eegnet", "channels": 8,
                           "trial_id": "same", "label": 0, "prediction": 0}] * 2)
    with pytest.raises(ValueError, match="more than once"):
        summarize_predictions(frame)


def test_completed_protocol_cannot_be_silently_overwritten(tmp_path):
    from neuroedge.config import load_config, initialize_run
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    cfg["_root"] = str(tmp_path)
    initialize_run(cfg)
    cfg["training"]["epochs"] += 1
    with pytest.raises(RuntimeError, match="DIFFERENT frozen protocol"):
        initialize_run(cfg)


def test_original_environment_is_preserved_on_resume(tmp_path, monkeypatch):
    import neuroedge.config as config
    from pathlib import Path
    import json
    cfg = config.load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    cfg["_root"] = str(tmp_path)
    monkeypatch.setattr(config, "environment", lambda: {"cpu": "first-host"})
    out = config.initialize_run(cfg)
    monkeypatch.setattr(config, "environment", lambda: {"cpu": "another-host"})
    config.initialize_run(cfg)
    assert json.loads((out / "environment.json").read_text())["cpu"] == "first-host"
    assert json.loads((out / "last_invocation_environment.json").read_text())["cpu"] == "another-host"


