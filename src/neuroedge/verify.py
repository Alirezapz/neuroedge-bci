"""Independent checks on real saved artifacts, not just software unit tests."""
import json
import numpy as np
import pandas as pd
import joblib
import torch
from .config import output_path, project_path, json_write, sha256
from .data import load_prepared, select_reference
from .models import TrainOnlyScaler, load_checkpoint
from .splits import assert_disjoint
from .experiment import probabilities


def verify(cfg):
    out = output_path(cfg)
    torch.set_num_threads(cfg["training"]["threads"])
    X, meta, names = load_prepared(cfg)
    audit = json.loads((out / "data_audit.json").read_text())
    for record in audit["records"]:
        raw = project_path(cfg, cfg["dataset"]["raw_dir"]) / f"S{record['subject']:03d}/S{record['subject']:03d}R{record['run']:02d}.edf"
        if sha256(raw) != record["source_sha256"]:
            raise AssertionError(f"Raw EDF changed since preparation: {raw.name}")
    identity = json.loads((out / "training_identity.json").read_text())
    for name, expected in identity["sources"].items():
        if sha256(project_path(cfg, "src/neuroedge") / name) != expected:
            raise AssertionError(f"Frozen training source changed: {name}")
    predictions = pd.read_csv(out / "predictions.csv")
    splits = json.loads((out / "splits.json").read_text())
    checked = []
    for split in splits:
        assert_disjoint(split["train"], split["validation"], split["test"])
        for count in cfg["evaluation"]["channels"]:
            selected = select_reference(X, names, count)
            train_idx = meta.subject.isin(split["train"]).to_numpy()
            test_idx = meta.subject.isin(split["test"]).to_numpy()
            expected_scaler = TrainOnlyScaler().fit(selected[train_idx])
            expected_trials = set(meta[meta.subject.isin(split["test"])].trial_id)
            for model, seeds in [("dummy", [-1]), ("csp_lda", [-1]), ("eegnet", cfg["evaluation"]["seeds"])]:
                for seed in seeds:
                    frame = predictions[(predictions.fold == split["fold"]) & (predictions.channels == count)
                                        & (predictions.model == model) & (predictions.seed == seed)]
                    if set(frame.trial_id) != expected_trials or frame.trial_id.duplicated().any():
                        raise AssertionError("Outer-test trial coverage mismatch.")
                    truth = meta.set_index("trial_id").loc[frame.trial_id, "label"].to_numpy()
                    np.testing.assert_array_equal(frame.label.to_numpy(), truth)
                    directory = out / "experiments" / f"fold_{split['fold']}" / f"channels_{count}" / f"{model}_seed_{seed}"
                    complete = json.loads((directory / "complete.json").read_text())
                    if complete["training_identity"] != identity["sha256"]:
                        raise AssertionError("Training identity mismatch.")
                    if sha256(directory / "predictions.csv") != complete["predictions_sha256"]:
                        raise AssertionError("Experiment prediction hash mismatch.")
                    if model == "eegnet":
                        estimator, scaler, cp = load_checkpoint(directory / "model.pt")
                        np.testing.assert_allclose(scaler.mean_, expected_scaler.mean_, rtol=0, atol=1e-7)
                        np.testing.assert_allclose(scaler.scale_, expected_scaler.scale_, rtol=0, atol=1e-7)
                        if cp["train_subjects"] != split["train"] or cp["test_subjects"] != split["test"]:
                            raise AssertionError("Checkpoint participant provenance mismatch.")
                        recomputed = probabilities(estimator, scaler.transform(selected[test_idx]))
                    else:
                        estimator = joblib.load(directory / "model.joblib")
                        input_X = selected[test_idx].astype(np.float64) if model == "csp_lda" else np.zeros((test_idx.sum(), 1))
                        recomputed = estimator.predict_proba(input_X)
                    aligned = frame.set_index("trial_id").loc[meta[test_idx].trial_id]
                    np.testing.assert_allclose(aligned.prob_right.to_numpy(), recomputed[:, 1], rtol=1e-5, atol=1e-6)
                    np.testing.assert_array_equal(aligned.prediction.to_numpy(), recomputed.argmax(1))
                    checked.append(str(directory.relative_to(out)))
    deployment = out / "deployment"
    if (deployment / "summary.json").exists():
        for fold in range(cfg["evaluation"]["folds"]):
            for count in cfg["evaluation"]["channels"]:
                cal = json.loads((deployment / f"fold_{fold}_channels_{count}" / "calibration.json").read_text())
                observed = set(meta.set_index("trial_id").loc[cal["trial_ids"], "subject"])
                if not observed <= set(splits[fold]["train"]):
                    raise AssertionError("Quantization calibration leakage.")
    result = {"status": "PASS", "experiments_checked": len(checked),
              "checks": ["raw/prepared data checksums", "train/validation/test participant disjointness",
                         "exact outer-test trial coverage", "saved truth equals audited labels",
                         "saved scaler matches training-only recomputation", "checkpoint provenance",
                         "frozen training source hashes unchanged",
                         "every saved model reproduces its saved test probabilities and decisions",
                         "quantization calibration participant membership when deployment exists"],
              "limitations": "These checks do not prove physiological validity or absence of every possible software defect."}
    json_write(out / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result

