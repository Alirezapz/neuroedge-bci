"""Train/validation diagnostics; never consults an outer-test score.

This is diagnostic development evidence, not a new test-set evaluation.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from .config import load_config, output_path, json_write, sha256
from .data import load_prepared, select_reference
from .models import load_checkpoint
from .experiment import probabilities


def metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p.argmax(1))),
            "balanced_accuracy": float(balanced_accuracy_score(y, p.argmax(1))),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "predicted_right_fraction": float((p.argmax(1) == 1).mean())}


@torch.no_grad()
def recalibrate_bn_training_only(model, train_X, batch_size=64):
    """Independent copy; deterministic layer-wise training-only moments.

    Dropout stays disabled. Earlier BN layers are calibrated before collecting
    inputs for later layers. Evaluation never uses validation/test batch moments.
    """
    calibrated = copy.deepcopy(model).eval()
    for layer in [m for m in calibrated.modules() if isinstance(m, nn.BatchNorm2d)]:
        total = torch.zeros(layer.num_features, dtype=torch.float64)
        squares = torch.zeros_like(total)
        count = 0

        def collect(module, inputs):
            nonlocal total, squares, count
            value = inputs[0].detach().double()
            total += value.sum(dim=(0, 2, 3)).cpu()
            squares += (value * value).sum(dim=(0, 2, 3)).cpu()
            count += value.shape[0] * value.shape[2] * value.shape[3]

        hook = layer.register_forward_pre_hook(collect)
        for start in range(0, len(train_X), batch_size):
            calibrated(torch.from_numpy(np.ascontiguousarray(train_X[start:start+batch_size, None], dtype=np.float32)))
        hook.remove()
        mean = total / count
        variance = ((squares - total * mean) / max(1, count-1)).clamp(min=1e-8)
        layer.running_mean.copy_(mean.to(layer.running_mean.dtype))
        layer.running_var.copy_(variance.to(layer.running_var.dtype))
    return calibrated


def diagnose(cfg, destination=None, channels=16, seed=7):
    out = output_path(cfg)
    target = Path(destination) if destination else out / "learning_diagnostics"
    target.mkdir(parents=True, exist_ok=True)
    X, meta, names = load_prepared(cfg)
    selected = select_reference(X, names, channels)
    splits = json.loads((out / "splits.json").read_text())
    torch.set_num_threads(cfg["training"]["threads"])
    rows, sources = [], []
    for split in splits:
        directory = out / "experiments" / f"fold_{split['fold']}" / f"channels_{channels}"
        checkpoint = directory / f"eegnet_seed_{seed}" / "model.pt"
        model, scaler, cp = load_checkpoint(checkpoint)
        before = sha256(checkpoint)
        train_idx = meta.subject.isin(split["train"]).to_numpy()
        calibrated = recalibrate_bn_training_only(model, scaler.transform(selected[train_idx]))
        for role in ["train", "validation"]:
            idx = meta.subject.isin(split[role]).to_numpy()
            labels = meta.loc[idx, "label"].to_numpy(np.int64)
            data = scaler.transform(selected[idx])
            for name, net in [("eegnet_saved", model), ("eegnet_training_bn_recalibration", calibrated)]:
                p = probabilities(net, data)
                rows.append({"fold": split["fold"], "role": role, "model": name,
                             "channels": channels, "n_trials": int(idx.sum()), **metrics(labels, p)})
            csp = joblib.load(directory / "csp_lda_seed_-1" / "model.joblib")
            p = csp.predict_proba(selected[idx].astype(np.float64))
            rows.append({"fold": split["fold"], "role": role, "model": "csp_lda_saved",
                         "channels": channels, "n_trials": int(idx.sum()), **metrics(labels, p)})
        if before != sha256(checkpoint):
            raise AssertionError("Original checkpoint was modified.")
        sources.append({"fold": split["fold"], "checkpoint_sha256": before,
                        "train_subjects": split["train"], "validation_subjects": split["validation"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "train_validation_metrics.csv", index=False)
    json_write(target / "provenance.json", {"sources": sources, "outer_test_scored": False,
               "original_checkpoints_unchanged": True,
               "note": "Recalibration is a diagnostic candidate fitted on training inputs only; no guarantee of improvement."})
    print(frame.groupby(["model", "role"])[["balanced_accuracy", "log_loss", "predicted_right_fraction"]].mean().to_string(), flush=True)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    diagnose(load_config(args.config), channels=args.channels, seed=args.seed)


if __name__ == "__main__":
    main()
