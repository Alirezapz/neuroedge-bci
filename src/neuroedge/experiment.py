"""Matched subject-group folds, actual training, OOF predictions, resumable artifacts."""
from __future__ import annotations
import copy
import json
import random
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from .config import digest, initialize_run, json_write, output_path, protocol, sha256
from .data import load_prepared, select_reference, channel_indices
from .models import EEGNet, TrainOnlyScaler
from .splits import make_splits, assert_disjoint


def mask(meta, subjects):
    return meta.subject.isin(subjects).to_numpy()


@torch.no_grad()
def probabilities(model, X, batch_size=128):
    model.eval()
    values = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size, None]))
        values.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(values)


def fit_eegnet(X, y, train_idx, val_idx, cfg, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    scaler = TrainOnlyScaler().fit(X[train_idx])
    train_x = scaler.transform(X[train_idx])
    val_x = scaler.transform(X[val_idx])
    model = EEGNet(channels=X.shape[1], samples=X.shape[2],
                   dropout=cfg["training"]["dropout"], kernel_length=cfg["training"]["kernel_length"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    tx = torch.from_numpy(train_x[:, None])
    ty = torch.from_numpy(y[train_idx])
    best_loss, best_epoch, best_state, waiting = float("inf"), 0, None, 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        order = rng.permutation(len(tx))
        losses = []
        for start in range(0, len(tx), cfg["training"]["batch_size"]):
            idx = order[start:start + cfg["training"]["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tx[idx]), ty[idx])
            loss.backward()
            optimizer.step()
            model.constrain()
            losses.append((len(idx), float(loss.detach())))
        p = probabilities(model, val_x)
        val_loss = float(log_loss(y[val_idx], p, labels=[0, 1]))
        history.append({"epoch": epoch, "train_loss": sum(n * loss for n, loss in losses) / len(tx),
                        "validation_loss": val_loss,
                        "validation_balanced_accuracy": float(balanced_accuracy_score(y[val_idx], p.argmax(1)))})
        if not np.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss; no test evaluation performed.")
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, best_state, waiting = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            waiting += 1
        if waiting >= cfg["training"]["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, history, best_epoch


def save_predictions(directory, meta, p, fold, channels, model, seed):
    frame = meta[["trial_id", "subject", "run", "onset", "label"]].copy()
    frame["prediction"] = np.argmax(p, axis=1)
    frame["prob_right"] = p[:, 1]
    frame["fold"], frame["channels"], frame["model"], frame["seed"] = fold, channels, model, seed
    frame.to_csv(directory / "predictions.csv", index=False)
    return frame


def run(cfg):
    import mne
    from mne.decoding import CSP
    mne.set_log_level("ERROR")
    out = initialize_run(cfg)
    X, meta, names = load_prepared(cfg)
    if sorted(meta.subject.unique().tolist()) != sorted(cfg["dataset"]["subjects"]):
        raise RuntimeError("Prepared cohort differs from the frozen protocol.")
    ev = cfg["evaluation"]
    splits = make_splits(meta.subject, ev["folds"], ev["split_seed"], ev["validation_fraction"])
    json_write(out / "splits.json", splits)
    source_paths = [Path(__file__), Path(__file__).with_name("data.py"),
                    Path(__file__).with_name("models.py"), Path(__file__).with_name("splits.py")]
    signature = digest({"protocol": protocol(cfg), "data": sha256(out / "data_manifest.json"),
                        "sources": {p.name: sha256(p) for p in source_paths}})
    ledger = out / "training_identity.json"
    if ledger.exists() and json.loads(ledger.read_text())["sha256"] != signature:
        raise RuntimeError("Training code/data/protocol changed. Use a new output directory, do not mix results.")
    json_write(ledger, {"sha256": signature, "sources": {p.name: sha256(p) for p in source_paths}})
    torch.set_num_threads(cfg["training"]["threads"])
    y = meta.label.to_numpy(np.int64)
    frames = []
    planned = len(splits) * len(ev["channels"]) * (2 + len(ev["seeds"]))
    completed = 0
    with threadpool_limits(limits=cfg["training"]["threads"]):
        for split in splits:
            assert_disjoint(split["train"], split["validation"], split["test"])
            train_idx, val_idx, test_idx = [np.flatnonzero(mask(meta, split[k])) for k in ("train", "validation", "test")]
            for count in ev["channels"]:
                selected = select_reference(X, names, count)
                for model_name, seeds in [("dummy", [-1]), ("csp_lda", [-1]), ("eegnet", ev["seeds"])]:
                    for seed in seeds:
                        directory = out / "experiments" / f"fold_{split['fold']}" / f"channels_{count}" / f"{model_name}_seed_{seed}"
                        directory.mkdir(parents=True, exist_ok=True)
                        complete = directory / "complete.json"
                        if complete.exists():
                            status = json.loads(complete.read_text())
                            if status["training_identity"] != signature:
                                raise RuntimeError("Completed experiment has a different identity.")
                            if sha256(directory / "predictions.csv") != status["predictions_sha256"]:
                                raise RuntimeError("Completed prediction file changed.")
                            frame = pd.read_csv(directory / "predictions.csv")
                            if set(frame.trial_id) != set(meta.iloc[test_idx].trial_id):
                                raise RuntimeError("Saved test identities do not match the outer split.")
                            frames.append(frame)
                            completed += 1
                            print(f"Resume {completed}/{planned}: {directory.relative_to(out)}", flush=True)
                            continue
                        started = time.perf_counter()
                        print(f"Train {completed + 1}/{planned}: fold={split['fold']} channels={count} model={model_name} seed={seed}", flush=True)
                        extra = {}
                        if model_name == "dummy":
                            estimator = DummyClassifier(strategy="prior").fit(np.zeros((len(train_idx), 1)), y[train_idx])
                            p = estimator.predict_proba(np.zeros((len(test_idx), 1)))
                            joblib.dump(estimator, directory / "model.joblib")
                        elif model_name == "csp_lda":
                            estimator = Pipeline([
                                ("csp", CSP(n_components=4, reg=0.1, log=True, norm_trace=False)),
                                ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
                            estimator.fit(selected[train_idx].astype(np.float64), y[train_idx])
                            p = estimator.predict_proba(selected[test_idx].astype(np.float64))
                            joblib.dump(estimator, directory / "model.joblib")
                        else:
                            model, scaler, history, best_epoch = fit_eegnet(selected, y, train_idx, val_idx, cfg, seed)
                            p = probabilities(model, scaler.transform(selected[test_idx]))
                            pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
                            torch.save({"model_spec": model.spec, "state_dict": model.state_dict(),
                                        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                                        "train_subjects": split["train"], "validation_subjects": split["validation"],
                                        "test_subjects": split["test"], "fold": split["fold"], "seed": seed,
                                        "channel_names": [names[i] for i in channel_indices(names, count)],
                                        "training_identity": signature}, directory / "model.pt")
                            extra = {"best_epoch": best_epoch, "epochs_run": len(history),
                                     "parameters": sum(p.numel() for p in model.parameters())}
                        frames.append(save_predictions(directory, meta.iloc[test_idx], p, split["fold"], count, model_name, seed))
                        json_write(complete, {"training_identity": signature, "elapsed_seconds": time.perf_counter() - started,
                                              "train_subjects": split["train"], "validation_subjects": split["validation"],
                                              "test_subjects": split["test"], "n_train": len(train_idx), "n_test": len(test_idx),
                                              "predictions_sha256": sha256(directory / "predictions.csv"), **extra})
                        completed += 1
                        json_write(out / "progress.json", {"completed": completed, "planned": planned, "status": "running"})
                        print(f"Done in {time.perf_counter() - started:.1f}s", flush=True)
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_csv(out / "predictions.csv", index=False)
    json_write(out / "progress.json", {"completed": completed, "planned": planned, "status": "complete"})
    return all_predictions

