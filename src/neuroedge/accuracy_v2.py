"""Validation-only model development followed by a fresh participant holdout.

Run: python -m neuroedge.accuracy_v2 develop|confirm|verify --plan configs/accuracy_v2.yaml
The published v0.1 training implementation and results are preserved.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import random
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .config import load_config, load_config as _load_original, json_write, sha256, digest, environment
from .data import load_prepared, prepare, select_reference
from .models import EEGNet, TrainOnlyScaler
from .experiment import probabilities
from .learning_diagnostics import metrics
from .report import bootstrap_subjects


def validate_plan(plan):
    groups = {k: set(plan[k]) for k in ["train_subjects", "validation_subjects", "confirmation_subjects"]}
    if any(not v for v in groups.values()):
        raise ValueError("All participant groups must be nonempty.")
    for key, values in groups.items():
        if len(values) != len(plan[key]) or any(not 1 <= s <= 109 for s in values):
            raise ValueError("Invalid or duplicate participant identifiers.")
    if groups["train_subjects"] & groups["validation_subjects"] or groups["train_subjects"] & groups["confirmation_subjects"] or groups["validation_subjects"] & groups["confirmation_subjects"]:
        raise ValueError("Participant leakage between development and confirmation groups.")
    if groups["train_subjects"] | groups["validation_subjects"] != set(plan["development_subjects"]):
        raise ValueError("Development subjects must equal train plus validation subjects.")
    if plan["channels"] not in [8, 16, 64]:
        raise ValueError("Unsupported electrode configuration.")


def load_plan(filename):
    path = Path(filename).resolve()
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_plan(plan)
    plan["_root"] = str(path.parent.parent)
    return plan


def freeze_plan(plan):
    out = Path(plan["_root"]) / plan["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in plan.items() if not k.startswith("_")}
    paths = [Path(__file__), Path(__file__).with_name("models.py"), Path(__file__).with_name("data.py")]
    identity = {"plan": public, "sources": {p.name: sha256(p) for p in paths}}
    identity["sha256"] = digest(identity)
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text())["sha256"] != identity["sha256"]:
        raise RuntimeError("Frozen v2 plan/code differs. Preserve this run; use a separately specified new run.")
    if not path.exists():
        json_write(path, identity)
        json_write(out / "environment.json", environment())
    return out, identity


def data_config(plan, subjects, role):
    cfg = _load_original(Path(plan["_root"]) / "configs/main.yaml")
    cfg["name"] = plan["name"] + "_" + role
    cfg["scope"] = "v2 data preparation; model selection does not use confirmation participants"
    cfg["dataset"]["subjects"] = list(subjects)
    cfg["output_dir"] = plan["output_dir"] + "/" + role
    return cfg


def get_data(plan, subjects, role):
    # Reuse original cached development bytes read-only when present.
    if role == "development_data":
        original = _load_original(Path(plan["_root"]) / "configs/main.yaml")
        if subjects == original["dataset"]["subjects"]:
            try:
                return load_prepared(original)
            except FileNotFoundError:
                pass
    cfg = data_config(plan, subjects, role)
    try:
        return load_prepared(cfg)
    except FileNotFoundError:
        return prepare(cfg)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    def __init__(self, bands=((8, 12), (12, 16), (16, 22), (22, 30)), sfreq=128):
        self.bands = bands
        self.sfreq = sfreq

    def _band(self, X, band):
        sos = butter(4, band, btype="bandpass", fs=self.sfreq, output="sos")
        return np.ascontiguousarray(sosfiltfilt(sos, np.asarray(X, dtype=np.float64), axis=-1))

    def fit(self, X, y):
        from mne.decoding import CSP
        self.filters_ = []
        for band in self.bands:
            csp = CSP(n_components=4, reg=0.1, log=True, norm_trace=False)
            csp.fit(self._band(X, band), y)
            self.filters_.append(csp)
        return self

    def transform(self, X):
        return np.concatenate([csp.transform(self._band(X, band))
                               for band, csp in zip(self.bands, self.filters_)], axis=1)


class UnitTraceCovariances(BaseEstimator, TransformerMixin):
    """OAS SPD covariance, normalized per trial; no evaluation-batch statistics."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from pyriemann.estimation import Covariances
        cov = Covariances(estimator="oas").transform(np.asarray(X, dtype=np.float64))
        return cov / np.trace(cov, axis1=1, axis2=2)[:, None, None]


def classical(name, plan):
    from mne.decoding import CSP
    if name == "csp_lda":
        return Pipeline([("csp", CSP(n_components=4, reg=0.1, log=True, norm_trace=False)),
                         ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
    if name == "filterbank_csp_lda":
        return Pipeline([("fbcsp", FilterBankCSP(bands=tuple(tuple(v) for v in plan["filter_bank"]))),
                         ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
    if name == "tangent_logistic":
        from pyriemann.tangentspace import TangentSpace
        return Pipeline([("cov", UnitTraceCovariances()),
                         ("tangent", TangentSpace(metric="riemann", tsupdate=False)),
                         ("scale", StandardScaler()),
                         ("logistic", LogisticRegression(C=1.0, max_iter=2000, random_state=plan["seed"]))])
    raise ValueError(name)


def subject_scores(meta, p):
    rows = []
    for subject in sorted(meta.subject.unique()):
        idx = meta.subject.to_numpy() == subject
        rows.append({"subject": int(subject), **metrics(meta.loc[idx, "label"].to_numpy(np.int64), p[idx])})
    return pd.DataFrame(rows)


def fit_neural(X, meta, train_subjects, validation_subjects, spec, plan, directory, fixed_epochs=None):
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    train_idx = meta.subject.isin(train_subjects).to_numpy()
    val_idx = meta.subject.isin(validation_subjects).to_numpy()
    scaler = TrainOnlyScaler().fit(X[train_idx])
    train_X = scaler.transform(X[train_idx])
    labels = meta.loc[train_idx, "label"].to_numpy(np.int64)
    model = EEGNet(channels=X.shape[1], samples=X.shape[2], dropout=spec["dropout"], kernel_length=32)
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.momentum = spec["bn_momentum"]
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    tx, ty = torch.from_numpy(train_X[:, None]), torch.from_numpy(labels)
    rng = np.random.default_rng(plan["seed"])
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    maximum = fixed_epochs if fixed_epochs is not None else plan["epochs"]
    for epoch in range(1, maximum+1):
        model.train()
        order = rng.permutation(len(tx))
        total = 0.0
        for start in range(0, len(tx), plan["batch_size"]):
            batch = order[start:start+plan["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tx[batch]), ty[batch])
            loss.backward()
            optimizer.step()
            model.constrain()
            total += len(batch) * float(loss.detach())
        row = {"epoch": epoch, "train_loss_with_dropout": total / len(tx)}
        train_p = probabilities(model, train_X)
        row.update({"train_"+key: value for key, value in metrics(labels, train_p).items()})
        if fixed_epochs is None:
            val_p = probabilities(model, scaler.transform(X[val_idx]))
            scores = subject_scores(meta[val_idx].reset_index(drop=True), val_p)
            val_loss = float(scores.log_loss.mean())
            row["validation_balanced_accuracy"] = float(scores.balanced_accuracy.mean())
            row["validation_log_loss"] = val_loss
            if val_loss < best - 1e-5:
                best, best_epoch, best_state, stale = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
        else:
            best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
        history.append(row)
        if epoch % 10 == 0:
            print(f"{spec['name']} epoch={epoch} train_BA={row['train_balanced_accuracy']:.3f} val_BA={row.get('validation_balanced_accuracy',float('nan')):.3f}", flush=True)
        if fixed_epochs is None and epoch >= plan["minimum_epochs"] and stale >= plan["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    torch.save({"model_spec": model.spec, "state_dict": model.state_dict(), "bn_momentum": spec["bn_momentum"],
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "best_epoch": best_epoch,
                "train_subjects": list(train_subjects), "validation_subjects": list(validation_subjects),
                "candidate": spec}, directory / "model.pt")
    return model, scaler, best_epoch, len(history)


def neural_prediction(directory, X):
    from .models import load_checkpoint
    model, scaler, cp = load_checkpoint(directory / "model.pt")
    return probabilities(model, scaler.transform(X))


def develop(plan):
    import mne
    mne.set_log_level("ERROR")
    out, identity = freeze_plan(plan)
    if (out / "selection.json").exists():
        print("Development selection already frozen; not changing it.", flush=True)
        return json.loads((out / "selection.json").read_text())
    X, meta, names = get_data(plan, plan["development_subjects"], "development_data")
    selected = select_reference(X, names, plan["channels"])
    train_idx = meta.subject.isin(plan["train_subjects"]).to_numpy()
    val_idx = meta.subject.isin(plan["validation_subjects"]).to_numpy()
    val_meta = meta[val_idx].reset_index(drop=True)
    torch.set_num_threads(plan["threads"])
    rows = []
    with threadpool_limits(limits=plan["threads"]):
        candidates = [{"name": name} for name in plan["classical_candidates"]] + plan["neural_candidates"]
        for spec in candidates:
            directory = out / "development" / spec["name"]
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / "result.json"
            if result_path.exists():
                rows.append(json.loads(result_path.read_text()))
                continue
            print("Develop", spec["name"], flush=True)
            started = time.perf_counter()
            if spec["name"] in plan["classical_candidates"]:
                estimator = classical(spec["name"], plan)
                estimator.fit(selected[train_idx].astype(np.float64), meta.loc[train_idx, "label"])
                p = estimator.predict_proba(selected[val_idx].astype(np.float64))
                train_p = estimator.predict_proba(selected[train_idx].astype(np.float64))
                joblib.dump(estimator, directory / "model.joblib")
                epoch, executed = None, None
                kind = "classical"
            else:
                model, scaler, epoch, executed = fit_neural(selected, meta, plan["train_subjects"], plan["validation_subjects"], spec, plan, directory)
                p = probabilities(model, scaler.transform(selected[val_idx]))
                train_p = probabilities(model, scaler.transform(selected[train_idx]))
                kind = "neural"
            scores = subject_scores(val_meta, p)
            scores.to_csv(directory / "validation_by_subject.csv", index=False)
            row = {"name": spec["name"], "kind": kind, "spec": spec,
                   "validation_balanced_accuracy": float(scores.balanced_accuracy.mean()),
                   "validation_log_loss": float(scores.log_loss.mean()),
                   "train_balanced_accuracy": metrics(meta.loc[train_idx, "label"].to_numpy(np.int64), train_p)["balanced_accuracy"],
                   "best_epoch": epoch, "epochs_executed": executed, "seconds": time.perf_counter()-started,
                   "protocol_sha256": identity["sha256"]}
            json_write(result_path, row)
            rows.append(row)
            print("Validation", spec["name"], row["validation_balanced_accuracy"], flush=True)
    ranked = sorted(rows, key=lambda r: (-r["validation_balanced_accuracy"], r["validation_log_loss"], r["name"]))
    table = pd.DataFrame([{k:v for k,v in row.items() if k not in ["spec"]} for row in ranked])
    table.to_csv(out / "development_summary.csv", index=False)
    selection = {"selected": ranked[0], "protocol_sha256": identity["sha256"],
                 "confirmation_subjects": plan["confirmation_subjects"],
                 "selection_used_confirmation_data": False,
                 "interpretation": "Development comparison only; no final accuracy claim until independent confirmation."}
    json_write(out / "selection.json", selection)
    print(table.to_string(index=False), flush=True)
    return selection


def confirm(plan):
    import mne
    mne.set_log_level("ERROR")
    out, identity = freeze_plan(plan)
    selection_path = out / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("Run development and freeze its selection before confirmation.")
    if (out / "confirmation" / "complete.json").exists():
        print("Confirmation already complete; use verify to reproduce saved predictions.", flush=True)
        return
    selection = json.loads(selection_path.read_text())
    selection_sha = sha256(selection_path)
    if selection["protocol_sha256"] != identity["sha256"]:
        raise RuntimeError("Development and confirmation identities differ.")
    # This receipt is written BEFORE confirmation data acquisition/scoring.
    json_write(out / "confirmation_started.json", {"selection_sha256": selection_sha,
               "selected": selection["selected"]["name"], "protocol_sha256": identity["sha256"],
               "confirmation_subjects": plan["confirmation_subjects"]})
    X, meta, names = get_data(plan, plan["development_subjects"], "development_data")
    test_X, test_meta, test_names = get_data(plan, plan["confirmation_subjects"], "confirmation_data")
    if set(meta.subject) & set(test_meta.subject):
        raise AssertionError("Confirmation participant leakage.")
    if names != test_names:
        raise AssertionError("Development and confirmation channel ordering differ.")
    train = select_reference(X, names, plan["channels"])
    test = select_reference(test_X, test_names, plan["channels"])
    torch.set_num_threads(plan["threads"])
    root = out / "confirmation"
    root.mkdir(exist_ok=True)
    chosen = selection["selected"]
    specs = [({"name":"csp_lda"}, "classical", None),
             ({"name":"eegnet_old_settings_25epochs", "bn_momentum":0.01, "dropout":0.5}, "neural", 25),
             (chosen["spec"], chosen["kind"], chosen["best_epoch"])]
    frames, score_frames = [], []
    seen = set()
    with threadpool_limits(limits=plan["threads"]):
        for spec, kind, epochs in specs:
            name = spec["name"]
            if name in seen:
                continue
            seen.add(name)
            directory = root / name
            directory.mkdir(exist_ok=True)
            print("Confirm once", name, flush=True)
            if kind == "classical":
                estimator = classical(name, plan)
                estimator.fit(train.astype(np.float64), meta.label.to_numpy(np.int64))
                p = estimator.predict_proba(test.astype(np.float64))
                joblib.dump(estimator, directory / "model.joblib")
            else:
                model, scaler, _, _ = fit_neural(train, meta, plan["development_subjects"], [], spec, plan, directory,
                                                fixed_epochs=int(epochs))
                p = probabilities(model, scaler.transform(test))
            predictions = test_meta[["trial_id", "subject", "run", "onset", "label"]].copy()
            predictions["model"], predictions["prediction"], predictions["prob_right"] = name, p.argmax(1), p[:,1]
            predictions.to_csv(directory / "predictions.csv", index=False)
            frames.append(predictions)
            scores = subject_scores(test_meta, p)
            scores["model"] = name
            score_frames.append(scores)
            json_write(directory / "complete.json", {"kind": kind, "selection_sha256": selection_sha,
                       "train_subjects": plan["development_subjects"], "confirmation_subjects": plan["confirmation_subjects"],
                       "predictions_sha256": sha256(directory / "predictions.csv")})
    pd.concat(frames, ignore_index=True).to_csv(root / "predictions.csv", index=False)
    per_subject = pd.concat(score_frames, ignore_index=True)
    per_subject.to_csv(root / "metrics_by_subject.csv", index=False)
    summary = []
    for name, group in per_subject.groupby("model"):
        lo, hi = bootstrap_subjects(group.balanced_accuracy)
        summary.append({"model": name, "subjects": len(group), "balanced_accuracy": float(group.balanced_accuracy.mean()),
                        "ci_low": lo, "ci_high": hi, "accuracy": float(group.accuracy.mean())})
    summary = pd.DataFrame(summary)
    summary.to_csv(root / "summary.csv", index=False)
    json_write(root / "complete.json", {"protocol_sha256": identity["sha256"], "selection_sha256": selection_sha,
               "models": sorted(seen), "participants": int(test_meta.subject.nunique()), "trials": len(test_meta),
               "selection_used_confirmation_data": False})
    if sha256(selection_path) != selection_sha:
        raise AssertionError("Selection changed during confirmation.")
    write_report(plan, out, summary, chosen)
    print(summary.to_string(index=False), flush=True)


def write_report(plan, out, summary, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = ["# NeuroEdge v0.2 — prospective confirmation", "",
             "The v0.1 20-participant cohort is now development data. All candidate choices used subjects 1–16 for fitting and 17–20 for validation. "
             "Subjects 21–30 were reserved for confirmation and acquired/scored only after selection was frozen. "
             "This is a new participant holdout, not a rerun of the v0.1 five-fold benchmark.", "",
             f"Fixed channels: {plan['channels']}; training seed: {plan['seed']}; selected candidate: {chosen['name']}. "
             f"Its development balanced accuracy was {chosen['validation_balanced_accuracy']:.3f}. "
             "All final models refit using the 20 development participants. Neural epoch counts were frozen before confirmation; "
             "the old-settings neural baseline uses a fixed 25 epochs. No confirmation data are used for scaling, tangent reference, checkpoint selection or calibration.", "",
             "| Model | Confirmation participants | Balanced accuracy | Descriptive 95% participant interval | Accuracy |",
             "|---|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.model} | {r.subjects} | {r.balanced_accuracy:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.accuracy:.3f} |")
    lines += ["", "## Interpretation boundaries", "",
              "Use the baselines measured on these same confirmation participants for comparisons. Do not subtract the v0.1 54.8% score: "
              "it used different evaluation participants and training sizes. Ten participants, one training seed, one validation split and one public dataset "
              "give a pilot result, not a universal performance guarantee. Intervals are conditional descriptive participant bootstrap intervals. "
              "The four-band CSP uses extra fixed zero-phase filters on the existing offline epochs. The tangent model uses OAS covariance, "
              "per-trial unit-trace normalization, a training-fitted Riemannian tangent reference with tsupdate=False, training-fitted feature scaling, "
              "and logistic regression. It uses no evaluation-batch reference. Longer neural training and BatchNorm momentum changes were candidates, "
              "not assumed improvements. No clinical, real-time, MCU, energy, algorithmic-novelty or state-of-the-art claim.", "",
              "## Reproduction", "",
              "Run python -m neuroedge.accuracy_v2 develop --plan configs/accuracy_v2.yaml, then confirm, then verify. "
              "Existing frozen runs are preserved. See development_summary.csv, selection.json, confirmation_started.json, "
              "the confirmation CSV files, individual model artifacts and training histories.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9,4.4), layout="constrained")
    y = summary.balanced_accuracy.to_numpy()*100
    positions = np.arange(len(summary))
    ax.errorbar(positions, y, yerr=np.stack([y-summary.ci_low.to_numpy()*100, summary.ci_high.to_numpy()*100-y]),
                fmt="o", capsize=5, color="#2563eb")
    ax.axhline(50, linestyle="--", color="#94a3b8")
    ax.set(xticks=positions, xticklabels=summary.model.tolist(), ylabel="Subject-mean balanced accuracy (%)",
           title="Fresh unseen-participant confirmation · 10 participants")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(out / "confirmation_accuracy.png", dpi=180)
    plt.close(fig)


def verify(plan):
    out, identity = freeze_plan(plan)
    selection_sha = sha256(out / "selection.json")
    receipt = json.loads((out / "confirmation_started.json").read_text())
    complete = json.loads((out / "confirmation" / "complete.json").read_text())
    if selection_sha != receipt["selection_sha256"] or selection_sha != complete["selection_sha256"]:
        raise AssertionError("Confirmation selection is not the frozen selection.")
    X, meta, names = get_data(plan, plan["confirmation_subjects"], "confirmation_data")
    test = select_reference(X, names, plan["channels"])
    torch.set_num_threads(plan["threads"])
    from .models import load_checkpoint
    train_X, train_meta, train_names = get_data(plan, plan["development_subjects"], "development_data")
    train = select_reference(train_X, train_names, plan["channels"])
    checked = []
    with threadpool_limits(limits=plan["threads"]):
        for name in complete["models"]:
            directory = out / "confirmation" / name
            info = json.loads((directory / "complete.json").read_text())
            predictions = pd.read_csv(directory / "predictions.csv").set_index("trial_id").loc[meta.trial_id]
            if info["kind"] == "classical":
                estimator = joblib.load(directory / "model.joblib")
                if "tangent" in estimator.named_steps and estimator.named_steps["tangent"].tsupdate:
                    raise AssertionError("Transductive tangent reference forbidden.")
                p = estimator.predict_proba(test.astype(np.float64))
                # Prediction must be independent of other evaluation examples.
                for i in [0, len(test)//2, len(test)-1]:
                    np.testing.assert_allclose(estimator.predict_proba(test[i:i+1].astype(np.float64)), p[i:i+1], atol=1e-6, rtol=1e-5)
            else:
                model, scaler, cp = load_checkpoint(directory / "model.pt")
                expected = TrainOnlyScaler().fit(train)
                np.testing.assert_allclose(scaler.mean_, expected.mean_, rtol=0, atol=1e-7)
                np.testing.assert_allclose(scaler.scale_, expected.scale_, rtol=0, atol=1e-7)
                if set(cp["train_subjects"]) != set(plan["development_subjects"]):
                    raise AssertionError("Neural scaling/training provenance differs.")
                p = probabilities(model, scaler.transform(test))
            np.testing.assert_allclose(predictions.prob_right.to_numpy(), p[:,1], atol=1e-6, rtol=1e-5)
            np.testing.assert_array_equal(predictions.prediction.to_numpy(), p.argmax(1))
            np.testing.assert_array_equal(predictions.label.to_numpy(), meta.label.to_numpy())
            checked.append(name)
    result = {"status":"PASS", "models_checked":checked, "independent_confirmation_subjects":plan["confirmation_subjects"],
              "selection_unchanged":True, "recomputed_predictions_match":True,
              "batch_independent_classical_inference":True, "training_only_neural_scaling":True,
              "note":"Verification does not prove clinical validity or remove the pilot study's statistical limitations."}
    json_write(out / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["develop","confirm","verify"])
    parser.add_argument("--plan", default="configs/accuracy_v2.yaml")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    {"develop":develop,"confirm":confirm,"verify":verify}[args.command](plan)


if __name__ == "__main__":
    # Canonical imports keep custom sklearn estimators portable in joblib files.
    from neuroedge.accuracy_v2 import main as canonical_main
    canonical_main()
