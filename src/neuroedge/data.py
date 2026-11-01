"""Real EDF acquisition, audit, and offline epoch preparation. Never synthesizes data."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from .config import digest, initialize_run, json_write, output_path, project_path, sha256

DATASET_URL = "https://physionet.org/content/eegmmidb/1.0.0/"
# Official public bucket advertised on the dataset page. No credentials required.
SOURCES = ["https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0",
           "https://www.physionet.org/files/eegmmidb/1.0.0"]
CHANNELS_8 = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"]
CHANNELS_16 = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4",
               "FC1", "FC2", "C1", "C2", "CP1", "CP2", "C5", "C6"]
# Bilaterally symmetric sensorimotor sets, nested and fixed before results.


def event_label(description, run):
    if run not in (4, 8, 12):
        raise ValueError("T1/T2 meanings depend on run: only imagery runs 4/8/12 allowed.")
    if description == "T1":
        return 0  # imagine opening/closing left fist
    if description == "T2":
        return 1  # imagine opening/closing right fist
    if description == "T0":
        return None
    raise ValueError(f"Unexpected annotation {description!r}.")


def channel_indices(names, count):
    selected = list(names) if count == 64 else {8: CHANNELS_8, 16: CHANNELS_16}.get(count)
    if selected is None or (count == 64 and len(names) != 64):
        raise ValueError("Supported channel configurations: 64, 16, 8.")
    missing = set(selected) - set(names)
    if missing:
        raise ValueError(f"Missing required channels: {sorted(missing)}")
    return [names.index(x) for x in selected]


def select_reference(X, names, count):
    """Average reference uses ONLY the channels available in this configuration."""
    idx = channel_indices(names, count)
    selected = np.asarray(X[:, idx, :], dtype=np.float32)
    return selected - selected.mean(axis=1, keepdims=True)


def _download_one(cfg, subject, run):
    relative = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
    path = project_path(cfg, cfg["dataset"]["raw_dir"]) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(".source.json")
    if path.exists() and sidecar.exists():
        provenance = json.loads(sidecar.read_text())
        if sha256(path) != provenance["sha256"]:
            raise RuntimeError(f"Cached file hash changed: {path}; inspect before replacing it.")
        return path
    errors = []
    for base in SOURCES:
        url = f"{base}/{relative}"
        tmp = path.with_suffix(".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NeuroEdge-research/0.1"})
            with urllib.request.urlopen(req, timeout=cfg["dataset"]["download_timeout"]) as r, tmp.open("wb") as f:
                for chunk in iter(lambda: r.read(1024 * 1024), b""):
                    f.write(chunk)
            with tmp.open("rb") as f:
                if f.read(8) != b"0       ":
                    raise ValueError("Not an EDF header (possibly an error page).")
            tmp.replace(path)
            json_write(sidecar, {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size})
            return path
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("Real EEG download failed; no substitute data generated.\n" + "\n".join(errors))


def download(cfg):
    requests = [(s, r) for s in cfg["dataset"]["subjects"] for r in cfg["dataset"]["runs"]]
    with ThreadPoolExecutor(max_workers=cfg["dataset"].get("download_workers", 4)) as pool:
        jobs = {pool.submit(_download_one, cfg, s, r): (s, r) for s, r in requests}
        for i, future in enumerate(as_completed(jobs), 1):
            future.result()
            s, r = jobs[future]
            print(f"EDF {i}/{len(jobs)}: S{s:03d} R{r:02d}", flush=True)


def prepare(cfg):
    import mne
    from mne.datasets.eegbci import standardize
    mne.set_log_level("ERROR")
    out = initialize_run(cfg)
    download(cfg)
    pp = cfg["preprocessing"]
    signature = digest({"subjects": cfg["dataset"]["subjects"], "runs": cfg["dataset"]["runs"],
                        "preprocessing": pp, "implementation": 1})[:16]
    cache = project_path(cfg, cfg["dataset"]["cache_dir"]) / signature
    cache.mkdir(parents=True, exist_ok=True)
    arrays, records, audit, names = [], [], [], None
    failures = []
    for subject in cfg["dataset"]["subjects"]:
        for run in cfg["dataset"]["runs"]:
            path = project_path(cfg, cfg["dataset"]["raw_dir"]) / f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
            item = {"subject": subject, "run": run, "source_sha256": sha256(path), "included": False}
            try:
                raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
                standardize(raw)
                raw.pick("eeg")
                if len(raw.ch_names) != 64:
                    raise ValueError(f"Expected 64 EEG channels, got {len(raw.ch_names)}.")
                if names is None:
                    names = list(raw.ch_names)
                    for count in (8, 16, 64):
                        channel_indices(names, count)
                if set(raw.ch_names) != set(names):
                    raise ValueError("Channel names differ from the frozen montage.")
                raw.reorder_channels(names)
                native_fs = float(raw.info["sfreq"])
                if not np.isfinite(raw.get_data()).all():
                    raise ValueError("Non-finite source signal.")
                annotations = []
                counts = {"left": 0, "right": 0, "rest": 0}
                for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
                    label = event_label(str(desc), run)
                    counts[{0: "left", 1: "right", None: "rest"}[label]] += 1
                    if label is not None:
                        if float(duration) + 1 / native_fs < pp["tmin"] + pp["duration"]:
                            raise ValueError(f"Task annotation too short at {onset}s: {duration}s.")
                        annotations.append((float(onset), float(duration), label))
                if min(counts["left"], counts["right"]) < 2:
                    raise ValueError("Each run must include both imagery classes.")
                # Fixed zero-phase IIR: offline filtering, NOT a causal online system.
                raw.filter(pp["low_hz"], pp["high_hz"], method="iir",
                           iir_params={"order": 4, "ftype": "butter", "output": "sos"},
                           phase="zero", verbose="ERROR")
                raw.resample(pp["sfreq"], verbose="ERROR")
                samples = raw.get_data(units="uV")
                n_times = int(round(pp["duration"] * pp["sfreq"]))
                for trial, (onset, duration, label) in enumerate(annotations):
                    start = int(round((onset + pp["tmin"]) * pp["sfreq"]))
                    epoch = samples[:, start:start + n_times]
                    if epoch.shape != (64, n_times):
                        raise ValueError("Incomplete epoch at recording boundary.")
                    arrays.append(epoch.astype(np.float32))
                    records.append({"trial_id": f"S{subject:03d}_R{run:02d}_T{trial:03d}",
                                    "subject": subject, "run": run, "onset": onset,
                                    "annotation_duration": duration, "label": label})
                item.update({"included": True, "native_sfreq": native_fs,
                             "duration_seconds": raw.n_times / raw.info["sfreq"],
                             "classes": counts, "n_channels": 64,
                             "peak_abs_uv_after_filter": float(np.abs(samples).max())})
                print(f"Prepared S{subject:03d} R{run:02d}: {counts}", flush=True)
            except Exception as e:
                item["reason"] = f"{type(e).__name__}: {e}"
                failures.append(item)
            audit.append(item)
    json_write(out / "data_audit.json", {"dataset": DATASET_URL, "records": audit,
                                         "exclusions": failures, "raw_data_replaced": False})
    if failures:
        # Do not silently replace a difficult/malformed subject or use partial subjects.
        raise RuntimeError(f"{len(failures)} failed recordings. See data_audit.json; cohort not changed automatically.")
    X = np.stack(arrays)
    meta = pd.DataFrame(records)
    np.savez_compressed(cache / "epochs.npz", X=X, y=meta.label.to_numpy(np.int64),
                        subjects=meta.subject.to_numpy(np.int64), channels=np.asarray(names))
    meta.to_csv(cache / "trials.csv", index=False)
    info = {"cache": str(cache.relative_to(Path(cfg["_root"]))),
            "n_trials": len(meta), "subjects": sorted(meta.subject.unique().tolist()),
            "channels": names, "sfreq": pp["sfreq"], "shape": list(X.shape),
            "epochs_sha256": sha256(cache / "epochs.npz"),
            "trials_sha256": sha256(cache / "trials.csv"), "preprocessing": pp}
    json_write(out / "data_manifest.json", info)
    print(f"REAL EEG prepared: {X.shape}; {len(info['subjects'])} subjects", flush=True)
    return X, meta, names


def load_prepared(cfg):
    path = output_path(cfg) / "data_manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run `neuroedge prepare --config ...` first. No synthetic fallback exists.")
    manifest = json.loads(path.read_text())
    cache = project_path(cfg, manifest["cache"])
    for filename, key in [("epochs.npz", "epochs_sha256"), ("trials.csv", "trials_sha256")]:
        if sha256(cache / filename) != manifest[key]:
            raise RuntimeError(f"Prepared data integrity check failed: {filename}")
    with np.load(cache / "epochs.npz", allow_pickle=False) as loaded:
        X, names = loaded["X"], loaded["channels"].tolist()
    meta = pd.read_csv(cache / "trials.csv")
    if len(meta) != len(X) or meta.trial_id.duplicated().any():
        raise RuntimeError("Trial identity mismatch.")
    return X, meta, names

