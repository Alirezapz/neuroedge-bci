"""Versioned real-EEG preparation for the broader, prospective frontier study.

Filters each continuous recording before epoching. Covariance caches contain
unreferenced sensors; a model references ONLY its selected sensors later.
No population parameter or target-subject statistic is estimated here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .config import digest, json_write, sha256
from .data import _download_one, event_label, channel_indices

BANDS = [(4, 40), (8, 30), (4, 8), (8, 12), (12, 16), (16, 22), (22, 30), (30, 40)]
WINDOWS = {"full": (0.5, 4.0), "late": (1.0, 4.0)}


def acquire(plan, subjects):
    cfg = {"_root": plan["_root"], "dataset": copy.deepcopy(plan["dataset"])}
    requests = [(s, r) for s in subjects for r in plan["dataset"]["runs"]]

    def one(pair):
        for attempt in range(plan["dataset"]["download_attempts"]):
            try:
                return _download_one(cfg, *pair)
            except RuntimeError as error:
                if (not str(error).startswith("Real EEG download failed;") or
                        attempt + 1 == plan["dataset"]["download_attempts"]):
                    raise
                time.sleep(min(2 ** attempt, 15))

    with ThreadPoolExecutor(max_workers=plan["dataset"]["download_workers"]) as pool:
        jobs = {pool.submit(one, pair): pair for pair in requests}
        for i, future in enumerate(as_completed(jobs), 1):
            future.result()
            if i % 10 == 0 or i == len(jobs):
                print(f"EDF ready {i}/{len(jobs)}", flush=True)


def empirical_covariances(X):
    """Per-trial, time-centered covariance, denominator T, without shrinkage."""
    centered = np.asarray(X, dtype=np.float64)
    centered = centered - centered.mean(axis=-1, keepdims=True)
    return np.einsum("nct,ndt->ncd", centered, centered, optimize=True) / centered.shape[-1]


def referenced_covariances(cov, indices):
    """Equivalent to selecting raw sensors, average referencing, then cov()."""
    selected = np.asarray(cov[..., indices, :][..., :, indices], dtype=np.float64)
    return (selected - selected.mean(axis=-1, keepdims=True)
            - selected.mean(axis=-2, keepdims=True)
            + selected.mean(axis=(-2, -1), keepdims=True))


def shrink_covariances(cov, amount=0.05):
    """SPD diagonal loading, per trial; no population or evaluation-batch fit."""
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[-1]
    trace = np.trace(cov, axis1=-2, axis2=-1)
    if not np.isfinite(cov).all() or (trace <= 0).any():
        raise ValueError("Invalid covariance: nonfinite or nonpositive trace.")
    return ((1 - amount) * cov + amount * trace[..., None, None] / n * np.eye(n))


def validate_annotations(raw, run, quality):
    annotations = []
    counts = {0: 0, 1: 0}
    tolerance = 1 / float(raw.info["sfreq"])
    previous_end = -np.inf
    for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration,
                                     raw.annotations.description):
        label = event_label(str(desc), run)
        if onset + tolerance < previous_end:
            raise ValueError("Overlapping source annotations.")
        previous_end = float(onset + duration)
        if label is None:
            continue
        if duration + tolerance < quality["minimum_task_duration"]:
            raise ValueError(f"Task annotation shorter than {quality['minimum_task_duration']}s.")
        if onset < 0 or onset + WINDOWS["full"][1] > raw.times[-1] + tolerance:
            raise ValueError("Task window outside source recording.")
        counts[label] += 1
        annotations.append((float(onset), label))
    if min(counts.values()) < quality["minimum_trials_per_class_per_run"]:
        raise ValueError("Insufficient trials of one class.")
    return annotations


def _subject(plan, subject, cache):
    import mne
    from mne.datasets.eegbci import standardize

    mne.set_log_level("ERROR")
    root = Path(plan["_root"])
    paths = [root / plan["dataset"]["raw_dir"] /
             f"S{subject:03d}/S{subject:03d}R{r:02d}.edf" for r in plan["dataset"]["runs"]]
    sources = [{"relative_path": str(p.relative_to(root)), "sha256": sha256(p)} for p in paths]
    signature = digest({"sources": sources, "quality": plan["quality"], "bands": BANDS,
                        "windows": WINDOWS, "data_source": sha256(__file__)})
    target = cache / f"S{subject:03d}.npz"
    receipt_path = cache / f"S{subject:03d}.json"
    if target.exists() and receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved["signature"] != signature or saved["sha256"] != sha256(target):
            raise RuntimeError("Subject cache identity differs; do not silently replace it.")
        return saved

    all_wide, all_smr, all_cov, records, audit, names = [], [], [], [], [], None
    for path, run in zip(paths, plan["dataset"]["runs"]):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        standardize(raw)
        raw.pick("eeg")
        if len(raw.ch_names) != 64:
            raise ValueError("Expected exactly 64 EEG channels.")
        if names is None:
            names = list(raw.ch_names)
            channel_indices(names, 16)
        if set(raw.ch_names) != set(names):
            raise ValueError("Source channel sets differ.")
        raw.reorder_channels(names)
        if not np.isfinite(raw.get_data()).all():
            raise ValueError("Nonfinite source EEG.")
        annotations = validate_annotations(raw, run, plan["quality"])
        cov_bands, wide, smr = [], None, None
        for band_index, (low, high) in enumerate(BANDS):
            filtered = raw.copy().filter(low, high, method="iir",
                iir_params={"order": 4, "ftype": "butter", "output": "sos"},
                phase="zero", verbose="ERROR")
            filtered.resample(128, verbose="ERROR")
            values = filtered.get_data(units="uV")
            epochs = []
            for onset, label in annotations:
                start = round((onset + 0.5) * 128)
                epoch = values[:, start:start + 448]
                if epoch.shape != (64, 448):
                    raise ValueError("Incomplete epoch after resampling.")
                epochs.append(epoch.astype(np.float32))
            epochs = np.stack(epochs)
            if band_index == 0:
                wide = epochs
            if band_index == 1:
                smr = epochs
            # window dimension: full then late; all filters precede epoching.
            cov_bands.append(np.stack([empirical_covariances(epochs),
                                       empirical_covariances(epochs[..., 64:])], axis=1))
            del filtered
        all_wide.append(wide)
        all_smr.append(smr)
        all_cov.append(np.stack(cov_bands, axis=2).astype(np.float32))
        for trial, (onset, label) in enumerate(annotations):
            records.append({"trial_id": f"S{subject:03d}_R{run:02d}_T{trial:03d}",
                            "subject": subject, "run": run, "onset": onset, "label": label})
        audit.append({"run": run, "native_sfreq": float(raw.info["sfreq"]),
                      "trials": len(annotations), "source_sha256": sha256(path)})
    np.savez_compressed(target, wide=np.concatenate(all_wide), smr=np.concatenate(all_smr),
                        cov=np.concatenate(all_cov))
    receipt = {"subject": subject, "signature": signature, "sha256": sha256(target),
               "sources": sources, "records": records, "channels": names, "audit": audit}
    json_write(receipt_path, receipt)
    return receipt


def prepare_cohort(plan, subjects, role):
    root = Path(plan["_root"])
    out = root / plan["output_dir"] / role
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        return load_cohort(plan, role)
    acquire(plan, subjects)
    cache_key = digest({"data_source": sha256(__file__), "quality": plan["quality"]})[:16]
    subject_cache = root / plan["dataset"]["cache_dir"] / cache_key / "subjects"
    subject_cache.mkdir(parents=True, exist_ok=True)
    receipts, excluded = [], []
    for i, subject in enumerate(subjects, 1):
        try:
            receipt = _subject(plan, subject, subject_cache)
            receipts.append(receipt)
            print(f"Prepared {role} S{subject:03d} ({i}/{len(subjects)})", flush=True)
        except ValueError as error:
            # Frozen, label-performance-independent rule: exclude the WHOLE
            # participant if any required recording fails structural QC.
            excluded.append({"subject": subject, "reason": str(error)})
            print(f"STRUCTURAL QC EXCLUSION S{subject:03d}: {error}", flush=True)
    json_write(out / "audit.json", {"requested_subjects": subjects, "exclusions": excluded,
        "included_subjects": [r["subject"] for r in receipts], "records": receipts,
        "rule": "Participant excluded only by frozen structural source-data QC; never by classifier performance."})
    minimum = plan["quality"]["minimum_confirmation_subjects"] if role == "confirmation_data" else len(subjects)
    if len(receipts) < minimum:
        raise RuntimeError(f"Only {len(receipts)} structurally valid participants; need {minimum}. See {out/'audit.json'}.")
    records = [row for receipt in receipts for row in receipt["records"]]
    names = receipts[0]["channels"]
    if any(receipt["channels"] != names for receipt in receipts):
        raise RuntimeError("Cohort channel order differs.")
    data_dir = root / plan["dataset"]["cache_dir"] / cache_key / digest([r["signature"] for r in receipts])[:16]
    data_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    arrays = {
        "wide": np.lib.format.open_memmap(data_dir / "wide.npy", mode="w+", dtype="float32", shape=(n,64,448)),
        "smr": np.lib.format.open_memmap(data_dir / "smr.npy", mode="w+", dtype="float32", shape=(n,64,448)),
        "cov": np.lib.format.open_memmap(data_dir / "cov.npy", mode="w+", dtype="float32", shape=(n,2,len(BANDS),64,64))}
    cursor = 0
    for receipt in receipts:
        with np.load(subject_cache / f"S{receipt['subject']:03d}.npz", allow_pickle=False) as loaded:
            size = len(receipt["records"])
            for name, array in arrays.items():
                array[cursor:cursor + size] = loaded[name]
            cursor += size
    for array in arrays.values():
        array.flush()
    pd.DataFrame(records).to_csv(data_dir / "trials.csv", index=False)
    manifest = {"subjects": [r["subject"] for r in receipts], "n_trials": n,
                "channels": names, "bands": BANDS, "windows": WINDOWS,
                "sfreq": 128, "data_source_sha256": sha256(__file__),
                "files": {name: {"path": str((data_dir/name).relative_to(root)),
                                  "sha256": sha256(data_dir/name)}
                          for name in ["wide.npy", "smr.npy", "cov.npy", "trials.csv"]}}
    json_write(manifest_path, manifest)
    return load_cohort(plan, role)


class Cohort:
    def __init__(self, root, manifest):
        self.manifest = manifest
        self.meta = pd.read_csv(root / manifest["files"]["trials.csv"]["path"])
        self.wide = np.load(root / manifest["files"]["wide.npy"]["path"], mmap_mode="r")
        self.smr = np.load(root / manifest["files"]["smr.npy"]["path"], mmap_mode="r")
        self.cov = np.load(root / manifest["files"]["cov.npy"]["path"], mmap_mode="r")
        self.names = manifest["channels"]

    def covariance_view(self, spec, rows):
        indices = channel_indices(self.names, spec["channels"])
        window = 0 if spec["window"] == "full" else 1
        bands = spec["bands"]
        cov = np.asarray(self.cov[rows, window][:, bands], dtype=np.float64)
        return referenced_covariances(cov, indices)

    def neural_view(self, spec, rows):
        indices = channel_indices(self.names, spec["channels"])
        source = self.smr if spec["bands"] == [1] else self.wide
        X = np.asarray(source[rows][:, indices], dtype=np.float32)
        if spec["window"] == "late":
            X = X[..., 64:]
        return np.ascontiguousarray(X - X.mean(axis=1, keepdims=True))


def load_cohort(plan, role):
    root = Path(plan["_root"])
    manifest = json.loads((root / plan["output_dir"] / role / "manifest.json").read_text())
    if manifest["data_source_sha256"] != sha256(__file__):
        raise RuntimeError("Prepared-data implementation changed.")
    for item in manifest["files"].values():
        if sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError("Prepared frontier data hash changed.")
    return Cohort(root, manifest)
