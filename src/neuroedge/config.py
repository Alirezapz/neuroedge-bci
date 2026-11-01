"""Configuration identity and experiment provenance."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import yaml


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["_root"] = str(path.parent.parent)
    cfg["_config_path"] = str(path)
    subjects = cfg["dataset"]["subjects"]
    if len(set(subjects)) != len(subjects) or any(not 1 <= s <= 109 for s in subjects):
        raise ValueError("Subjects must be unique integers in [1, 109].")
    if cfg["dataset"]["runs"] != [4, 8, 12]:
        raise ValueError("This protocol is ONLY left/right hand imagery, runs 4/8/12.")
    if cfg["evaluation"]["folds"] < 2:
        raise ValueError("At least two outer folds are required.")
    if len(subjects) - (len(subjects) + cfg["evaluation"]["folds"] - 1) // cfg["evaluation"]["folds"] < 2:
        raise ValueError("Need at least two non-test subjects for train/validation.")
    if len(set(cfg["evaluation"]["seeds"])) != len(cfg["evaluation"]["seeds"]):
        raise ValueError("Duplicate seeds are not independent repetitions.")
    p = cfg["preprocessing"]
    if not (0 < p["low_hz"] < p["high_hz"] < p["sfreq"] / 2):
        raise ValueError("Invalid frequency bounds.")
    return cfg


def protocol(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def project_path(cfg, path):
    return Path(cfg["_root"]) / path


def output_path(cfg):
    return project_path(cfg, cfg["output_dir"])


def environment():
    packages = {}
    for p in ["numpy", "scipy", "pandas", "scikit-learn", "mne", "torch", "onnx", "onnxruntime", "matplotlib"]:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = "NOT INSTALLED"
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {"python": sys.version, "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "packages": packages}


def initialize_run(cfg):
    out = output_path(cfg)
    out.mkdir(parents=True, exist_ok=True)
    spec = protocol(cfg)
    signature = digest(spec)
    frozen = out / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text())["sha256"] != signature:
        raise RuntimeError("This output directory has a DIFFERENT frozen protocol. Choose a new output_dir.")
    if not frozen.exists():
        json_write(frozen, {"sha256": signature, "config": spec,
                           "note": "Frozen before data acquisition and test evaluation; not a public preregistration."})
    # Preserve the original execution environment when resuming on another host.
    if not (out / "environment.json").exists():
        json_write(out / "environment.json", environment())
    else:
        json_write(out / "last_invocation_environment.json", environment())
    source_root = Path(__file__).parent
    source_hashes = {p.name: sha256(p) for p in sorted(source_root.glob("*.py"))}
    if not (out / "source_hashes.json").exists():
        json_write(out / "source_hashes.json", source_hashes)
    return out

