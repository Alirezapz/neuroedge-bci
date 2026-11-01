"""Explicit secondary arm: alignment using a separate eyes-open baseline (R01).

This arm requires an unlabeled resting recording from each new participant.
It never estimates alignment from task/test trials, their labels, or other
evaluation participants. It must not be reported as zero-calibration decoding.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .config import digest, json_write, sha256
from .frontier_data import (BANDS, acquire, empirical_covariances,
                            referenced_covariances, shrink_covariances)
from .data import channel_indices


def inverse_sqrt(C):
    values,vectors=np.linalg.eigh(C)
    if (values<=0).any():
        raise ValueError("Alignment reference is not SPD.")
    return (vectors * values**-.5) @ vectors.T


def prepare_rest(plan,cohort,role):
    import mne
    from mne.datasets.eegbci import standardize

    root=Path(plan["_root"])
    manifest_path=root/plan["output_dir"]/role/"rest_manifest.json"
    if manifest_path.exists():
        return load_rest(plan,cohort,role)
    subjects=cohort.manifest["subjects"]
    rest_plan=copy.deepcopy(plan)
    rest_plan["dataset"]["runs"]=[1]
    acquire(rest_plan,subjects)
    cache=root/plan["dataset"]["cache_dir"]/"rest"/sha256(__file__)[:16]
    cache.mkdir(parents=True,exist_ok=True)
    references=[]
    audit=[]
    for i,subject in enumerate(subjects,1):
        source=root/plan["dataset"]["raw_dir"]/f"S{subject:03d}/S{subject:03d}R01.edf"
        target=cache/f"S{subject:03d}.npy"
        receipt=target.with_suffix(".json")
        signature=digest({"source":sha256(source),"implementation":sha256(__file__),
                          "preparation":sha256(Path(__file__).with_name("frontier_data.py"))})
        if target.exists() and receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous["signature"]!=signature or previous["sha256"]!=sha256(target):
                raise RuntimeError("Resting reference cache identity changed.")
            C=np.load(target,allow_pickle=False)
        else:
            raw=mne.io.read_raw_edf(source,preload=True,verbose="ERROR")
            standardize(raw)
            raw.pick("eeg")
            if set(raw.ch_names)!=set(cohort.names):
                raise ValueError(f"Resting channel mismatch for S{subject:03d}.")
            raw.reorder_channels(cohort.names)
            if raw.times[-1]<30 or not np.isfinite(raw.get_data()).all():
                raise ValueError(f"Invalid separate resting recording for S{subject:03d}.")
            covariances=[]
            for low,high in BANDS:
                filtered=raw.copy().filter(low,high,method="iir",
                    iir_params={"order":4,"ftype":"butter","output":"sos"},phase="zero",verbose="ERROR")
                filtered.resample(128,verbose="ERROR")
                # Fixed trimming removes recording/filter boundary transients.
                values=filtered.get_data(units="uV")[:,128:-128]
                covariances.append(empirical_covariances(values[None])[0])
            C=np.stack(covariances)
            np.save(target,C,allow_pickle=False)
            json_write(receipt,{"signature":signature,"sha256":sha256(target),
                                "subject":subject,"source_sha256":sha256(source)})
        references.append(C)
        audit.append({"subject":subject,"source_run":1,"task_trials_used":0,
                      "source_sha256":sha256(source),"cache_sha256":sha256(target)})
        if i%10==0 or i==len(subjects):
            print(f"Separate resting references ready {i}/{len(subjects)}",flush=True)
    path=cache/(digest(subjects)[:16]+".npy")
    np.save(path,np.stack(references),allow_pickle=False)
    manifest={"subjects":subjects,"path":str(path.relative_to(root)),"sha256":sha256(path),
              "source_sha256":sha256(__file__),"audit":audit,
              "scope":"Secondary rest-calibrated arm; separate R01 only, no task-test statistic or label."}
    json_write(manifest_path,manifest)
    return load_rest(plan,cohort,role)


def load_rest(plan,cohort,role):
    root=Path(plan["_root"])
    manifest=json.loads((root/plan["output_dir"]/role/"rest_manifest.json").read_text())
    if manifest["source_sha256"]!=sha256(__file__) or sha256(root/manifest["path"])!=manifest["sha256"]:
        raise RuntimeError("Resting calibration identity changed.")
    if manifest["subjects"]!=cohort.manifest["subjects"]:
        raise RuntimeError("Resting and task participant identities differ.")
    return RestCalibratedCohort(cohort,np.load(root/manifest["path"],allow_pickle=False))


class RestCalibratedCohort:
    def __init__(self,cohort,rest):
        self.base,self.rest=cohort,rest
        self.manifest,self.meta,self.names=cohort.manifest,cohort.meta,cohort.names
        self._mapping={s:i for i,s in enumerate(self.manifest["subjects"])}
        self._transforms={}

    def matrices(self,spec):
        key=(spec["channels"],tuple(spec["bands"]))
        if key not in self._transforms:
            indices=channel_indices(self.names,spec["channels"])
            C=referenced_covariances(self.rest[:,spec["bands"]],indices)
            C=shrink_covariances(C,.05)
            self._transforms[key]=np.stack([np.stack([inverse_sqrt(c) for c in person]) for person in C])
        return self._transforms[key]

    def covariance_view(self,spec,rows):
        C=self.base.covariance_view(spec,rows)
        if spec.get("arm","zero_calibration")=="zero_calibration":
            return C
        subjects=self.meta.iloc[rows].subject.to_numpy()
        A=self.matrices(spec)[[self._mapping[int(s)] for s in subjects]]
        return A @ C @ np.swapaxes(A,-1,-2)

    def neural_view(self,spec,rows):
        X=self.base.neural_view(spec,rows)
        if spec.get("arm","zero_calibration")=="zero_calibration":
            return X
        subjects=self.meta.iloc[rows].subject.to_numpy()
        A=self.matrices(spec)[[self._mapping[int(s)] for s in subjects],0]
        return np.ascontiguousarray(np.einsum("ncd,ndt->nct",A,X),dtype=np.float32)
