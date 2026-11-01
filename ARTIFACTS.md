# Research artifacts and restoration

The GitHub distribution includes the source, English reports, aggregate
comparisons and scientific figures directly. Exact trained models, scalers,
large prediction tables and remaining audit records are stored as split ZIP
archives under `artifacts/`. These are ordinary Git files, without Git LFS,
an external model host or a paid API requirement.

After cloning or downloading **all** repository files:

```bash
python scripts/restore_model_bundle.py --verify-only
python scripts/restore_model_bundle.py
```

The command requires Python 3.11 or newer and only the standard library. It
restores 5,619 exact experiment files at their original relative paths. It
verifies each part's size and Git blob hash, ZIP integrity and every restored
file's SHA-256. Repeating restoration accepts identical existing files and
refuses to overwrite different local files. It never deserializes a model.
Allow roughly 1.5 GB free disk space for a complete clone, temporary archive
assembly and restoration. Individual file download links on GitHub are not a
substitute for downloading every archive part.

| Archive | Parts | Contents |
|---|---:|---|
| `model_artifacts.part*` | 64 | 1,002 earlier model, deployment and scaler files |
| `research_extension.part*` | 154 | 1,286 later model and large record files |
| `research_records.part*` | 8 | 3,331 remaining prediction, training and verification records |

[MANIFEST.json](artifacts/MANIFEST.json) contains the exact part order, file
sizes, integrity hashes and provenance. Learned model artifacts may retain
training-derived features. Load serialized joblib/pickle and PyTorch objects
only from a trusted source, with the pinned environment. Original dataset
terms remain applicable; see [DATA_LICENSES.md](DATA_LICENSES.md).

## Preserved studies

The latest corrected covariance study retains 71 candidate configurations,
7,668 donor fits, 972 saved selected model/stack artifacts and 217,728 actual
E prediction rows. All 168 reported method/requirement conditions and 1,092
paired comparisons are retained. The earlier session-adaptation study retains
540 saved artifacts, 54 deterministic fusion evaluations, 1,350 donor jobs,
85,536 prediction rows and 330 paired comparisons. These are repeated
evaluations of nine people, not new independent participants.

All earlier completed evidence, adverse findings, exact source snapshots,
split identities and verification receipts are preserved. The invalidated
covariance prototype is explicitly marked; its incomplete model output is
excluded. Publication changes affect packaging, navigation and licensing,
not the saved scientific results.

## Recording exclusions and local reconstruction

Raw EDF/MATLAB recordings, processed EEG caches, virtual environments,
browser executables, credentials, four recorded-signal replay HTML pages and
the earlier individual heatmap image are excluded. The HTML comparison
reports contain aggregate figures and tables. Download and open them locally.

Raw sources can be reacquired using the retained source URLs, file hashes and
preprocessing commands. `python scripts/restore_external_data.py` requires
the archived 18 raw BNCI hashes and five processed fingerprints to match.
A changed numerical environment or new method requires a new named run;
never remove an identity guard to force continuation.

Full license scope and source attribution are in DATA_LICENSES.md and
THIRD_PARTY.md. The root MIT license does not relicense either EEG dataset.
