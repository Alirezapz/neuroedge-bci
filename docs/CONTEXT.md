# Pre-cue contrast and left/right symmetry

The context study compares task-only features with task-minus-pre-cue contrasts,
left/right reflection, covariance features and compact filter-bank neural
models. It evaluates 63 base configurations in three participant-group source
folds. Source data contain 104 valid PhysioNet people; all previously inspected
people are development data. Source selection freezes before original BNCI
acquisition. All methods use the same 16 sensorimotor electrodes.

Task windows are 0.5–4 seconds after cue. Context requires an extra 1.5-second
rest segment ending 0.25 seconds before cue. The baseline extractor cannot
access future task samples. This extra input requirement must be reported.

Direct transfer uses source-only final models with no BNCI fitting. External
LOSO refits the frozen algorithms on eight BNCI people while excluding BOTH
sessions of the ninth. These are different training conditions. Personal
calibration additionally needs labels from the new person's first session.

Read `CONTEXT_METHODS.md` for exact preprocessing, fixed selections and limits.
The complete results are in `runs/context_fusion/REPORT.md` and `results.html`.
The new session-calibration extension is separate and exploratory.

## Windows reproduction

From PowerShell in the folder containing `pyproject.toml`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/new_run.py --config configs/context_fusion.yaml --name context_local
.\.venv\Scripts\python.exe -m neuroedge.context_study all --plan configs/context_local.yaml
```

Use the generated configuration consistently. Archived experiments reject
changed code/data identities. To restore the same archived context inputs,
`scripts/context_continue.py --plan configs/context_fusion.yaml` requires exact
fingerprint agreement. Do not remove guards to force a resume. For the complete
gating and personal suite, use the linked guide rather than running child
registration after external acquisition.
