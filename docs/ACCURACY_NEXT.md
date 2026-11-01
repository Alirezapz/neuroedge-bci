# Accuracy investigation after smoke

A smoke test verifies software execution. Its tiny cohort and three epochs do
not establish useful predictive performance. The accuracy extension separates
learning diagnostics, development and prospective confirmation.

The completed small study used16 training and4 validation people followed by10
new people. Longer training was selected on validation but did not outperform
matched-holdout references. The expanded study used60 training,10 validation
and10 different confirmation people, with refitting on70 development people
before confirmation. This sequential study does not isolate a causal effect
of dataset size. Both positive and negative findings remain archived.

Keep local data and `.venv` when extracting updated source. Open PowerShell
in the folder containing `pyproject.toml`; use the explicit environment Python
if an activated `python` resolves elsewhere.

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/new_run.py --config configs/accuracy_expanded.yaml --name accuracy_expanded_local
python -m neuroedge.accuracy_expanded develop --plan configs/accuracy_expanded_local.yaml
python -m neuroedge.accuracy_expanded confirm --plan configs/accuracy_expanded_local.yaml
python -m neuroedge.accuracy_expanded verify --plan configs/accuracy_expanded_local.yaml
python -m neuroedge.accuracy_present --plan configs/accuracy_expanded_local.yaml
```

Create a local named configuration only once. Do not reuse confirmation results
to retune a model while still calling the same people untouched confirmation.
The report, learning curves and stored-prediction replay appear in the named
run directory. The replay displays eight of the selected decoder's sixteen
electrodes and performs no live inference. Subsequent broader studies and the
current cross-session expansion are documented separately in README.md.
