# Broad frontier study

This archived study evaluates 224 predefined base configurations: CSP,
filter-bank CSP, tangent geometry, power classifiers, trees, EEGNet, compact
shallow log-power and temporal-attention adaptations. Three source group folds
precede a 20-person validation comparison of singles, averaging, weighted voting
and logistic stacking. Final neural candidates average fixed seeds 7,17,27.

The primary zero-calibration arm and a secondary arm requiring a separate R01
eyes-open resting recording are distinct. Rest alignment uses no task-test
labels, but it requires extra calibration EEG. Each channel condition references
only its own electrodes. Offline processing is not causal device validation.

The one prospective confirmation requested people81–109; 24 passed fixed
structural audits. Five exclusions preceded scoring. Earlier inspected
people71–80 are now development data. Negative ablations are retained.

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/new_run.py --config configs/frontier.yaml --name frontier_local
python -m neuroedge.frontier all --plan configs/frontier_local.yaml
```

On Windows use `.\.venv\Scripts\python.exe`. Keep existing data and environment
folders when updating source. Reproduction creates several GiB of caches; these
are omitted from the deliverable. Observed resource costs are recorded in
`runs/frontier/resources.json` and are not a universal runtime promise.

Open `runs/frontier/results.html` or `REPORT.md`; error analysis and raw
predictions are alongside them. `docs/FRONTIER_METHODS.md` defines exact choices.
The archived models belong to their original host. Fresh reproduction uses a
fresh plan/output name. `scripts/frontier_continue.py --plan configs/frontier.yaml`
restores the archived inputs only if original fingerprints match. Changes to
methods, dependencies that alter bytes, or cohorts require another run identity.
