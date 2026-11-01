# Commands executed for this study

Run from the project root with its installed Python dependencies. The actual
host used Python3.12 and CPU-only Torch2.9.1; full versions are in environment.json.
The execution used PYTHONPATH=src, OPENBLAS_NUM_THREADS=4, OMP_NUM_THREADS=4 and
MKL_NUM_THREADS=4, with the Python interpreter from the existing project venv.

```bash
python -m neuroedge.frontier all --plan configs/frontier.yaml
```

This executes screening, secondary validation/selection, final source refitting,
one fresh confirmation, independent verification, then the derived report.

Source-only diagnostic controls were executed separately, without changing the
frozen model search or using confirmation data:

```bash
python scripts/frontier_controls.py --plan configs/frontier.yaml
```

Software checks include synthetic fixtures (never research accuracy) and a
small real-development-signal functional probe. The final complete unit and
integration suite is recorded in tests/frontier_test_results.xml. The independent
archived75-model inference check is recorded in tests/archive_main_verify.txt.

```bash
python -m pytest -q --junitxml=tests/frontier_test_results.xml
python scripts/frontier_collect_evidence.py --plan configs/frontier.yaml
```

The last command validates every screening fold's source-person/trial receipts,
checks all temporary fitted-model hashes and copies audit receipts and neural
learning histories into the portable results. It does not refit or select models.

For another host, create a fresh named configuration before execution. Preserved
completed outputs must not be mixed with different configuration, data bytes,
training source or numerical settings.
