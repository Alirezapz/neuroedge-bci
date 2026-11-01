# Working on NeuroEdge

This research engineering repository contains archived benchmarks and a broader frontier study. It is not a clinical product and currently claims no new EEG algorithm. Research quality and evidence determine the scope; no degree level is imposed.

- Read README.md, docs/METHODS.md and the chosen YAML configuration before changing methods.
- Preserve completed experiments. Use a new configuration name/output directory for changed training methods, cohorts or hyperparameters. Never merge incompatible predictions or remove training identity guards to force a resume.
- Do not tune methods on outer-test performance. Select hyperparameters using training/validation participants only and document any new evaluation protocol before its first test evaluation.
- Keep all participants and trials disjoint as specified in splits.json. Fit population statistics, CSP, neural normalization and quantization calibration using training data only.
- Use only real EEG for research metrics and demos. Synthetic fixtures are acceptable only for clearly labeled software tests. No fabricated results, citations or device measurements.
- Keep the reduced-channel reference restricted to the selected electrodes. Do not hide access to all 64 channels in an 8-channel experiment.
- Run relevant software tests and, for completed experiments, the independent `neuroedge verify` command. A passing unit test suite alone is not evidence of scientific validity.
- Preserve source attribution and the dataset's ODC Attribution License. Raw EDF and large caches are excluded from the deliverable and version control; exact download sources and hashes are retained.
- Label offline filtering, recorded-data replay, mixed-precision quantization and model-only CPU timing accurately. MCU, energy, real-time and clinical claims require their own actual evidence.
- Work incrementally and retain exact commands, versions, raw predictions, checkpoints, split manifests, and negative results.

The frontier extension is a broader research study, not constrained to a degree
level. Preserve the zero-calibration arm. Its explicitly separate secondary
rest-calibrated arm may estimate participant-specific alignment from that
participant's R01 eyes-open resting recording only, without task trials or
labels. Report its calibration requirement separately; never merge its scores
with the zero-calibration benchmark.

