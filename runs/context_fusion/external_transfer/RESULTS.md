# Pre-cue contrast, reflection and source-only fusion

A hypothesis-driven sequential exploratory study. The previously inspected PhysioNet people are development data. Neither their cross-validation nor reused validation is an independent new confirmation. All methods and external protocols were fixed before external acquisition.

63 base configurations completed 189 participant-group source fits. Source development: 80 people; reused selection: 24 people. 35 singles/combinations were compared for selection. Three fixed seeds are averaged for advanced neural validation/final fits; the historical neural reference retains seed 7.

## Input conditions

Every model uses the same 16 specified sensorimotor electrodes and their own common average reference. The trial-only condition requires [0.5,4) seconds of imagery after the cue. The context condition additionally requires [-1.75,-0.25) seconds of preceding rest from that same trial, not a task-batch estimate or an R01 recording. Task EEG uses the historical offline zero-phase bandpass. Pre-cue EEG uses TWO forward causal fourth-order Butterworth passes, matching the squared magnitude response, followed by resampling of the bounded past segment only. Its phase differs from task filtering. A software perturbation check ensures that task/future samples cannot change extracted context. The complete decoder is still offline.

The CSP and EEGNet references retain the 8–30 Hz, [1,4)-second canonical input. Advanced methods use [0.5,4) seconds and different bands/representations. Matched electrodes do not isolate architecture from these other pipeline changes.

## Frozen choices

- zero_calibration: `ensemble_b908c5a6cf69`, reused-development BA 66.98%.
  `{"arm": "zero_calibration", "input": "raw_constituent_probabilities", "kind": "mean", "members": ["neural_7cd5002f5f5a", "neural_cb9a3966fcb8", "neural_129423b51902", "power_1e0f4017a801", "power_2e699ed1b88c"]}`
- pre_cue: `ensemble_d6d2d766901d`, reused-development BA 66.16%.
  `{"arm": "pre_cue", "input": "raw_constituent_probabilities", "kind": "mean", "members": ["fbcsp_a18f7ce005af", "neural_e95c45c1089c", "power_235237cff6b7", "tangent_f39cd8b1fd34"]}`

## Two separately defined external evaluations

External dataset: BNCI2014_001 / BCI Competition IV-2a, all nine people and both sessions. Only left/right hand trials are retained, using the supplied class mapping 1→left and 2→right; feet/tongue are outside this binary task. The MATLAB trial index is 1-based trial start, with the imagery cue two seconds later. The 16 matching EEG channels are selected from the known 22-channel montage. EOG channels never enter a model. All finite left/right trials are retained, including flagged artifacts; flags are preserved for descriptive error analysis. This is not a reproduction of the original four-class competition or its causal scoring procedure.

1. **Direct transfer:** base models are fitted on all 104 PhysioNet development people; no BNCI task data or label fits a transform or model.
2. **Frozen-algorithm external LOSO:** the already-selected base algorithms refit on the other eight BNCI people. Both sessions of the held person are excluded. Hyperparameters, epochs, seeds, ensemble coefficients and temperature remain fixed from PhysioNet; BNCI scores never select a new pipeline. The training condition is different from direct transfer.

## Direct cross-dataset transfer

| Pipeline | People | Balanced accuracy | 95% participant interval | Accuracy | Log loss |
|---|---:|---:|---:|---:|---:|
| CSP reference | 9 | 66.63% | [57.52, 76.74]% | 66.63% | 0.6301 |
| EEGNet reference | 9 | 66.94% | [57.25, 76.81]% | 66.94% | 0.6434 |
| Selected: trial only | 9 | 73.69% | [64.58, 83.41]% | 73.69% | 0.5325 |
| Selected: pre-cue context | 9 | 71.84% | [64.47, 80.09]% | 71.84% | 0.5714 |

Actual external task trials: 2592; people: 9. Intervals resample people, not sessions, windows or seeds. LOSO intervals are conditional descriptions and do not account for every dependence from overlapping training cohorts.

## Frozen-algorithm external leave-one-person-out

| Pipeline | People | Balanced accuracy | 95% participant interval | Accuracy | Log loss |
|---|---:|---:|---:|---:|---:|
| CSP reference | 9 | 57.60% | [52.43, 63.81]% | 57.60% | 0.6840 |
| EEGNet reference | 9 | 67.59% | [58.26, 76.97]% | 67.59% | 0.6490 |
| Selected: trial only | 9 | 73.03% | [64.43, 82.33]% | 73.03% | 0.5405 |
| Selected: pre-cue context | 9 | 68.94% | [61.50, 77.31]% | 68.94% | 0.5922 |

Actual external task trials: 2592; people: 9. Intervals resample people, not sessions, windows or seeds. LOSO intervals are conditional descriptions and do not account for every dependence from overlapping training cohorts.

## Paired participant differences

| Protocol | Selected | Reference | Difference (pp) | 95% paired interval | Same input |
|---|---|---|---:|---:|---|
| external_transfer | selected_zero_calibration | csp_reference | +7.06 | [+4.40, +9.72] | True |
| external_transfer | selected_zero_calibration | eegnet_reference | +6.75 | [+0.81, +11.23] | True |
| external_transfer | selected_pre_cue | selected_zero_calibration | -1.85 | [-4.32, +0.77] | False |
| external_loso | selected_zero_calibration | csp_reference | +15.43 | [+8.76, +22.11] | True |
| external_loso | selected_zero_calibration | eegnet_reference | +5.44 | [+0.62, +9.38] | True |
| external_loso | selected_pre_cue | selected_zero_calibration | -4.09 | [-6.13, -1.93] | False |

## Source-only learning diagnostics

These scores are not held-out accuracy. Three filter-bank variants (task-only, context, and equivariant context) use a separate diagnostic setting: the SAME 32 balanced source examples from four source people, 150 epochs, dropout 0, learning rate .003.
- context=False, symmetry=none: 100.0% training accuracy, zero held-out examples.
- context=True, symmetry=none: 100.0% training accuracy, zero held-out examples.
- context=True, symmetry=equivariant: 100.0% training accuracy, zero held-out examples.

Ten within-source-person label permutations with participant-group CV average 50.29% BA. These controls test the learning path and loss of predictable signal under shuffling; they do not prove absence of every leakage.

Source-only negative controls: preceding-rest-only BA 53.49%; run/cue-ordinal-only BA 48.72%. The former is slightly above chance; these checks do not prove the absence of every possible confound. Neither external labels nor task EEG enter these control features.


Archived canonical task-input check: 104 people, 4680 epochs; maximum absolute difference 0 microvolts for the matched 16-electrode, 8–30 Hz late input. New pre-cue features are additional inputs, not a hidden change to the reference.


## Interpretation and provenance

The original 24-person confirmation has been reused as development here, so its new selection scores are not new independent evidence. The external protocols change both dataset and, in LOSO, training cohort. Do not compare their raw percentages to a previous PhysioNet test as if they were a paired accuracy gain. A paired interval crossing zero does not demonstrate a reliable positive gain. Context-versus-trial comparisons change the input requirement as well as the selected algorithm.

72 matched source-development comparisons retain adverse effects. These are exploratory, share overlapping source folds, and are not multiplicity-adjusted causal claims. Temperature scaling uses source OOF labels only and leaves the 0.5 decision boundary unchanged. Any remaining probability-quality problem must be reported rather than calibrated on the external holdout.

The contribution is a tested combination/hypothesis and a reproducible evaluation, not a claim that filter banks, ERD, mirroring or robust group weighting were invented here. The network is a compact independent FBCNet-inspired adaptation. Group weighting is inspired by Group DRO, with source participant groups and fixed eta; it is not a complete reproduction of the cited optimization studies. No degree, publication, SOTA, clinical or real-time-hardware claim follows from these experiments.

**Independent verification: PASS.** All final transfer and external LOSO fits were reinferred from real EEG. Saved OOF predictions, source trial/person identities, source-fitted temperatures and combination coefficients were checked. The archived frontier study is retained unchanged.

## Primary references

- FBCNet: https://arxiv.org/abs/2104.01233
- Channel- and label-flip augmentation: https://pubmed.ncbi.nlm.nih.gov/40039924/
- ERD/ERS principles: https://pubmed.ncbi.nlm.nih.gov/10576479/
- Group DRO: https://arxiv.org/abs/1911.08731
- Original external dataset description: https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf
- MATLAB loader/channel documentation: https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html
