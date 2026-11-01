# NeuroEdge frontier study

A bounded, broad sequential exploratory search, followed by one new participant holdout. This is not an exhaustive search of every EEG method or a claim of a new algorithm.

Discovery: 60 participants, 3-fold participant-group cross-validation. Secondary validation: 20 different participants. Previously inspected participants 71–80 are explicitly development data now. Only the secondary validation chooses the final pipelines; confirmation does not choose a model, seed, weight or epoch.

Base configurations completed: 224. Secondary validation configurations, including combinations: 40. Confirmation requested 29 participants; 24 pass the frozen structural-data QC rule.

## Operating conditions

The primary zero-calibration arm uses no EEG or statistic from a new participant before that task trial. The secondary rest-calibrated arm additionally requires that participant's separate R01 eyes-open baseline. Its covariance whitening uses no task/test trial or label, but its extra calibration requirement must be reported. All other learned population transforms, CSP, tangent references, scaling, classifiers and combination weights use source participants only.

## Frozen development selections

- zero_calibration: `neural_9f9df95c770c`; validation balanced accuracy 69.49%; kind single.
  eegnet; 64 electrodes; 4–40 Hz; [0.5,4) s; augmentation=none; fixed epochs=20; seeds=[7, 17, 27].
- rest_calibrated: `ensemble_e072ea230711`; validation balanced accuracy 69.37%; kind ensemble.
  Members: neural_rest_274806d85130, tangent_rest_2ad33320ab99, csp_rest_48b836864369, fbcsp_rest_37498e7ddc70, power_rest_951ddd75361c; rule: weighted.
  Frozen source-OOF vote weights (same member order): 0.66785701, 0.17076045, 7.0593952e-18, 2.1786124e-17, 0.16138254.
  `neural_rest_274806d85130`: eegnet; 16 electrodes; 4–40 Hz; [0.5,4) s; augmentation=none; fixed epochs=27; seeds=[7, 17, 27].
  `tangent_rest_2ad33320ab99`: tangent; 16 electrodes; 8–30 Hz; [1,4) s; classifier=logistic; metric=logeuclid; normalize=True.
  `csp_rest_48b836864369`: csp; 16 electrodes; 8–30 Hz; [1,4) s; classifier=lda; components=4; reg=0.1; normalize=False.
  `fbcsp_rest_37498e7ddc70`: fbcsp; 16 electrodes; 4–8, 8–12, 12–16, 16–22, 22–30, 30–40 Hz; [1,4) s; classifier=logistic; components=4; reg=0.1; normalize=True; select=None.
  `power_rest_951ddd75361c`: power; 64 electrodes; 4–8, 8–12, 12–16, 16–22, 22–30, 30–40 Hz; [1,4) s; classifier=logistic.
  The complete pipeline requires access to 64 electrodes, including its separate resting reference.

## Prospective confirmation

All final models are refitted using the 80 development participants before confirmation data acquisition. Advanced neural candidates average the three fixed training seeds; the original EEGNet baseline retains its fixed seed and 25 epochs. Intervals below bootstrap participants, not windows; they are conditional descriptive intervals, not guarantees.

| Pipeline | Condition | Participants | Balanced accuracy | 95% participant interval | Accuracy | Log loss |
|---|---|---:|---:|---:|---:|---:|
| csp16_baseline | zero_calibration | 24 | 57.48% | [53.16, 62.36]% | 57.22% | 0.7039 |
| eegnet16_baseline | zero_calibration | 24 | 60.05% | [55.17, 65.35]% | 59.91% | 1.0145 |
| selected_zero_calibration | zero_calibration | 24 | 63.74% | [59.53, 68.38]% | 63.61% | 0.8294 |
| selected_rest_calibrated | rest_calibrated | 24 | 68.72% | [63.41, 74.34]% | 68.61% | 0.5964 |

The selected zero-calibration pipeline has log loss 0.8294, worse than 0.6931 for constant 0.5 probabilities, despite better-than-chance decisions. Its probability confidence is therefore a remaining limitation under log loss. No probability calibration is fitted using confirmation data.

## Paired participant differences

| Compared pipeline | Reference | Difference (percentage points) | 95% paired interval | Same calibration requirement |
|---|---|---:|---:|---|
| selected_zero_calibration | csp16_baseline | +6.26 | [+2.74, +10.00] | True |
| selected_zero_calibration | eegnet16_baseline | +3.68 | [-1.25, +8.13] | True |
| selected_rest_calibrated | selected_zero_calibration | +4.98 | [+2.36, +7.86] | False |

An interval crossing zero does not establish a reliable positive gain. The selected zero-calibration pipeline also changes sensor count, window and preprocessing relative to the references; its comparison is a complete-pipeline comparison, not an isolated architecture effect. The rest-versus-zero comparison changes the calibration requirement as well as the selected pipeline; it cannot be interpreted as an isolated architectural improvement.

## Structural exclusions

- S088: Overlapping source annotations.
- S089: Overlapping source annotations.
- S092: Overlapping source annotations.
- S100: Overlapping source annotations.
- S104: Task annotation shorter than 4.0s.

## Matched development alignment ablation

All fifteen rest-calibrated configurations are paired with the same uncalibrated hyperparameters on the same discovery group folds. This is development evidence and its intervals do not include all dependence from overlapping source folds. No configuration is omitted for a negative effect. See rest_alignment_development_ablation.csv.

| Family | Uncalibrated configuration | Base BA | Rest-calibrated BA | Difference (pp) |
|---|---|---:|---:|---:|
| csp | csp_20861754d5f1 | 57.94% | 59.43% | +1.50 |
| csp | csp_e98f105d056a | 59.35% | 61.41% | +2.06 |
| csp | csp_f61e99bbafbf | 55.26% | 53.62% | -1.65 |
| csp | csp_932116692d4f | 58.27% | 54.18% | -4.08 |
| fbcsp | fbcsp_a7c4860f3798 | 60.31% | 60.37% | +0.07 |
| fbcsp | fbcsp_c9bf007def5b | 58.56% | 59.83% | +1.27 |
| tangent | tangent_4bc54895eb46 | 61.36% | 64.06% | +2.70 |
| tangent | tangent_7885956645eb | 61.35% | 63.70% | +2.35 |
| tangent | tangent_942bc3c6daba | 60.74% | 60.78% | +0.04 |
| tangent | tangent_6ff26f8e7431 | 60.81% | 60.78% | -0.03 |
| fbtangent | fbtangent_312607a3456b | 59.00% | 57.67% | -1.33 |
| fbtangent | fbtangent_dec69063d71e | 57.91% | 57.73% | -0.18 |
| power | power_7b64baeb8d5d | 57.28% | 59.09% | +1.81 |
| power | power_946cbffa3c62 | 57.29% | 60.34% | +3.05 |
| neural | neural_f25b86d6911d | 64.14% | 66.75% | +2.60 |

An additional 262 matched development comparisons isolate specified channel, window, band, normalization, augmentation or shrinkage changes while keeping the other candidate fields fixed. All rows, including adverse effects, are retained in matched_development_ablations.csv. These exploratory comparisons do not drive a new confirmation selection or establish a multiplicity-adjusted effect.

## Source-only learning controls

These controls do not change selection and are not held-out performance results. With dropout disabled and a separate diagnostic learning setting, each neural architecture is trained for 120 epochs on 32 balanced real discovery examples.

- eegnet: 100.0% accuracy on the same 32 training examples; no held-out examples.
- shallow: 100.0% accuracy on the same 32 training examples; no held-out examples.
- attention: 100.0% accuracy on the same 32 training examples; no held-out examples.

Ten within-participant label-shuffle controls, retaining the subject-group split, average 50.27% balanced accuracy (range 49.39–51.21%). This is consistent with a working learning path and loss of predictable signal after label permutation; it does not itself establish generalization or prove that every possible leakage is absent.

Exclusions follow the frozen whole-participant structural rule, before any confirmation model score. No participant is dropped for low accuracy. The task and cohort differ from public within-person benchmarks; published accuracy numbers cannot be substituted as matched baselines.

## Search and verification scope

Classical methods cover regularized CSP/LDA, CSP/RBF-SVM, six-band CSP with train-only feature selection, single/multiband Riemannian tangent logistic models, minimum-distance-to-mean, spectral power with linear, RBF and tree classifiers. Neural methods include EEGNet, independent compact shallow log-power and convolution/attention adaptations, two sensor sets, mixup and same-class segment recombination. They are not exact reproductions of every cited network. Mean voting, diverse mean voting, nonnegative source-OOF-weighted voting and source-OOF logistic stacking are evaluated on secondary validation. Neural source-only inner participant validation chooses epoch counts; a full source refit then makes every discovery OOF prediction. Final epoch policies and seeds are fixed before confirmation. Failed or negative methods remain in the screen table.

Independent verification: **PASS**. All screening and validation metrics are recomputed from saved predictions. Final selected constituents and baselines are reinferred from real EEG; scalers are recomputed from source data, and individual/batch predictions are compared.

These are offline cue-aligned experiments. No causal online, clinical, real-time hardware, MCU, energy, state-of-the-art or degree-level novelty claim follows from a larger model search. More test-informed iteration requires another untouched dataset/cohort.

## Reproduction

`python -m neuroedge.frontier all --plan configs/frontier.yaml`

For a new host, first create a fresh plan/output name with scripts/new_run.py. Raw EDF, large covariance caches and temporary screening feature/model caches are omitted from the archive; final models, predictions, audit receipts and exact source hashes remain. Reacquire data to rerun inference verification.

## Method references

- EEGMMIDB: https://physionet.org/content/eegmmidb/1.0.0/
- EEGNet (Lawhern et al.): https://arxiv.org/abs/1611.08024
- Shallow ConvNet ideas (Schirrmeister et al.): https://arxiv.org/abs/1703.05051
- EEG Conformer ideas (Song et al.): https://github.com/eeyhsong/EEG-Conformer
- Euclidean alignment (He and Wu): https://arxiv.org/abs/1808.05464
- Train-fitted tangent mapping: https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html
