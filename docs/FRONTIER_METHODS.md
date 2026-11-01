# Frontier study: wider search, calibration ablation and new confirmation

This extension raises the methodological and engineering scope without assigning
a degree level or assuming a high score. It is a sequential exploratory study;
there is no claim of a new algorithm, an exhaustive literature search, or a
state-of-the-art result. Earlier completed studies and negative results remain.

## Separation of participants and decisions

| Stage | Participants | Permitted use |
|---|---|---|
| Discovery | 1–60 | Three shuffled subject-group folds for candidate screening and source out-of-fold probabilities |
| Secondary validation | 61–80 | Compare a frozen screening shortlist and combination rules; select one pipeline per operating condition |
| Prospective confirmation | 81–109, subject to frozen structural QC | One evaluation after final membership, hyperparameters, seeds, epoch policy and ensemble rules are fixed |

Participants 71–80 have previously been inspected and are explicitly development
data now. The discovery group folds use seed 20261031. Only discovery participants
enter base-model screening or the fitting of ensemble weights/meta-classifiers.
Secondary-validation labels never fit a base model, scaler, tangent reference,
CSP filter, stacking coefficient or vote weight. They select among those fitted
pipelines. The large search can overfit development choices; only the new holdout
supports the final reported generalization result. The protocol is frozen
locally, not publicly preregistered.

Advanced neural epoch selection is nested within each discovery source fold:
20% of its source participants choose the epoch by validation log loss; that
model is discarded, then all source-fold participants are refitted for the
chosen epoch count before predicting the discovery held-out participants.
The median discovery-fold epoch count is fixed for secondary validation and
final refitting. Advanced neural validation/final predictions average seeds
7, 17, 27; the original 25-epoch EEGNet baseline retains seed 7. Selecting a
best final-test seed is prohibited.

## Two explicitly different operating conditions

The primary **zero-calibration** condition uses no participant-specific EEG
reference or fitted statistic from the new participant. Common average reference
and trial RMS are deterministic within-example transforms. Every population
quantity is learned from source participants only.

The secondary **rest-calibrated** condition additionally uses the new person's
separate R01 eyes-open resting recording, without task trials or labels.
Continuous resting EEG is independently filtered into each relevant band,
resampled, trimmed by one second at both ends, and reduced to a covariance.
Selected electrodes alone provide the common average reference. A fixed 5%
diagonal loading gives an SPD matrix, whose inverse square root whitens both
source and target task data relative to each person's own resting recording.
This adapts the Euclidean-alignment idea to a separate resting reference rather
than estimating a reference from the evaluation task batch. It does not use
other evaluation people, task labels or future task trials. Its extra resting
acquisition must be disclosed and its score must not replace a zero-calibration
score. Neither condition demonstrates causal/online decoding: filters are
zero-phase and windows are cue aligned.

Fifteen prespecified calibrated variants have a corresponding uncalibrated base
configuration. Their development predictions allow a matched representation
ablation, independently of which operating-condition winner is selected. Final
comparison between different winners changes both calibration and the selected
pipeline and is not an isolated alignment-effect estimate.

## Data and structural quality control

The task remains imagined left/right fist movement, EEGMMIDB v1.0.0 runs 4,8,12.
Each continuous recording is independently filtered with fourth-order
Butterworth IIR zero-phase filtering into 4–40, 8–30, 4–8, 8–12, 12–16, 16–22,
22–30 and 30–40 Hz bands. Resample to 128 Hz and extract [0.5,4) seconds;
the alternative late window is [1,4). All band filters precede epoching.
The 8–30 Hz late epochs were checked to match the archived original data
preparation byte for byte on all 900 available original-cohort epochs.

The fixed quality rule requires 64 expected sensors, finite signals, no
overlapping source annotations, task annotations of at least four seconds
(within one native sample), complete task windows, and at least two examples
of each class per required run. A failing recording excludes its whole
participant and records the reason. Network failures stop acquisition rather
than excluding people. Every discovery/development participant must pass;
confirmation must retain at least 20 people. The rule is fixed before acquiring
the new confirmation group and cannot exclude anyone by classifier performance.
Native sampling rates are read from each EDF and properly resampled, including
any files recorded at 128 Hz. R01 additionally requires at least 30 seconds,
matching sensors and finite signal; invalid resting references stop the
calibration preparation rather than silently changing its cohort.

Raw unreferenced covariance caches preserve electrode access boundaries.
Selecting a 16-electrode submatrix and applying its own reference is equivalent
to referencing those 16 original signals, without using the omitted sensors.
No artifact threshold or component is selected using confirmation accuracy.
All raw sources, byte hashes, structural audit decisions and derived-data hashes
are retained; raw EDF and large cache arrays are excluded from distribution.

## Specified model search

There are 224 unique base configurations in `candidates.json`:

- Regularized covariance-domain CSP with 4/8 components, shrinkage .01/.1/.3,
  two sensor counts, two windows, two broad bands and shrinkage LDA or RBF SVM.
- Six-band CSP, 2/4 components per band, trace normalization, optional training-only
  ANOVA feature selection, shrinkage LDA/logistic/RBF SVM.
- Single/multiband tangent-space models, log-Euclidean/Riemannian references,
  trial trace normalization on/off, train-fitted scaling and regularized logistic
  regression; selected 64-channel models include train-fitted PCA to 128 features.
- Minimum distance to Riemannian/log-Euclidean class means.
- Per-channel band power and relative band power with LDA, logistic regression,
  RBF SVM, Extra Trees and histogram gradient boosting.
- EEGNet, compact shallow log-power and temporal/spatial convolution plus
  two-layer attention adaptations; 16/64 channels; no augmentation, mixup, or
  same-class source-only segment recombination. The two additional architectures
  are independently implemented compact adaptations, not exact reproductions of
  the full published ShallowFBCSPNet or EEG Conformer. EEGNet uses source-fitted
  scaling and, for advanced candidates, trial-global RMS normalization and faster
  BatchNorm running-statistic updates. The archived original baseline settings
  remain a separate candidate and final reference.
- Fifteen separate-rest-calibrated counterparts of fixed classical/EEGNet
  configurations.

Parameters and code are frozen before model screening. SVM probability calibration
is fitted only within the source training data; its internal calibration folds
are trial folds, while the reported discovery folds are participant disjoint.
All tangent transforms use `tsupdate=False`. There is no evaluation-batch
normalization or tangent-reference refitting.

Advanced neural optimization uses AdamW at 0.001, weight decay 0.0001 and gradient
norm clipping at 5. The reference EEGNet keeps Adam, zero weight decay, dropout 0.5,
BatchNorm momentum 0.01, source-channel scaling and 25 epochs; the common new
training wrapper also clips its gradient norm at 5. Thus it is a fresh reference
configuration based on the archived settings, not a claim that its training
execution is byte-identical to the old engine. The original engine and its old
artifacts remain unchanged. The new EEGNet uses BatchNorm momentum 0.1; the
shallow and attention models are the compact adaptations defined in code.

The shortlist includes the six highest discovery scores per condition, the best
configuration from each family, a non-baseline neural candidate, and the fixed
original baselines. Secondary validation compares these source-fitted singles,
top-2/3/5 averages, family-diverse averages, nonnegative source-OOF weighted voting,
and logistic stacking with C=.01/.1/1. Vote weights minimize source-OOF log loss
plus a fixed small L2 penalty under simplex constraints. Stacking uses source-OOF
log odds. Secondary validation chooses the highest participant-mean balanced
accuracy, then lowest log loss, then stable name order. Combination coefficients
are frozen from discovery OOF data. Final constituent models refit all 80
development participants, while retaining the frozen combination rules.

## Confirmation, uncertainty and independent verification

The two selected operating-condition pipelines and the original CSP/EEGNet
baselines are evaluated on exactly the same structurally valid new participants.
Baselines use the original 16-channel, 8–30 Hz, [1,4) pipeline. Covariance-domain
CSP variants are independently implemented; the original CSP reference uses the
unchanged MNE implementation from the archived engine.

Report participant-mean balanced accuracy, ordinary accuracy and log loss;
95% percentile intervals resample people 10,000 times with seed 43. Compare
pipelines with paired participant differences using the same bootstrap. These
are conditional descriptive intervals, not a population-power guarantee or a
retraining bootstrap. Do not select or modify a method after seeing confirmation.

`frontier verify` recomputes all discovery/secondary-validation metrics from
their saved predictions; independently reruns every selected final constituent
and baseline on real EEG; checks trial identities, subject separation, model
hashes, frozen selection receipts, source-only neural scalers and individual
versus batch inference. Temporary feature fits, screening models and validation
constituent fits are reproducible but may be omitted from packaging; final
models, meta-classifiers, predictions and provenance remain.

References are in the generated report, `THIRD_PARTY.md` and `REFERENCES.bib`.
