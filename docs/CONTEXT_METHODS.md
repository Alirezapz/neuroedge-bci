# Pre-cue contrast and reflection: frozen external evaluation

This follows the completed frontier study. Its 24-person holdout has now been
inspected and is explicitly reused development data. No result on those people,
or on repeatedly inspected PhysioNet development folds, is called a new
independent confirmation. The archived study and its negative results remain.

## Hypotheses and attribution

Some old development examples have high within-person ranking AUC but strongly
biased class decisions. This suggests testing nuisance-offset reduction; it
does not prove its physiological cause. The new experiments test:

1. Active-only log-power versus pre-cue log-power contrasts, including spatial
   CSP projections and per-trial covariance whitening/tangent features.
2. No reflection, source channel-and-label reflection augmentation, and an
   explicit reflection-equivariant predictor. Hemispheric reflection swaps the
   fixed left/right electrode pairs and labels. Midline channels stay fixed.
3. A compact independent FBCNet-inspired network: six 4–40 Hz subbands, eight
   spatial filters/band, shared BatchNorm/SiLU, four temporal log-variance
   segments, dropout 0.3, and an optional shared pre-cue log-variance subtraction.
4. Source-participant risk weighting inspired by Group DRO, with eta 0.01,
   compared to the same ordinary-risk network. This is a defined stochastic
   adaptation, not an exact reproduction of a published training experiment.
5. Fixed source-OOF temperature scaling and combinations of neural and classical
   methods, including mixed active/context voting and stacking.

Filter banks, ERD/ERS, reflection augmentation, covariance whitening and Group
DRO have prior art. The research contribution being investigated is the
specified combination, symmetry constraint and matched evaluation. No novelty
or degree level is asserted merely because this combination is implemented.

Primary sources:

- FBCNet, Mane et al.: https://arxiv.org/abs/2104.01233
- Hoshino et al., channel-and-label flip (2024):
  https://pubmed.ncbi.nlm.nih.gov/40039924/
- Pfurtscheller and Lopes da Silva, ERD/ERS:
  https://pubmed.ncbi.nlm.nih.gov/10576479/
- Sagawa et al., Group DRO: https://arxiv.org/abs/1911.08731
- A further related mirror-metric learning paper is Luo et al. (2026),
  https://doi.org/10.1016/j.eswa.2026.132612 . This project does not implement
  or claim to have invented that paper's triplet-loss algorithm.

## Fixed data and input requirements

Source: PhysioNet EEGMMIDB people 1–80. Reused selection: the 24 structurally
valid people from 81–109 in the prior study. Previously documented exclusions
88,89,92,100,104 remain structural exclusions; no accuracy-based exclusion is
introduced. Runs 4/8/12 retain T1=left (0), T2=right (1). All 104 people are
development data for this sequential study.

All models access the SAME 16 sensorimotor electrodes, in the archived order.
Their average reference never accesses omitted sensors. Task windows are
[0.5,4) seconds after the imagery cue, sampled to 128 Hz; the archived references
use 8–30 Hz and [1,4) seconds. Eight fixed filters supply broad 4–40/8–30 Hz and
six subbands 4–8/8–12/12–16/16–22/22–30/30–40 Hz. Full and two half-window
covariances encode temporal changes for power/filter-bank CSP candidates.

The **zero_calibration** condition uses the task trial alone. The **pre_cue**
condition additionally uses [-1.75,-0.25) seconds from the same trial's
preceding rest. This is an extra input requirement, not an R01 calibration or
evaluation-batch estimate. PhysioNet context must lie entirely within an
annotated T0 segment. No future task label or target population statistic fits
the context representation.

Task filtering remains the archived offline forward/backward IIR. Applying that
filter across the cue to extract a baseline would let future task samples
affect the baseline. Therefore context instead uses two FORWARD causal
fourth-order Butterworth passes at native rate, initialized with the initial
sample's steady state. The squared magnitude matches the task filter, but its
phase differs. That causal filter runs on the preceding continuous recording;
reproduction must preserve its state/warmup rather than filtering a standalone
1.5-second raw excerpt. Its bounded 1.5-second segment is independently resampled with
polyphase resampling and line padding. No future sample enters that resampling.
A perturbation software check verifies this boundary. The complete task decoder
is still offline and cue aligned; no causal-device claim is made.

## Candidate budget and fitting

There are 63 frozen base configurations: 27 multiscale power, 12 filter-bank
CSP, 12 tangent, 11 neural and one archived CSP reference. The power classifiers
use logistic regression or RBF SVM; C is 0.1 or 1.0. CSP/tangent classifiers are
source-standardized logistic models. Every reflected training example gets the
opposite label. Equivariant classical inference averages the original
probabilities and the class-reversed reflected probabilities. The equivariant
network averages corresponding logits before softmax. Inference uses fixed
source-fitted transforms and is independent of other evaluation examples.

Three shuffled participant-group source folds use split seed 20261101. Neural
source fits use seed 7 and the fixed epoch policy, so no held-fold checkpoint
selection is required: new candidates use 35 epochs, the original EEGNet
reference uses 25. Advanced networks use AdamW at 0.001, weight decay 0.0001,
gradient norm clipping 5 and the documented max-norm constraints. The archived
reference uses Adam, its original architecture/scaler/BatchNorm and clipping 5.
Final advanced neural validation/refit predictions average seeds 7,17,27.

For every advanced base candidate, a POSITIVE scalar temperature is fitted to
the 80-person source OOF probabilities, minimizing log loss with log(T) in
[-3,3]. It preserves the 0.5 decision boundary and reflection law. Reference
temperatures remain 1. No external label adjusts probability calibration.

The shortlist includes source top-four per input condition, each family's best,
the two references, and every specified combination member. Reused selection
compares singles, top-2/3/5 means, family-diverse means, nonnegative voting and
stacking (C=.01/.1/1). Pre-cue combinations additionally mix their top three
with the task-only top three. Meta-classifiers and vote weights fit raw source
OOF probabilities, not selection or external labels. Vote fitting minimizes
OOF log loss plus .005 times squared weights under simplex constraints.
Selection maximizes participant-mean balanced accuracy, then minimizes log
loss, then stable name order. All rules are frozen before external acquisition.

## Two predeclared external protocols

The untouched dataset is BNCI2014_001 / BCI Competition IV-2a: all nine people,
both T/E sessions, left/right classes only. This is a binary offline research
evaluation, not the original four-class competition procedure. Original data
and source licensing: https://bnci-horizon-2020.eu/database/data-sets .
Task timing and electrode mapping are grounded in the original description:
https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf . The
official MATLAB-loader interpretation is also documented by MOABB:
https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html .

MATLAB trial indices are 1-based trial start; the cue is two seconds later.
Thus active windows are [2.5,6) seconds relative to trial start and context is
[.25,1.75). Native rate is 250 Hz, X is microvolts, class 1 is left and class 2
is right. The 16 matching EEG electrodes are selected from the first 22
channels; EOG never enters classification. All finite left/right trials are
retained, including artifact flags. Missing optional flags mean unknown rather
than evidence of clean EEG. Network/format failures stop execution, not a
performance-based cohort replacement. Expected binary count is 144/session,
288/person and 2592 total.

1. **Direct transfer:** fit base constituents on all 104 PhysioNet people,
   freeze them before external EEG acquisition, and predict all nine BNCI
   people. No BNCI task data or label fits any model or transform.
2. **Frozen-algorithm external LOSO:** refit those SAME selected constituents on
   the other eight BNCI people. Both sessions of the held person remain out.
   Epochs, seeds, architecture, temperature and combination coefficients are
   unchanged from source development. All nine folds are prespecified; no
   BNCI result reselects a pipeline. This condition uses labeled data from other
   BNCI people and is reported separately from direct transfer.

Both compare the selected task-only and pre-cue pipelines to matched 16-channel
CSP and EEGNet references on the SAME people. Participant bootstrap uses
10,000 draws, seed 43; paired differences resample within-person differences.
Sessions and seeds are never counted as independent people. These are
conditional descriptive intervals and do not include all dependence due to
overlapping LOSO training cohorts or repeated exploratory development.

## Integrity and interpretation

Data, methods, candidate grid, training people/trials, weights and predictions
have identity guards. Incomplete work can resume, but a scientific-source or
method change needs a new output name. Only software fixtures are synthetic.
Research metrics always come from actual downloaded EEG. Source fitted-model
receipts and histories remain; temporary screening/validation binaries may be
omitted. Final transfer and external LOSO artifacts remain for independent
reinference verification. No external score chooses a seed, threshold or
method. Failure to improve is retained as evidence, not hidden.
