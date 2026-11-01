# Session-disjoint calibration expansion

This is an exploratory continuation on BNCI2014_001, whose earlier external
results have already been inspected. It is not a new untouched-dataset
confirmation. The configuration, candidate list, scientific code and parent
prediction hash are frozen locally before the new session-E evaluation.

## Question and supervision

Can support-local covariance geometry and a source-logit residual improve
cross-session decoding at the same label budget? How does performance change
when calibration increases from 4/8/16 to 32/64/144 labeled first-session trials?
The primary comparison is **anchored dual versus intercept-only at 16 labels**.
Every other method and budget is a reported secondary comparison. Results at
144 labels must not be presented as zero-calibration accuracy.

All nine people, all 16 fixed sensorimotor electrodes and all 144 left/right
session-E trials per person are retained. Artifact flags are descriptive only.
The first n chronological session-T left/right trials form the support set.
No E labels, E population statistics, E batch alignment, E checkpoint selection
or E parameter selection are permitted. The original offline filter and fixed
0.5–4 s task window are retained. No pre-cue EEG enters these new methods.

## Parameter selection and isolation

For an outer person, C and the fusion weight are selected using the other eight
people's T data only. Each donor's first n T trials emulate calibration; validation
uses only complete runs strictly later than the support's last run. For the
144-label arm, selection uses a 96-label pseudo-support with the remaining later
runs as validation. This deliberate mismatch is reported. The final fit still
uses exactly the outer person's specified budget. Target T trials beyond that
budget cannot affect its settings. Parameters maximize donor-person mean balanced
accuracy, then minimize mean log loss, then prefer smaller C/weight. The same
144-label budget is not selected after looking at the results.

## References and proposed combinations

Established controls are CSP+shrinkage LDA, filter-bank CSP+shrinkage LDA,
filter-bank tangent logistic regression, tangent RBF SVM, and spectral logistic
regression. The fixed pretrained ensemble and EEGNet predictions are reused as
source baselines, alongside same-budget source-intercept adaptation.

For each support-local tangent transform, shrink each covariance by 0.1 toward
its trace-scaled identity, normalize trace to 16, whiten with the arithmetic
mean of support covariances, take the matrix logarithm, and vectorize the upper
triangle with sqrt(2) off-diagonal weights. Use the six non-overlapping bands
4–8, 8–12, 12–16, 16–22, 22–30 and 30–40 Hz. Spectral features are log channel
powers across all eight bands and the full/two half windows, with each window's
global power removed. Feature mean/variance are fitted on support only; each
feature variance is shrunk halfway toward the block's mean variance.

**Anchored tangent residual** fits a balanced logistic residual around the fixed
source logit using the support-local tangent linear kernel K = X X' / d.
**Anchored dual residual** uses the equal mean of the tangent and spectral linear
kernels. Both minimize balanced logistic loss plus alpha' K alpha/(2 n C)
and an intercept Gaussian penalty b^2/(2 n 16); fixed 1e-6 jitter makes the
kernel solve stable. This uses the existing independently tested residual
optimizer with a new support-local feature map and linear kernel.

**Donor-selected fusion** combines intercept-adapted source with an equal mean
of FBCSP, tangent RBF SVM and anchored dual probabilities; the mixture weight
is selected on donor T validation. SVM scores use sigmoid(decision_function)
as explicitly uncalibrated probability surrogates; this preserves its decision
boundary and does not claim calibrated uncertainty.

These are proposed combinations implemented for this project. Covariance
alignment, tangent classification, filter banks, residual adaptation and
ensembling have prior art; no first-in-literature invention is asserted.
See Bleuze, Mattout and Congedo (2022), *Tangent space alignment*,
https://doi.org/10.3389/fnhum.2022.1049985, and the existing references.

## Evidence and uncertainty

Save every donor candidate prediction, final model, support identity, test
prediction, paired participant difference and source/data/code hash. Verify
all final predictions by independently loading the checkpoints and refitting
from support data; reconstruct parameter selection from donor predictions.
Bootstrap whole people, not trials or seeds. Intervals are descriptive for
nine people, overlap in donor selection and many secondary comparisons; they
are not multiplicity-adjusted confirmatory intervals. Preserve negative results.
No clinical, real-time or hardware accuracy claim follows from this study.
