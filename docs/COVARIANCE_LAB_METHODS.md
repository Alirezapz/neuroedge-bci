# Expanded covariance and donor-stack study

The final configuration is `configs/covariance_lab_corrected.yaml`, with output
`runs/covariance_lab_corrected`. The initial `covariance_lab` run is invalidated:
an SPD-only eigenvalue floor was incorrectly applied to a symmetric matrix
logarithm before exponentiation. Its original source and screen are preserved
for audit. The corrected study refits every donor and selected final model;
the initial partial E output is not scientific evidence. A constant-SPD
regression preserves negative logarithm eigenvalues and checks the correction.

This is a new, bounded exploratory study on the previously inspected
BNCI 2014_001 dataset. It is not untouched confirmation. Source code, candidates,
selection and operating conditions are frozen before the new session-E scores.
The fixed primary condition is donor stacking, 16 sensors, strict prediction,
144 chronological session-T labels. Budgets 16/64 and 22 sensors are secondary.

## Data and operating conditions

Retain the same nine people and every 144 left/right E trial per person. The
16-sensor input is the exact previous covariance cache. The 22-sensor condition
uses all EEG electrodes, never the three EOG electrodes, with its OWN common
average reference. Both use the previous eight bands and fixed 0.5–4 s offline
task window, represented by full/two half-window covariance. All 18 raw MATLAB
files must retain their original SHA 256 identities.

Strict prediction estimates every transform from T support only. The separately
reported **batch-unlabeled** condition may use the mean covariance of the entire
unlabeled prediction batch. It transports those matrices to the support mean
by S^(1/2) B^(-1/2) C B^(-1/2) S^(1/2), with fixed 0.01 trace shrinkage. It uses
future unlabeled trials and cannot be presented as independent-trial, causal,
zero-calibration or strict performance. E labels never enter this operation.

## Selection and proposed combinations

All candidate parameter choices use T support/later-run validation from other
people only. For an outer person, exclude ALL its T/E data from selection.
For budget 144, donor pseudo-support is 96 so later T runs remain for validation;
the final personal model still receives 144 labels. No parameter, sensor set,
budget or input condition is selected using E performance.

Compare shrinkage FBCSP+LDA, FBCSP RBF SVM, training-only Fisher-selected FBCSP
SVM, tangent logistic, tangent RBF SVM, joint CSP/tangent SVM, multi-order
covariance-prototype logistic, and source-anchored multiview residuals. The
candidate JSON is authoritative for every parameter combination. SVM scores
use sigmoid(decision_function), an uncalibrated probability surrogate.

Covariances are trace-normalized with declared diagonal shrinkage. CSP uses
class-mean generalized eigenvectors and selected extreme components. Tangent
features whiten at the support arithmetic mean, apply matrix logarithms and
sqrt(2)-weighted symmetric vectorization. Feature variance is shrunk 0.1 toward
its block mean. Joint SVM concatenates block-normalized CSP and tangent views.
Multi-order prototypes use harmonic, log-Euclidean and arithmetic class means;
log-Euclidean distances become support-standardized logistic features. This is
a simplified power-mean feature implementation, not an exact reproduction of
the published affine-invariant Mean Field classifier.

The proposed multiview residual preserves the fixed source logit and uses the
equal mean of a tangent Gaussian kernel and a CSP linear kernel. The optimizer
and source model are unchanged. The proposed donor stack learns a logistic
combination of family scores. To construct meta-training data for donor d,
select each family's setting using donors other than d and the outer person.
The final family settings use all eight allowed donors. Choose stack C with
leave-one-donor-out meta-validation. Equal probability and geometric/logit
means are explicit combination controls. These are combinations of prior ideas,
not a claim of worldwide algorithmic invention.

## Interpretation and evidence

Every candidate donor prediction, selected setting, nested meta identity,
support ID, final model and actual E prediction is retained. Independent
verification replays every checkpoint, reconstructs selections/stack inputs and
recomputes participant metrics. Preserve all negative results. Compare with
previous fusion and tangent SVM on IDENTICAL E trials, budgets and 16 sensors;
22-sensor and batch-aligned rows have additional requirements. Bootstrap whole
people; intervals are descriptive and unadjusted for secondary comparisons.

Prior work: [Ang et al., FBCSP](https://doi.org/10.3389/fnins.2012.00039),
[Mean Field documentation and primary references](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.classification.MeanField.html),
and [Bleuze et al., tangent alignment](https://doi.org/10.3389/fnhum.2022.1049985).
