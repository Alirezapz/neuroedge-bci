# Source-prior personal kernels

This archived supplementary study preserves the selected five-component source
ensemble and learns a nonlinear correction from a small personal support set.
The fixed primary condition is an equal spectral/geometry kernel with the first
16 chronological session-T left/right labels. All 144 session-E trials per
person remain out of fitting. No neural weight is fine-tuned.

Spectral features are relative log power in eight bands and three windows.
Geometry features are symmetric matrix-log coordinates of trace-normalized,
0.1-shrunk covariance, with sqrt(2) off-diagonal weights. Feature means and
standard deviations come from source people only. This standardized coordinate
distance is not the classic affine-invariant distance.

For source prediction p0, support S, residual coefficients alpha and intercept b:

\[
z(x)=\operatorname{logit}(p_0(x))+k(x,S)^T\alpha+b.
\]

Minimize balanced logistic loss plus

\[
\frac{\alpha^T(K+10^{-6}I)\alpha}{2nC}+\frac{b^2}{2n\,4^2}.
\]

Gaussian kernels use mean squared source-standardized feature distance. The
primary kernel averages the two kernels equally. A Cholesky coordinate change
and analytic gradient solve the regularized problem; a numerical-gradient
regression checks the implementation. It is not a Gaussian-process posterior
or a medical uncertainty estimate.

Three kernels and four C values give 12 source-development choices. First16
chronological run4/8 examples from 24 previously inspected selection people
supply support; run12 selects C. Their base model was trained on 80 other
people. Final feature scalers are refitted on all 104 PhysioNet people.

The mathematical specification preceded external acquisition. Implementation,
source scalers and C selection froze after raw acquisition but before the first
external scores. `specification.json`, `implementation.json` and
`source_selection.json` preserve that sequence. It is a sequential exploratory
study, not a publicly preregistered trial.

All 4/8/16 budgets are retained. Single-class support keeps the base model;
the person is neither discarded nor extended to find another class. The matched
comparison is against intercept calibration on the SAME E trials and labels.

See `runs/residual_personal/REPORT.md` and `results.html`,
`configs/residual_personal.yaml`, `personal_kernels.py`, `residual_study.py`
and `residual_verify.py`. `scripts/run_research.py` runs the five original
substudies in their registered order. Repeating the same nine people does not
create a new independent confirmation.
