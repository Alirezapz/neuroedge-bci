# Session-disjoint personal intercept calibration

A supplementary protocol registered before external acquisition. This requires labeled examples from the new person and is a different input/training condition from zero-calibration decoding. No network or feature weight is fine-tuned.

The first 16 chronological left/right trials of session T are the prespecified primary support set. All 144 session-E trials per person remain out of the adaptation fit. The SAME test trials are scored with and without calibration. Budgets 4/8/16 are all reported; no external score chooses the budget, prior or source algorithm.

Only one log-odds intercept is fitted with a class-balanced likelihood and a fixed zero-mean Gaussian prior (standard deviation 4). If the chronological support contains just one class, keep the source model without looking ahead or excluding the person. Source pipelines/ensembles are inherited from the parent PhysioNet-only selection.

| Pipeline | Session-T labeled support | Session-E balanced accuracy | 95% participant interval | Accuracy |
|---|---:|---:|---:|---:|
| CSP reference | 0 | 66.74% | [57.48, 77.16]% | 66.74% |
| CSP reference | 16 | 68.36% | [59.88, 77.16]% | 68.36% |
| EEGNet reference | 0 | 66.13% | [56.17, 77.01]% | 66.13% |
| EEGNet reference | 16 | 67.05% | [57.95, 76.62]% | 67.05% |
| Selected: pre-cue context | 0 | 71.60% | [62.65, 80.86]% | 71.60% |
| Selected: pre-cue context | 16 | 72.15% | [64.35, 80.17]% | 72.15% |
| Selected: trial only | 0 | 72.15% | [62.04, 83.41]% | 72.15% |
| Selected: trial only | 16 | 73.07% | [64.58, 82.48]% | 73.07% |

## Paired calibration effect

- CSP reference: +1.62 percentage points; 95% paired interval [-2.01, +6.25].
- EEGNet reference: +0.93 percentage points; 95% paired interval [-2.31, +4.63].
- Selected: pre-cue context: +0.54 percentage points; 95% paired interval [-2.47, +3.78].
- Selected: trial only: +0.93 percentage points; 95% paired interval [-1.55, +4.09].

A paired interval crossing zero does not establish a reliable positive calibration gain. These nine-person conditional intervals do not include support-set resampling or every source-training uncertainty. The public dataset supplies the labeled support; this is a recorded-data calibration simulation, not a live user study. The adaptation is an existing type of calibration idea being tested for the observed offset hypothesis, not a new general algorithm.

Independent verification: **PASS**. First-session chronology, test-session separation, all budgets, intercept fits, predictions and metrics were recomputed.
