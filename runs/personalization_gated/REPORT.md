# Session-disjoint personal intercept calibration

A supplementary protocol registered before external acquisition. This requires labeled examples from the new person and is a different input/training condition from zero-calibration decoding. No network or feature weight is fine-tuned.

The first 16 chronological left/right trials of session T are the prespecified primary support set. All 144 session-E trials per person remain out of the adaptation fit. The SAME test trials are scored with and without calibration. Budgets 4/8/16 are all reported; no external score chooses the budget, prior or source algorithm.

Only one log-odds intercept is fitted with a class-balanced likelihood and a fixed zero-mean Gaussian prior (standard deviation 4). If the chronological support contains just one class, keep the source model without looking ahead or excluding the person. Source pipelines/ensembles are inherited from the parent PhysioNet-only selection.

| Pipeline | Session-T labeled support | Session-E balanced accuracy | 95% participant interval | Accuracy |
|---|---:|---:|---:|---:|
| Fixed population gate | 0 | 71.45% | [62.58, 80.94]% | 71.45% |
| Fixed population gate | 16 | 74.15% | [67.21, 81.94]% | 74.15% |
| Trial quality gate | 0 | 67.67% | [59.65, 76.77]% | 67.67% |
| Trial quality gate | 16 | 69.14% | [60.88, 77.55]% | 69.14% |

## Paired calibration effect

- Fixed population gate: +2.70 percentage points; 95% paired interval [-0.08, +5.86].
- Trial quality gate: +1.47 percentage points; 95% paired interval [-1.31, +4.32].

A paired interval crossing zero does not establish a reliable positive calibration gain. These nine-person conditional intervals do not include support-set resampling or every source-training uncertainty. The public dataset supplies the labeled support; this is a recorded-data calibration simulation, not a live user study. The adaptation is an existing type of calibration idea being tested for the observed offset hypothesis, not a new general algorithm.

Independent verification: **PASS**. First-session chronology, test-session separation, all budgets, intercept fits, predictions and metrics were recomputed.
