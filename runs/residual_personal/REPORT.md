# Nonlinear personal residual calibration

Primary hybrid, 16 labeled support trials: BA 72.99%, compared with 73.07% for intercept-only calibration on the SAME session-E trials. Paired difference -0.08 pp [-0.23, +0.00].

This is a supplementary sequential source-development hypothesis. Its mathematical specification was recorded before external acquisition. Implementation and source-only C selection were frozen after raw acquisition but before the first external transfer scores. The raw external EEG or its labels never fit a population transform or select C, kernel, budget or primary hypothesis.

All methods inherit the SAME source-selected task-only five-component ensemble. They use the same chronological session-T support and all 144 session-E test trials per person. No deep weight is fine-tuned. Unlike zero-calibration decoding, these methods require personal labels.

## Method

The source ensemble logit is retained as a prior. A regularized support-kernel residual and one intercept correct it. Spectral features are relative log variances in full/two half windows across eight bands and sixteen sensors. Geometry features are upper-triangle vectors of the matrix logarithm of trace-normalized, 0.1-shrunk full-window covariance. Off-diagonal entries receive sqrt(2) weighting. Feature means/std come from source participants only. Gaussian kernels use mean squared standardized feature distance; the primary hybrid is the equal mean of the two kernels.

Population scaling uses people 1–80 during source selection, then all 104 development people for external evaluation. Each kernel chooses C from [0.01,0.1,1,10] using first16 chronological source run4/8 labels and held-out run12 trials of the previously inspected 24 development people. Those scores are development evidence. All 12 choices are retained.

For new people, support is the first chronological left/right session-T trials. Budgets4/8/16 are all reported; 16 and the hybrid kernel were fixed as primary. If support contains only one class, keep the source decoder. All features use task EEG only and the same sixteen available electrodes. The complete signal pipeline is offline.

| Method | Source-selected C | Source-development run12 BA |
|---|---:|---:|
| Spectral kernel | 10 | 64.43% |
| Covariance-geometry kernel | 10 | 64.84% |
| Equal kernel fusion (primary) | 0.01 | 64.40% |

## All external conditions

| Method | Session-T support | Session-E BA | 95% participant interval | Log loss |
|---|---:|---:|---:|---:|
| Intercept only | 0 | 72.15% | [62.04, 83.41]% | 0.5426 |
| Intercept only | 4 | 72.15% | [63.89, 81.64]% | 0.5426 |
| Intercept only | 8 | 72.53% | [64.43, 81.64]% | 0.5372 |
| Intercept only | 16 | 73.07% | [64.58, 82.48]% | 0.5317 |
| Covariance-geometry kernel | 0 | 72.15% | [62.04, 83.41]% | 0.5426 |
| Covariance-geometry kernel | 4 | 70.83% | [60.88, 81.40]% | 0.5442 |
| Covariance-geometry kernel | 8 | 71.53% | [62.50, 81.25]% | 0.5309 |
| Covariance-geometry kernel | 16 | 74.00% | [65.28, 83.26]% | 0.5169 |
| Equal kernel fusion (primary) | 0 | 72.15% | [62.04, 83.41]% | 0.5426 |
| Equal kernel fusion (primary) | 4 | 72.15% | [63.89, 81.64]% | 0.5426 |
| Equal kernel fusion (primary) | 8 | 72.53% | [64.43, 81.64]% | 0.5372 |
| Equal kernel fusion (primary) | 16 | 72.99% | [64.43, 82.41]% | 0.5317 |
| Spectral kernel | 0 | 72.15% | [62.04, 83.41]% | 0.5426 |
| Spectral kernel | 4 | 69.06% | [58.80, 80.09]% | 0.5560 |
| Spectral kernel | 8 | 69.52% | [60.34, 79.63]% | 0.5397 |
| Spectral kernel | 16 | 73.77% | [65.28, 83.10]% | 0.5149 |

## Paired differences

| Method | Reference | Difference (pp) | 95% paired interval |
|---|---|---:|---:|
| Spectral kernel | same_budget_bias | +0.69 | [-1.47, +3.16] |
| Spectral kernel | unadapted_same_session | +1.62 | [-1.16, +4.48] |
| Covariance-geometry kernel | same_budget_bias | +0.93 | [-1.00, +3.01] |
| Covariance-geometry kernel | unadapted_same_session | +1.85 | [-1.54, +5.71] |
| Equal kernel fusion (primary) | same_budget_bias | -0.08 | [-0.23, +0.00] |
| Equal kernel fusion (primary) | unadapted_same_session | +0.85 | [-1.62, +4.01] |

Intervals resample people, not trials or seeds. They are descriptive, conditional on these fitted models and not adjusted for all secondary comparisons. An interval crossing zero does not establish a reliable positive improvement. Kernels, covariance logarithms and logistic regularization are established tools; this experiment does not prove worldwide algorithmic novelty, SOTA, clinical use or a degree level.

Independent verification checks all12 source choices, both source-only scalers, chronological support identities and all external method/budget/person predictions.
