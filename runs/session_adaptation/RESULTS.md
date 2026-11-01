# NeuroEdge: expanded cross-session calibration

The fixed primary hypothesis, anchored dual with 16 first-session labels, scores 74.15% balanced accuracy. Its paired difference versus same-budget intercept calibration is +1.08 percentage points [-0.46, +2.62]. This does not establish a reliable positive gain.

In the prespecified secondary 144-label condition, donor-selected fusion scores 85.11% [76.93, 92.52]%. It requires the complete first-session left/right support. It is the highest observed mean in this screen; that observation does not turn it into an independently selected confirmatory winner.

All values below use the SAME 1,296 session-E trials from nine people, sixteen electrodes and left/right classes. All 144 trials per person are retained. Labeled support is taken chronologically from the first session only. More labels change the calibration requirement.

This dataset and earlier results were previously inspected. The new study is exploratory, not an untouched external confirmation. E labels never choose the method, C, mixture weight or budget. All settings use only other people's session T. Nine-person bootstrap intervals are descriptive and are not adjusted for multiple secondary comparisons or all shared-donor dependencies.

## Matched comparison: balanced accuracy

| Method | 4 | 8 | 16 | 32 | 64 | 144 |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen source ensemble | 72.15% | 72.15% | 72.15% | 72.15% | 72.15% | 72.15% |
| Frozen source EEGNet | 66.13% | 66.13% | 66.13% | 66.13% | 66.13% | 66.13% |
| Source + intercept | 72.15% | 72.53% | 73.07% | 73.69% | 72.30% | 72.22% |
| Personal CSP + LDA | 59.41% | 64.20% | 65.74% | 67.82% | 71.30% | 73.92% |
| Personal filter-bank CSP + LDA | 52.39% | 60.34% | 65.90% | 71.06% | 77.39% | 82.25% |
| Personal spectral logistic | 53.32% | 59.65% | 66.67% | 68.52% | 73.46% | 78.55% |
| Personal tangent logistic | 59.65% | 60.73% | 68.44% | 73.23% | 78.78% | 82.48% |
| Personal tangent RBF SVM | 54.32% | 50.00% | 59.65% | 70.37% | 79.71% | 83.64% |
| Anchored tangent residual | 71.53% | 69.68% | 74.00% | 77.78% | 81.17% | 84.10% |
| Anchored dual residual (primary) | 70.45% | 71.53% | 74.15% | 76.31% | 78.32% | 83.49% |
| Donor-selected fusion | 71.91% | 70.76% | 74.15% | 75.31% | 79.48% | 85.11% |

Columns are the number of labeled session-T support trials. Frozen source rows are repeated controls and use no personal labels. All target-only and anchored methods have identical support identities within each column. The 144-label arm is a full first-session supervised condition, not a zero-calibration result.

## All method/budget metrics and paired intervals

| Method | Labels | Balanced accuracy | 95% participant interval | Difference vs intercept (pp) | 95% paired interval (pp) | Macro F1 | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anchored dual residual (primary) | 4 | 70.45% | [61.27, 80.79]% | -1.70 | [-3.86, +0.23] | 0.6925 | 0.5495 |
| Anchored dual residual (primary) | 8 | 71.53% | [62.81, 81.17]% | -1.00 | [-2.16, +0.08] | 0.7091 | 0.5358 |
| Anchored dual residual (primary) | 16 | 74.15% | [65.58, 83.41]% | +1.08 | [-0.46, +2.62] | 0.7384 | 0.5249 |
| Anchored dual residual (primary) | 32 | 76.31% | [67.44, 85.11]% | +2.62 | [-2.24, +7.87] | 0.7568 | 0.4944 |
| Anchored dual residual (primary) | 64 | 78.32% | [69.60, 87.11]% | +6.02 | [-0.08, +13.27] | 0.7765 | 0.4490 |
| Anchored dual residual (primary) | 144 | 83.49% | [75.39, 90.97]% | +11.27 | [+3.47, +20.68] | 0.8334 | 0.3674 |
| Anchored tangent residual | 4 | 71.53% | [62.81, 81.48]% | -0.62 | [-2.62, +1.00] | 0.7091 | 0.5427 |
| Anchored tangent residual | 8 | 69.68% | [59.72, 79.94]% | -2.85 | [-5.94, -0.08] | 0.6787 | 0.5359 |
| Anchored tangent residual | 16 | 74.00% | [64.81, 83.64]% | +0.93 | [-1.54, +3.70] | 0.7334 | 0.5201 |
| Anchored tangent residual | 32 | 77.78% | [69.14, 86.57]% | +4.09 | [+1.00, +8.02] | 0.7729 | 0.4841 |
| Anchored tangent residual | 64 | 81.17% | [73.61, 88.50]% | +8.87 | [+2.85, +15.43] | 0.8064 | 0.4281 |
| Anchored tangent residual | 144 | 84.10% | [77.47, 90.51]% | +11.88 | [+4.86, +19.98] | 0.8399 | 0.3760 |
| Personal CSP + LDA | 4 | 59.41% | [53.01, 66.90]% | -12.73 | [-21.60, -5.40] | 0.5362 | 10.8048 |
| Personal CSP + LDA | 8 | 64.20% | [56.94, 72.07]% | -8.33 | [-17.36, +0.00] | 0.5987 | 4.0677 |
| Personal CSP + LDA | 16 | 65.74% | [55.63, 76.39]% | -7.33 | [-11.88, -2.85] | 0.6358 | 1.3370 |
| Personal CSP + LDA | 32 | 67.82% | [58.56, 78.09]% | -5.86 | [-8.87, -2.55] | 0.6673 | 0.7596 |
| Personal CSP + LDA | 64 | 71.30% | [60.34, 82.33]% | -1.00 | [-6.48, +7.33] | 0.6975 | 0.5680 |
| Personal CSP + LDA | 144 | 73.92% | [63.19, 84.57]% | +1.70 | [-3.94, +12.04] | 0.7290 | 0.5162 |
| Donor-selected fusion | 4 | 71.91% | [63.66, 81.25]% | -0.23 | [-1.08, +0.85] | 0.7112 | 0.5554 |
| Donor-selected fusion | 8 | 70.76% | [61.73, 80.71]% | -1.77 | [-3.40, -0.39] | 0.7001 | 0.5464 |
| Donor-selected fusion | 16 | 74.15% | [65.66, 83.33]% | +1.08 | [-0.39, +2.55] | 0.7337 | 0.5336 |
| Donor-selected fusion | 32 | 75.31% | [67.13, 83.87]% | +1.62 | [-1.54, +5.40] | 0.7481 | 0.5176 |
| Donor-selected fusion | 64 | 79.48% | [71.68, 86.96]% | +7.18 | [+0.62, +14.81] | 0.7875 | 0.4809 |
| Donor-selected fusion | 144 | 85.11% | [76.93, 92.52]% | +12.89 | [+4.09, +23.15] | 0.8503 | 0.4188 |
| Personal filter-bank CSP + LDA | 4 | 52.39% | [48.77, 56.87]% | -19.75 | [-29.48, -11.19] | 0.4502 | 6.5555 |
| Personal filter-bank CSP + LDA | 8 | 60.34% | [53.86, 66.90]% | -12.19 | [-17.36, -7.33] | 0.5590 | 12.8259 |
| Personal filter-bank CSP + LDA | 16 | 65.90% | [60.57, 71.53]% | -7.18 | [-13.89, -0.31] | 0.6374 | 6.8899 |
| Personal filter-bank CSP + LDA | 32 | 71.06% | [64.12, 77.01]% | -2.62 | [-10.19, +5.32] | 0.7055 | 3.1132 |
| Personal filter-bank CSP + LDA | 64 | 77.39% | [69.06, 84.49]% | +5.09 | [-4.55, +16.21] | 0.7638 | 1.4125 |
| Personal filter-bank CSP + LDA | 144 | 82.25% | [73.99, 89.43]% | +10.03 | [+0.23, +21.53] | 0.8217 | 0.6631 |
| Source + intercept | 4 | 72.15% | [63.89, 81.64]% | +0.00 | [+0.00, +0.00] | 0.7133 | 0.5426 |
| Source + intercept | 8 | 72.53% | [64.43, 81.64]% | +0.00 | [+0.00, +0.00] | 0.7207 | 0.5372 |
| Source + intercept | 16 | 73.07% | [64.58, 82.48]% | +0.00 | [+0.00, +0.00] | 0.7280 | 0.5317 |
| Source + intercept | 32 | 73.69% | [64.58, 83.87]% | +0.00 | [+0.00, +0.00] | 0.7318 | 0.5319 |
| Source + intercept | 64 | 72.30% | [62.19, 83.33]% | +0.00 | [+0.00, +0.00] | 0.7185 | 0.5313 |
| Source + intercept | 144 | 72.22% | [61.65, 83.26]% | +0.00 | [+0.00, +0.00] | 0.7173 | 0.5317 |
| Frozen source EEGNet | 4 | 66.13% | [56.17, 77.01]% | -6.02 | [-10.42, -1.39] | 0.6309 | 0.6442 |
| Frozen source EEGNet | 8 | 66.13% | [56.17, 77.01]% | -6.40 | [-9.95, -3.24] | 0.6309 | 0.6442 |
| Frozen source EEGNet | 16 | 66.13% | [56.17, 77.01]% | -6.94 | [-9.95, -4.32] | 0.6309 | 0.6442 |
| Frozen source EEGNet | 32 | 66.13% | [56.17, 77.01]% | -7.56 | [-10.19, -5.25] | 0.6309 | 0.6442 |
| Frozen source EEGNet | 64 | 66.13% | [56.17, 77.01]% | -6.17 | [-9.03, -3.16] | 0.6309 | 0.6442 |
| Frozen source EEGNet | 144 | 66.13% | [56.17, 77.01]% | -6.10 | [-8.95, -2.62] | 0.6309 | 0.6442 |
| Frozen source ensemble | 4 | 72.15% | [62.04, 83.41]% | -0.00 | [-4.40, +5.63] | 0.7038 | 0.5426 |
| Frozen source ensemble | 8 | 72.15% | [62.04, 83.41]% | -0.39 | [-4.01, +3.55] | 0.7038 | 0.5426 |
| Frozen source ensemble | 16 | 72.15% | [62.04, 83.41]% | -0.93 | [-4.09, +1.55] | 0.7038 | 0.5426 |
| Frozen source ensemble | 32 | 72.15% | [62.04, 83.41]% | -1.54 | [-4.48, +0.31] | 0.7038 | 0.5426 |
| Frozen source ensemble | 64 | 72.15% | [62.04, 83.41]% | -0.15 | [-3.70, +2.85] | 0.7038 | 0.5426 |
| Frozen source ensemble | 144 | 72.15% | [62.04, 83.41]% | -0.08 | [-4.55, +3.78] | 0.7038 | 0.5426 |
| Personal spectral logistic | 4 | 53.32% | [46.22, 61.19]% | -18.83 | [-27.16, -11.65] | 0.4519 | 0.7423 |
| Personal spectral logistic | 8 | 59.65% | [53.24, 66.59]% | -12.89 | [-20.99, -5.40] | 0.5515 | 0.8329 |
| Personal spectral logistic | 16 | 66.67% | [58.49, 75.54]% | -6.40 | [-10.73, -1.31] | 0.6492 | 0.9326 |
| Personal spectral logistic | 32 | 68.52% | [59.95, 77.62]% | -5.17 | [-12.35, +3.94] | 0.6783 | 1.0264 |
| Personal spectral logistic | 64 | 73.46% | [64.51, 82.25]% | +1.16 | [-5.71, +10.19] | 0.7247 | 0.6241 |
| Personal spectral logistic | 144 | 78.55% | [70.06, 86.73]% | +6.33 | [-1.54, +15.97] | 0.7838 | 0.4936 |
| Personal tangent logistic | 4 | 59.65% | [51.70, 68.13]% | -12.50 | [-17.75, -6.94] | 0.5204 | 0.6963 |
| Personal tangent logistic | 8 | 60.73% | [51.39, 69.60]% | -11.81 | [-17.13, -6.10] | 0.5497 | 0.8543 |
| Personal tangent logistic | 16 | 68.44% | [58.41, 78.40]% | -4.63 | [-9.03, -0.54] | 0.6704 | 0.6460 |
| Personal tangent logistic | 32 | 73.23% | [65.35, 81.17]% | -0.46 | [-5.40, +5.02] | 0.7265 | 0.5418 |
| Personal tangent logistic | 64 | 78.78% | [69.83, 86.81]% | +6.48 | [-1.39, +15.35] | 0.7816 | 0.5301 |
| Personal tangent logistic | 144 | 82.48% | [74.00, 90.12]% | +10.26 | [+1.31, +20.83] | 0.8235 | 0.4054 |
| Personal tangent RBF SVM | 4 | 54.32% | [50.46, 59.18]% | -17.82 | [-26.08, -10.88] | 0.4372 | 0.7016 |
| Personal tangent RBF SVM | 8 | 50.00% | [50.00, 50.00]% | -22.53 | [-31.64, -14.43] | 0.3333 | 0.6910 |
| Personal tangent RBF SVM | 16 | 59.65% | [52.85, 67.82]% | -13.43 | [-20.37, -8.26] | 0.5255 | 0.6749 |
| Personal tangent RBF SVM | 32 | 70.37% | [61.50, 79.24]% | -3.32 | [-11.03, +4.40] | 0.6838 | 0.6545 |
| Personal tangent RBF SVM | 64 | 79.71% | [70.21, 87.96]% | +7.41 | [-1.24, +16.98] | 0.7938 | 0.6211 |
| Personal tangent RBF SVM | 144 | 83.64% | [74.61, 91.44]% | +11.42 | [+1.62, +22.69] | 0.8315 | 0.5890 |

## Fusion versus stronger personal controls at 144 labels

| Reference | Fusion difference (pp) | Descriptive paired interval (pp) |
|---|---:|---:|
| Personal filter-bank CSP + LDA | +2.85 | [+1.39, +4.48] |
| Personal tangent logistic | +2.62 | [+0.69, +4.32] |
| Personal tangent RBF SVM | +1.47 | [-0.23, +3.32] |
| Anchored tangent residual | +1.00 | [-1.31, +3.40] |
| Anchored dual residual (primary) | +1.62 | [+0.62, +2.62] |

The small differences against stronger personal controls do not establish reliable superiority when their intervals include zero. All 330 unordered method-pair/budget comparisons, including adverse results, are in `all_pairwise_comparisons.csv`; intervals are not multiplicity-adjusted.


## Selection, algorithms and limits

Each outer person is excluded from every donor model/parameter choice. For each of the other eight people, pseudo-support uses the first n T trials, and validation uses strictly later T runs. For the 144-label arm, selection deliberately emulates 96 labels because no later T run would remain. The outer person still receives exactly 144 support labels in the final fit. This selection-budget mismatch is a limitation.

Controls include CSP, filter-bank CSP, spectral logistic regression, tangent logistic regression and tangent RBF SVM. The source ensemble and EEGNet are the archived PhysioNet-only models. The proposed anchored residuals use support-local whitening and linear tangent/spectral kernels to correct that fixed source logit. Donor-selected fusion blends the intercept source with an equal mean of three personal decoders. These are project-specific combinations of established ideas, not a proven first-in-literature algorithm.

Tangent SVM uses sigmoid decision scores as uncalibrated probability surrogates; its log loss is reported with that limitation. Single-class supports retain the source for anchored adaptation and use the observed support prior for target-only models. Offline cue-aligned preprocessing does not establish real-time or clinical utility.

Exact equations, prior work, input requirements and selection rules: [SESSION_ADAPTATION_METHODS.md](../../docs/SESSION_ADAPTATION_METHODS.md).

## Historical conditions

Earlier direct transfer: selected ensemble 73.69%, EEGNet 66.94%, evaluated on both BNCI sessions (2,592 trials). Earlier external LOSO: 73.03% versus 67.59%, trained on the other eight people's two sessions. Earlier small-label calibration used the same E session as this new study; its source ensemble was 72.15%, intercept/16 was 73.07%, hybrid Gaussian kernel/16 was 72.99%, and geometry kernel/16 was 74.00%. These historical conditions have different training or test sets; do not subtract their headline percentages to claim a matched gain. The complete archived reports and all adverse results remain included.

## Reproduction and saved evidence

```bash
python scripts/restore_external_data.py
python -m neuroedge.session_study all --plan configs/session_adaptation.yaml
python -m pytest -q
```

The restoration command requires every original raw and processed input fingerprint to match. For a changed method or environment that changes bytes, copy the YAML to a new name and output directory; never remove identity guards or overwrite this study.

Donor development: 1350 fit/evaluation jobs; 51.2 seconds on this host. Final evaluation: 540 saved fit/evaluation artifacts plus 54 deterministic fusion results; 22.7 seconds. These counts include unchanged source-control evaluations and are not counts of distinct new algorithms.

Independent verification: **PASS**, 540 checkpoint replays and support refits, 54 donor-only parameter choices, 85,536 actual prediction rows.

- `donor_predictions.csv`: every C and donor validation trial.
- `donor_splits.json`: support/validation IDs and the 144-to-96 selection rule.
- `selection.json`: every outer-person C/weight and all donor scores.
- `fit_receipts.json` and `models/`: all personal support IDs and saved models.
- `predictions.csv`: all actual session-E predictions.
- `metrics_by_subject.csv`, `summary.csv`, `paired_differences.csv`: all participant results and uncertainty.
- `all_pairwise_comparisons.csv`: all 330 method-pair/budget comparisons, including stronger controls.
- `protocol.json`, `environment.json`, `verification.json`: exact evidence identities.
