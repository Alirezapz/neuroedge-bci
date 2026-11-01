# NeuroEdge research continuation

65 base configurations, 195 participant-group source fits. All previously inspected PhysioNet people are development data. Core-model source selection preceded external acquisition. All methods use 16 EEG channels. Pre-cue context and labeled personal calibration have extra requirements.

## external_transfer

| Model | Balanced accuracy | 95% participant interval | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| csp_reference | 66.63% | [57.52, 76.74]% | 66.63% | 0.6301 |
| eegnet_reference | 66.94% | [57.25, 76.81]% | 66.94% | 0.6434 |
| selected_zero_calibration | 73.69% | [64.58, 83.41]% | 73.69% | 0.5325 |
| selected_pre_cue | 71.84% | [64.47, 80.09]% | 71.84% | 0.5714 |

## external_loso

| Model | Balanced accuracy | 95% participant interval | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| csp_reference | 57.60% | [52.43, 63.81]% | 57.60% | 0.6840 |
| eegnet_reference | 67.59% | [58.26, 76.97]% | 67.59% | 0.6490 |
| selected_zero_calibration | 73.03% | [64.43, 82.33]% | 73.03% | 0.5405 |
| selected_pre_cue | 68.94% | [61.50, 77.31]% | 68.94% | 0.5922 |

## personal_16

| Model | Balanced accuracy | 95% participant interval | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| csp_reference | 68.36% | [59.88, 77.16]% | 68.36% | 0.6247 |
| eegnet_reference | 67.05% | [57.95, 76.62]% | 67.05% | 0.6436 |
| selected_zero_calibration | 73.07% | [64.58, 82.48]% | 73.07% | 0.5317 |
| selected_pre_cue | 72.15% | [64.35, 80.17]% | 72.15% | 0.5683 |

Direct transfer, external LOSO, and session-disjoint labeled calibration have different training/test conditions. These scores are not paired improvements over the previous PhysioNet confirmation. See RESULTS.md and the underlying method-complete reports for paired intervals, all budgets, negative findings, provenance and limitations.

<!-- RESIDUAL_PERSONAL -->

## Supplementary source-prior personal kernels

All comparisons use identical session-E trials; nonzero methods use the same16 chronological session-T labels. Mathematical specification preceded acquisition; implementation, source scalers and C froze before the first external scores. Primary hybrid: paired BA difference vs intercept-only -0.08 pp [-0.23, +0.00]. See personal_kernels_summary.csv and the residual study report for all three kernels, every budget and limitations.
