# NeuroEdge — expanded-cohort confirmation

This expanded-cohort study follows the retained small-cohort v2 confirmation, which did not show a reliable benefit of longer training. Development training participants: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]. Validation participants: [61, 62, 63, 64, 65, 66, 67, 68, 69, 70]. Confirmation participants: [71, 72, 73, 74, 75, 76, 77, 78, 79, 80]. The expanded plan was frozen before acquiring or scoring its new confirmation cohort. This is a sequential exploratory study; previous negative results remain part of the evidence. Its different cohorts do not permit a causal attribution to dataset size alone.

Fixed channels: 16; training seed: 7; selected candidate: csp_lda. Its development balanced accuracy was 0.621. All final models refit using the 70 development participants. Neural epoch counts were frozen before confirmation; the old-settings neural baseline uses a fixed 25 epochs. No confirmation data are used for scaling, tangent reference, checkpoint selection or calibration.

| Model | Confirmation participants | Balanced accuracy | Descriptive 95% participant interval | Accuracy |
|---|---:|---:|---:|---:|
| csp_lda | 10 | 0.612 | [0.536, 0.696] | 0.611 |
| eegnet_old_settings_25epochs | 10 | 0.631 | [0.536, 0.741] | 0.638 |

## Interpretation boundaries

Use the baselines measured on these same confirmation participants for comparisons. Do not subtract the v0.1 54.8% score: it used different evaluation participants and training sizes. Ten participants, one training seed, one validation split and one public dataset give a pilot result, not a universal performance guarantee. Intervals are conditional descriptive participant bootstrap intervals. The four-band CSP uses extra fixed zero-phase filters on the existing offline epochs. The tangent model uses OAS covariance, per-trial unit-trace normalization, a training-fitted Riemannian tangent reference with tsupdate=False, training-fitted feature scaling, and logistic regression. It uses no evaluation-batch reference. Longer neural training and BatchNorm momentum changes were candidates, not assumed improvements. No clinical, real-time, MCU, energy, algorithmic-novelty or state-of-the-art claim.

## Reproduction

Run python -m neuroedge.accuracy_expanded develop --plan configs/accuracy_expanded.yaml, then confirm, then verify. Existing frozen runs are preserved. See development_summary.csv, selection.json, confirmation_started.json, the confirmation CSV files, individual model artifacts and training histories.

## Prespecified source-size comparison

The same CSP+LDA pipeline trained on 20 source participants scored 57.49% balanced accuracy, versus 61.19% with 70 source participants on the same 10 confirmation participants. The paired difference was +3.70 percentage points, descriptive 95% participant interval [-1.89, +8.62]. This supplementary comparison was frozen before confirmation began. One fixed nested source cohort cannot establish a universal monotonic benefit of more data. See source_size_control for per-participant results and the original fitted control model.
