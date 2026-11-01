# NeuroEdge: expanded covariance and donor-stack study

The fixed primary donor stack, 16 sensors, strict prediction and 144 T labels, scores 84.95% balanced accuracy [76.54, 92.28]%. The paired difference versus the previous same-budget strict fusion is -0.15 pp [-1.31, +1.16]. This interval does not establish reliable positive improvement.

Previously inspected dataset; exploratory continuation. Nine people, binary left/right imagery, the same 1, 296 E trials. Sensor and unlabeled-batch requirements are reported separately. All parameters and combinations were frozen using T data only before new E scoring.

The highest observed secondary mean is 87.04% for Equal probability mean, 22 sensors, batch_unlabeled, 144 T labels. This is a post-evaluation descriptive maximum, not a newly selected confirmatory winner.

## Requirements and fair interpretation

- Strict: every fitted transform uses T support only; independent E trials do not affect one another.
- Batch-unlabeled: the complete unlabeled E batch supplies a covariance center. This uses future unlabeled trials and is not strict, causal, zero-calibration or independent-trial performance.
- 22 sensors: all EEG electrodes, with a 22-channel common average; EOG is excluded. This changes the sensor requirement from the previous 16-channel study.
- All methods require the indicated personal T labels. Reference rows always retain their actual 16-sensor, strict requirement even in a22-sensor or batch panel.
- The previous fusion benchmark is 85.11% at 144 labels. Its values use exactly the same E trials as this report.

## 16 sensors / batch_unlabeled

Columns are first-session labeled support counts. Values are participant-mean balanced accuracy.

| Method | 16 | 64 | 144 |
| --- | --- | --- | --- |
| Shrinkage FBCSP + LDA | 66.67% | 80.32% | 82.56% |
| FBCSP RBF SVM | 65.05% | 78.32% | 83.64% |
| Fisher-selected FBCSP SVM | 66.13% | 81.87% | 85.11% |
| Tangent logistic | 66.82% | 79.48% | 82.48% |
| Tangent RBF SVM | 66.90% | 78.63% | 83.26% |
| Joint CSP/tangent SVM | 65.12% | 79.09% | 84.41% |
| Multi-order covariance prototypes | 66.05% | 77.62% | 81.25% |
| Anchored multiview residual | 73.69% | 82.72% | 85.73% |
| Nested donor stack (primary) | 73.53% | 82.72% | 85.80% |
| Equal probability mean | 69.14% | 81.71% | 85.65% |
| Equal logit mean | 66.98% | 80.48% | 84.95% |
| Previous fusion (16 sensors, strict) | 74.15% | 79.48% | 85.11% |
| Previous tangent SVM (16, strict) | 59.65% | 79.71% | 83.64% |
| Previous intercept (16, strict) | 73.07% | 72.30% | 72.22% |

## 16 sensors / strict

Columns are first-session labeled support counts. Values are participant-mean balanced accuracy.

| Method | 16 | 64 | 144 |
| --- | --- | --- | --- |
| Shrinkage FBCSP + LDA | 64.89% | 78.24% | 82.79% |
| FBCSP RBF SVM | 64.51% | 77.85% | 81.79% |
| Fisher-selected FBCSP SVM | 64.35% | 80.94% | 83.41% |
| Tangent logistic | 68.06% | 79.55% | 83.64% |
| Tangent RBF SVM | 67.44% | 78.70% | 81.64% |
| Joint CSP/tangent SVM | 64.66% | 79.78% | 83.02% |
| Multi-order covariance prototypes | 61.65% | 76.31% | 79.94% |
| Anchored multiview residual | 74.61% | 81.17% | 84.80% |
| Nested donor stack (primary) | 74.38% | 82.56% | 84.95% |
| Equal probability mean | 67.82% | 80.71% | 84.34% |
| Equal logit mean | 65.82% | 79.63% | 84.03% |
| Previous fusion (16 sensors, strict) | 74.15% | 79.48% | 85.11% |
| Previous tangent SVM (16, strict) | 59.65% | 79.71% | 83.64% |
| Previous intercept (16, strict) | 73.07% | 72.30% | 72.22% |

## 22 sensors / batch_unlabeled

Columns are first-session labeled support counts. Values are participant-mean balanced accuracy.

| Method | 16 | 64 | 144 |
| --- | --- | --- | --- |
| Shrinkage FBCSP + LDA | 66.28% | 80.79% | 85.73% |
| FBCSP RBF SVM | 66.98% | 80.94% | 83.87% |
| Fisher-selected FBCSP SVM | 65.82% | 79.63% | 85.19% |
| Tangent logistic | 69.06% | 81.94% | 83.87% |
| Tangent RBF SVM | 68.06% | 81.64% | 84.41% |
| Joint CSP/tangent SVM | 66.51% | 80.86% | 84.03% |
| Multi-order covariance prototypes | 66.28% | 77.93% | 82.72% |
| Anchored multiview residual | 74.85% | 82.02% | 85.34% |
| Nested donor stack (primary) | 72.84% | 83.26% | 86.57% |
| Equal probability mean | 69.68% | 82.64% | 87.04% |
| Equal logit mean | 67.05% | 82.33% | 86.57% |
| Previous fusion (16 sensors, strict) | 74.15% | 79.48% | 85.11% |
| Previous tangent SVM (16, strict) | 59.65% | 79.71% | 83.64% |
| Previous intercept (16, strict) | 73.07% | 72.30% | 72.22% |

## 22 sensors / strict

Columns are first-session labeled support counts. Values are participant-mean balanced accuracy.

| Method | 16 | 64 | 144 |
| --- | --- | --- | --- |
| Shrinkage FBCSP + LDA | 66.13% | 80.02% | 82.64% |
| FBCSP RBF SVM | 67.05% | 78.70% | 82.25% |
| Fisher-selected FBCSP SVM | 67.75% | 80.48% | 83.64% |
| Tangent logistic | 68.13% | 80.48% | 84.49% |
| Tangent RBF SVM | 68.29% | 80.17% | 84.10% |
| Joint CSP/tangent SVM | 65.51% | 81.48% | 82.95% |
| Multi-order covariance prototypes | 63.43% | 77.39% | 80.79% |
| Anchored multiview residual | 75.69% | 82.18% | 84.49% |
| Nested donor stack (primary) | 73.77% | 83.02% | 86.50% |
| Equal probability mean | 70.76% | 82.56% | 85.57% |
| Equal logit mean | 66.59% | 81.64% | 85.34% |
| Previous fusion (16 sensors, strict) | 74.15% | 79.48% | 85.11% |
| Previous tangent SVM (16, strict) | 59.65% | 79.71% | 83.64% |
| Previous intercept (16, strict) | 73.07% | 72.30% | 72.22% |

## All intervals, losses and matched reference differences

| Sensors | Condition | Labels | Method | BA | 95% participant interval | Difference vs previous fusion (pp) | 95% paired interval (pp) | Log loss |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 16 | batch_unlabeled | 16 | Anchored multiview residual | 73.69% | [64.43, 83.18]% | -0.46 | [-4.09, +2.16] | 0.5244 |
| 16 | batch_unlabeled | 16 | Nested donor stack (primary) | 73.53% | [63.89, 83.41]% | -0.62 | [-4.40, +2.62] | 0.5329 |
| 16 | batch_unlabeled | 16 | Equal logit mean | 66.98% | [59.03, 75.31]% | -7.18 | [-11.65, -2.62] | 0.6632 |
| 16 | batch_unlabeled | 16 | Equal probability mean | 69.14% | [60.11, 78.40]% | -5.02 | [-8.80, -1.16] | 0.5856 |
| 16 | batch_unlabeled | 16 | Shrinkage FBCSP + LDA | 66.67% | [58.87, 74.85]% | -7.48 | [-12.04, -2.93] | 7.7022 |
| 16 | batch_unlabeled | 16 | FBCSP RBF SVM | 65.05% | [55.32, 74.07]% | -9.10 | [-14.43, -3.86] | 0.6449 |
| 16 | batch_unlabeled | 16 | Joint CSP/tangent SVM | 65.12% | [54.40, 75.54]% | -9.03 | [-14.89, -3.32] | 0.6449 |
| 16 | batch_unlabeled | 16 | Multi-order covariance prototypes | 66.05% | [56.79, 74.92]% | -8.10 | [-14.51, -1.46] | 0.6252 |
| 16 | batch_unlabeled | 16 | Previous intercept (16, strict) | 73.07% | [64.58, 82.48]% | -1.08 | [-2.55, +0.39] | 0.5317 |
| 16 | batch_unlabeled | 16 | Previous fusion (16 sensors, strict) | 74.15% | [65.66, 83.33]% | +0.00 | [+0.00, +0.00] | 0.5336 |
| 16 | batch_unlabeled | 16 | Previous tangent SVM (16, strict) | 59.65% | [52.85, 67.82]% | -14.51 | [-21.91, -8.79] | 0.6749 |
| 16 | batch_unlabeled | 16 | Fisher-selected FBCSP SVM | 66.13% | [56.33, 75.23]% | -8.02 | [-14.66, -1.47] | 0.6381 |
| 16 | batch_unlabeled | 16 | Tangent logistic | 66.82% | [55.94, 76.62]% | -7.33 | [-14.04, -1.16] | 0.6092 |
| 16 | batch_unlabeled | 16 | Tangent RBF SVM | 66.90% | [54.94, 77.78]% | -7.25 | [-14.89, -0.23] | 0.6642 |
| 16 | batch_unlabeled | 64 | Anchored multiview residual | 82.72% | [74.30, 90.28]% | +3.24 | [+1.77, +4.71] | 0.4052 |
| 16 | batch_unlabeled | 64 | Nested donor stack (primary) | 82.72% | [74.00, 90.43]% | +3.24 | [+1.62, +4.94] | 0.3813 |
| 16 | batch_unlabeled | 64 | Equal logit mean | 80.48% | [70.68, 88.81]% | +1.00 | [-2.01, +3.86] | 0.4056 |
| 16 | batch_unlabeled | 64 | Equal probability mean | 81.71% | [72.53, 89.74]% | +2.24 | [+0.00, +4.48] | 0.4658 |
| 16 | batch_unlabeled | 64 | Shrinkage FBCSP + LDA | 80.32% | [70.91, 88.27]% | +0.85 | [-2.16, +4.17] | 1.6106 |
| 16 | batch_unlabeled | 64 | FBCSP RBF SVM | 78.32% | [68.21, 86.96]% | -1.16 | [-4.17, +1.54] | 0.5426 |
| 16 | batch_unlabeled | 64 | Joint CSP/tangent SVM | 79.09% | [69.14, 87.42]% | -0.39 | [-3.78, +2.78] | 0.5697 |
| 16 | batch_unlabeled | 64 | Multi-order covariance prototypes | 77.62% | [68.60, 85.65]% | -1.85 | [-5.17, +1.70] | 0.5033 |
| 16 | batch_unlabeled | 64 | Previous intercept (16, strict) | 72.30% | [62.19, 83.33]% | -7.18 | [-14.81, -0.62] | 0.5313 |
| 16 | batch_unlabeled | 64 | Previous fusion (16 sensors, strict) | 79.48% | [71.68, 86.96]% | +0.00 | [+0.00, +0.00] | 0.4809 |
| 16 | batch_unlabeled | 64 | Previous tangent SVM (16, strict) | 79.71% | [70.21, 87.96]% | +0.23 | [-2.24, +2.39] | 0.6211 |
| 16 | batch_unlabeled | 64 | Fisher-selected FBCSP SVM | 81.87% | [71.91, 90.59]% | +2.39 | [-1.39, +6.33] | 0.5552 |
| 16 | batch_unlabeled | 64 | Tangent logistic | 79.48% | [70.22, 87.58]% | -0.00 | [-2.47, +1.85] | 0.4668 |
| 16 | batch_unlabeled | 64 | Tangent RBF SVM | 78.63% | [69.37, 86.88]% | -0.85 | [-3.01, +0.93] | 0.5662 |
| 16 | batch_unlabeled | 144 | Anchored multiview residual | 85.73% | [78.16, 92.59]% | +0.62 | [-1.16, +2.39] | 0.3543 |
| 16 | batch_unlabeled | 144 | Nested donor stack (primary) | 85.80% | [77.78, 92.90]% | +0.69 | [-0.46, +1.77] | 0.3126 |
| 16 | batch_unlabeled | 144 | Equal logit mean | 84.95% | [76.31, 92.36]% | -0.15 | [-1.62, +1.23] | 0.3238 |
| 16 | batch_unlabeled | 144 | Equal probability mean | 85.65% | [77.16, 92.90]% | +0.54 | [-0.93, +2.08] | 0.4041 |
| 16 | batch_unlabeled | 144 | Shrinkage FBCSP + LDA | 82.56% | [72.99, 90.59]% | -2.55 | [-4.86, -0.62] | 0.7729 |
| 16 | batch_unlabeled | 144 | FBCSP RBF SVM | 83.64% | [74.46, 91.59]% | -1.47 | [-3.32, +0.31] | 0.4877 |
| 16 | batch_unlabeled | 144 | Joint CSP/tangent SVM | 84.41% | [75.08, 92.36]% | -0.69 | [-2.62, +1.00] | 0.4968 |
| 16 | batch_unlabeled | 144 | Multi-order covariance prototypes | 81.25% | [71.76, 89.51]% | -3.86 | [-6.40, -1.47] | 0.4870 |
| 16 | batch_unlabeled | 144 | Previous intercept (16, strict) | 72.22% | [61.65, 83.26]% | -12.89 | [-23.15, -4.09] | 0.5317 |
| 16 | batch_unlabeled | 144 | Previous fusion (16 sensors, strict) | 85.11% | [76.93, 92.52]% | +0.00 | [+0.00, +0.00] | 0.4188 |
| 16 | batch_unlabeled | 144 | Previous tangent SVM (16, strict) | 83.64% | [74.61, 91.44]% | -1.47 | [-3.32, +0.23] | 0.5890 |
| 16 | batch_unlabeled | 144 | Fisher-selected FBCSP SVM | 85.11% | [75.85, 93.29]% | +0.00 | [-2.09, +2.08] | 0.4728 |
| 16 | batch_unlabeled | 144 | Tangent logistic | 82.48% | [73.23, 90.59]% | -2.62 | [-4.86, -0.31] | 0.3656 |
| 16 | batch_unlabeled | 144 | Tangent RBF SVM | 83.26% | [74.15, 90.97]% | -1.85 | [-4.63, +0.69] | 0.5558 |
| 16 | strict | 16 | Anchored multiview residual | 74.61% | [66.12, 83.87]% | +0.46 | [-1.47, +2.08] | 0.5193 |
| 16 | strict | 16 | Nested donor stack (primary) | 74.38% | [65.43, 83.95]% | +0.23 | [-2.16, +2.70] | 0.5315 |
| 16 | strict | 16 | Equal logit mean | 65.82% | [56.71, 75.08]% | -8.33 | [-13.97, -2.70] | 0.6815 |
| 16 | strict | 16 | Equal probability mean | 67.82% | [58.95, 76.77]% | -6.33 | [-11.50, -1.16] | 0.5894 |
| 16 | strict | 16 | Shrinkage FBCSP + LDA | 64.89% | [56.25, 73.77]% | -9.26 | [-15.20, -3.47] | 7.9240 |
| 16 | strict | 16 | FBCSP RBF SVM | 64.51% | [54.55, 73.92]% | -9.65 | [-17.13, -1.70] | 0.6502 |
| 16 | strict | 16 | Joint CSP/tangent SVM | 64.66% | [55.17, 73.92]% | -9.49 | [-15.97, -2.85] | 0.6481 |
| 16 | strict | 16 | Multi-order covariance prototypes | 61.65% | [53.09, 71.45]% | -12.50 | [-21.30, -5.40] | 0.6838 |
| 16 | strict | 16 | Previous intercept (16, strict) | 73.07% | [64.58, 82.48]% | -1.08 | [-2.55, +0.39] | 0.5317 |
| 16 | strict | 16 | Previous fusion (16 sensors, strict) | 74.15% | [65.66, 83.33]% | +0.00 | [+0.00, +0.00] | 0.5336 |
| 16 | strict | 16 | Previous tangent SVM (16, strict) | 59.65% | [52.85, 67.82]% | -14.51 | [-21.91, -8.79] | 0.6749 |
| 16 | strict | 16 | Fisher-selected FBCSP SVM | 64.35% | [54.63, 74.23]% | -9.80 | [-16.82, -3.70] | 0.6424 |
| 16 | strict | 16 | Tangent logistic | 68.06% | [58.02, 78.01]% | -6.10 | [-11.57, -0.93] | 0.6098 |
| 16 | strict | 16 | Tangent RBF SVM | 67.44% | [56.33, 77.85]% | -6.71 | [-13.19, -0.31] | 0.6664 |
| 16 | strict | 64 | Anchored multiview residual | 81.17% | [72.69, 88.81]% | +1.70 | [-0.54, +3.94] | 0.4312 |
| 16 | strict | 64 | Nested donor stack (primary) | 82.56% | [74.23, 89.89]% | +3.09 | [+1.16, +5.09] | 0.4201 |
| 16 | strict | 64 | Equal logit mean | 79.63% | [70.60, 87.50]% | +0.15 | [-2.62, +3.16] | 0.4389 |
| 16 | strict | 64 | Equal probability mean | 80.71% | [71.22, 88.81]% | +1.23 | [-1.77, +4.17] | 0.4765 |
| 16 | strict | 64 | Shrinkage FBCSP + LDA | 78.24% | [69.60, 85.96]% | -1.23 | [-4.09, +1.85] | 2.1090 |
| 16 | strict | 64 | FBCSP RBF SVM | 77.85% | [68.90, 85.57]% | -1.62 | [-4.48, +1.70] | 0.5609 |
| 16 | strict | 64 | Joint CSP/tangent SVM | 79.78% | [70.45, 87.58]% | +0.31 | [-2.55, +3.24] | 0.5602 |
| 16 | strict | 64 | Multi-order covariance prototypes | 76.31% | [66.05, 84.95]% | -3.16 | [-7.33, +1.16] | 0.5316 |
| 16 | strict | 64 | Previous intercept (16, strict) | 72.30% | [62.19, 83.33]% | -7.18 | [-14.81, -0.62] | 0.5313 |
| 16 | strict | 64 | Previous fusion (16 sensors, strict) | 79.48% | [71.68, 86.96]% | +0.00 | [+0.00, +0.00] | 0.4809 |
| 16 | strict | 64 | Previous tangent SVM (16, strict) | 79.71% | [70.21, 87.96]% | +0.23 | [-2.24, +2.39] | 0.6211 |
| 16 | strict | 64 | Fisher-selected FBCSP SVM | 80.94% | [72.53, 88.19]% | +1.47 | [-1.93, +5.09] | 0.5536 |
| 16 | strict | 64 | Tangent logistic | 79.55% | [70.60, 87.27]% | +0.08 | [-2.08, +1.93] | 0.4719 |
| 16 | strict | 64 | Tangent RBF SVM | 78.70% | [69.37, 87.04]% | -0.77 | [-3.32, +1.54] | 0.5748 |
| 16 | strict | 144 | Anchored multiview residual | 84.80% | [76.93, 91.82]% | -0.31 | [-1.39, +0.85] | 0.3817 |
| 16 | strict | 144 | Nested donor stack (primary) | 84.95% | [76.54, 92.28]% | -0.15 | [-1.31, +1.16] | 0.3555 |
| 16 | strict | 144 | Equal logit mean | 84.03% | [75.46, 91.44]% | -1.08 | [-2.85, +1.00] | 0.3455 |
| 16 | strict | 144 | Equal probability mean | 84.34% | [75.54, 91.90]% | -0.77 | [-2.39, +0.93] | 0.4197 |
| 16 | strict | 144 | Shrinkage FBCSP + LDA | 82.79% | [74.23, 90.51]% | -2.31 | [-4.78, +0.08] | 1.0362 |
| 16 | strict | 144 | FBCSP RBF SVM | 81.79% | [72.30, 89.89]% | -3.32 | [-5.63, -1.00] | 0.5221 |
| 16 | strict | 144 | Joint CSP/tangent SVM | 83.02% | [74.00, 90.90]% | -2.08 | [-3.47, -0.46] | 0.5344 |
| 16 | strict | 144 | Multi-order covariance prototypes | 79.94% | [71.45, 87.73]% | -5.17 | [-7.87, -2.85] | 0.4871 |
| 16 | strict | 144 | Previous intercept (16, strict) | 72.22% | [61.65, 83.26]% | -12.89 | [-23.15, -4.09] | 0.5317 |
| 16 | strict | 144 | Previous fusion (16 sensors, strict) | 85.11% | [76.93, 92.52]% | +0.00 | [+0.00, +0.00] | 0.4188 |
| 16 | strict | 144 | Previous tangent SVM (16, strict) | 83.64% | [74.61, 91.44]% | -1.47 | [-3.32, +0.23] | 0.5890 |
| 16 | strict | 144 | Fisher-selected FBCSP SVM | 83.41% | [73.77, 91.82]% | -1.70 | [-3.78, +0.54] | 0.4940 |
| 16 | strict | 144 | Tangent logistic | 83.64% | [75.69, 90.13]% | -1.47 | [-4.01, +0.93] | 0.3688 |
| 16 | strict | 144 | Tangent RBF SVM | 81.64% | [72.92, 89.20]% | -3.47 | [-5.17, -1.70] | 0.5162 |
| 22 | batch_unlabeled | 16 | Anchored multiview residual | 74.85% | [66.05, 84.11]% | +0.69 | [-2.55, +3.55] | 0.5212 |
| 22 | batch_unlabeled | 16 | Nested donor stack (primary) | 72.84% | [62.50, 83.41]% | -1.31 | [-5.63, +3.01] | 0.5207 |
| 22 | batch_unlabeled | 16 | Equal logit mean | 67.05% | [58.33, 75.69]% | -7.10 | [-12.73, -0.31] | 0.6647 |
| 22 | batch_unlabeled | 16 | Equal probability mean | 69.68% | [59.80, 79.17]% | -4.48 | [-9.49, +1.62] | 0.5842 |
| 22 | batch_unlabeled | 16 | Shrinkage FBCSP + LDA | 66.28% | [57.64, 74.92]% | -7.87 | [-13.58, -1.00] | 8.2522 |
| 22 | batch_unlabeled | 16 | FBCSP RBF SVM | 66.98% | [57.48, 75.77]% | -7.18 | [-12.65, -0.46] | 0.6390 |
| 22 | batch_unlabeled | 16 | Joint CSP/tangent SVM | 66.51% | [57.18, 75.31]% | -7.64 | [-12.96, -1.54] | 0.6443 |
| 22 | batch_unlabeled | 16 | Multi-order covariance prototypes | 66.28% | [57.18, 75.46]% | -7.87 | [-13.12, -2.16] | 0.6213 |
| 22 | batch_unlabeled | 16 | Previous intercept (16, strict) | 73.07% | [64.58, 82.48]% | -1.08 | [-2.55, +0.39] | 0.5317 |
| 22 | batch_unlabeled | 16 | Previous fusion (16 sensors, strict) | 74.15% | [65.66, 83.33]% | +0.00 | [+0.00, +0.00] | 0.5336 |
| 22 | batch_unlabeled | 16 | Previous tangent SVM (16, strict) | 59.65% | [52.85, 67.82]% | -14.51 | [-21.91, -8.79] | 0.6749 |
| 22 | batch_unlabeled | 16 | Fisher-selected FBCSP SVM | 65.82% | [57.02, 74.31]% | -8.33 | [-14.74, -0.69] | 0.6434 |
| 22 | batch_unlabeled | 16 | Tangent logistic | 69.06% | [59.26, 78.70]% | -5.09 | [-10.26, +0.31] | 0.6019 |
| 22 | batch_unlabeled | 16 | Tangent RBF SVM | 68.06% | [56.94, 78.78]% | -6.10 | [-12.73, +0.31] | 0.6675 |
| 22 | batch_unlabeled | 64 | Anchored multiview residual | 82.02% | [73.92, 89.66]% | +2.55 | [+0.23, +4.55] | 0.4320 |
| 22 | batch_unlabeled | 64 | Nested donor stack (primary) | 83.26% | [74.61, 90.82]% | +3.78 | [+1.39, +6.25] | 0.3876 |
| 22 | batch_unlabeled | 64 | Equal logit mean | 82.33% | [72.99, 90.43]% | +2.85 | [+0.00, +5.79] | 0.3939 |
| 22 | batch_unlabeled | 64 | Equal probability mean | 82.64% | [73.46, 90.66]% | +3.16 | [+0.77, +5.79] | 0.4673 |
| 22 | batch_unlabeled | 64 | Shrinkage FBCSP + LDA | 80.79% | [70.99, 89.43]% | +1.31 | [-1.85, +4.63] | 1.5055 |
| 22 | batch_unlabeled | 64 | FBCSP RBF SVM | 80.94% | [71.06, 89.35]% | +1.47 | [-2.16, +5.02] | 0.5496 |
| 22 | batch_unlabeled | 64 | Joint CSP/tangent SVM | 80.86% | [71.53, 88.73]% | +1.39 | [-1.54, +4.17] | 0.5591 |
| 22 | batch_unlabeled | 64 | Multi-order covariance prototypes | 77.93% | [67.98, 86.50]% | -1.54 | [-5.09, +2.16] | 0.4566 |
| 22 | batch_unlabeled | 64 | Previous intercept (16, strict) | 72.30% | [62.19, 83.33]% | -7.18 | [-14.81, -0.62] | 0.5313 |
| 22 | batch_unlabeled | 64 | Previous fusion (16 sensors, strict) | 79.48% | [71.68, 86.96]% | +0.00 | [+0.00, +0.00] | 0.4809 |
| 22 | batch_unlabeled | 64 | Previous tangent SVM (16, strict) | 79.71% | [70.21, 87.96]% | +0.23 | [-2.24, +2.39] | 0.6211 |
| 22 | batch_unlabeled | 64 | Fisher-selected FBCSP SVM | 79.63% | [67.98, 89.66]% | +0.15 | [-4.71, +4.94] | 0.5432 |
| 22 | batch_unlabeled | 64 | Tangent logistic | 81.94% | [72.69, 89.89]% | +2.47 | [-0.23, +4.78] | 0.4677 |
| 22 | batch_unlabeled | 64 | Tangent RBF SVM | 81.64% | [72.22, 89.58]% | +2.16 | [-0.54, +4.71] | 0.6192 |
| 22 | batch_unlabeled | 144 | Anchored multiview residual | 85.34% | [77.16, 92.75]% | +0.23 | [-2.47, +2.78] | 0.3461 |
| 22 | batch_unlabeled | 144 | Nested donor stack (primary) | 86.57% | [77.85, 93.83]% | +1.47 | [-0.46, +3.55] | 0.3182 |
| 22 | batch_unlabeled | 144 | Equal logit mean | 86.57% | [77.39, 94.37]% | +1.47 | [-1.00, +3.70] | 0.3068 |
| 22 | batch_unlabeled | 144 | Equal probability mean | 87.04% | [78.16, 94.60]% | +1.93 | [+0.00, +3.94] | 0.3917 |
| 22 | batch_unlabeled | 144 | Shrinkage FBCSP + LDA | 85.73% | [76.62, 93.36]% | +0.62 | [-2.31, +3.47] | 0.8090 |
| 22 | batch_unlabeled | 144 | FBCSP RBF SVM | 83.87% | [74.92, 91.59]% | -1.23 | [-3.47, +0.93] | 0.4894 |
| 22 | batch_unlabeled | 144 | Joint CSP/tangent SVM | 84.03% | [73.84, 92.75]% | -1.08 | [-3.86, +1.31] | 0.4914 |
| 22 | batch_unlabeled | 144 | Multi-order covariance prototypes | 82.72% | [73.15, 90.97]% | -2.39 | [-4.78, -0.23] | 0.3870 |
| 22 | batch_unlabeled | 144 | Previous intercept (16, strict) | 72.22% | [61.65, 83.26]% | -12.89 | [-23.15, -4.09] | 0.5317 |
| 22 | batch_unlabeled | 144 | Previous fusion (16 sensors, strict) | 85.11% | [76.93, 92.52]% | +0.00 | [+0.00, +0.00] | 0.4188 |
| 22 | batch_unlabeled | 144 | Previous tangent SVM (16, strict) | 83.64% | [74.61, 91.44]% | -1.47 | [-3.32, +0.23] | 0.5890 |
| 22 | batch_unlabeled | 144 | Fisher-selected FBCSP SVM | 85.19% | [76.08, 93.06]% | +0.08 | [-1.93, +1.77] | 0.4676 |
| 22 | batch_unlabeled | 144 | Tangent logistic | 83.87% | [74.77, 91.51]% | -1.23 | [-4.01, +1.54] | 0.3753 |
| 22 | batch_unlabeled | 144 | Tangent RBF SVM | 84.41% | [75.54, 92.36]% | -0.69 | [-2.70, +1.39] | 0.5192 |
| 22 | strict | 16 | Anchored multiview residual | 75.69% | [67.05, 84.95]% | +1.54 | [-0.23, +3.40] | 0.5141 |
| 22 | strict | 16 | Nested donor stack (primary) | 73.77% | [63.81, 84.03]% | -0.39 | [-4.24, +3.70] | 0.5330 |
| 22 | strict | 16 | Equal logit mean | 66.59% | [57.56, 75.62]% | -7.56 | [-13.97, -0.69] | 0.6579 |
| 22 | strict | 16 | Equal probability mean | 70.76% | [61.03, 80.40]% | -3.40 | [-9.49, +3.55] | 0.5785 |
| 22 | strict | 16 | Shrinkage FBCSP + LDA | 66.13% | [57.25, 75.00]% | -8.02 | [-14.66, -0.93] | 8.1949 |
| 22 | strict | 16 | FBCSP RBF SVM | 67.05% | [57.56, 76.77]% | -7.10 | [-12.73, -1.16] | 0.6432 |
| 22 | strict | 16 | Joint CSP/tangent SVM | 65.51% | [56.79, 74.15]% | -8.64 | [-15.35, -1.54] | 0.6476 |
| 22 | strict | 16 | Multi-order covariance prototypes | 63.43% | [54.55, 73.07]% | -10.73 | [-18.13, -4.63] | 0.6578 |
| 22 | strict | 16 | Previous intercept (16, strict) | 73.07% | [64.58, 82.48]% | -1.08 | [-2.55, +0.39] | 0.5317 |
| 22 | strict | 16 | Previous fusion (16 sensors, strict) | 74.15% | [65.66, 83.33]% | +0.00 | [+0.00, +0.00] | 0.5336 |
| 22 | strict | 16 | Previous tangent SVM (16, strict) | 59.65% | [52.85, 67.82]% | -14.51 | [-21.91, -8.79] | 0.6749 |
| 22 | strict | 16 | Fisher-selected FBCSP SVM | 67.75% | [58.49, 77.39]% | -6.40 | [-14.20, +1.16] | 0.6426 |
| 22 | strict | 16 | Tangent logistic | 68.13% | [58.02, 78.16]% | -6.02 | [-11.81, +0.15] | 0.5708 |
| 22 | strict | 16 | Tangent RBF SVM | 68.29% | [57.71, 78.63]% | -5.86 | [-11.88, +0.15] | 0.6690 |
| 22 | strict | 64 | Anchored multiview residual | 82.18% | [74.69, 89.12]% | +2.70 | [+1.16, +4.55] | 0.4141 |
| 22 | strict | 64 | Nested donor stack (primary) | 83.02% | [74.46, 90.66]% | +3.55 | [+1.62, +5.48] | 0.3839 |
| 22 | strict | 64 | Equal logit mean | 81.64% | [72.22, 89.74]% | +2.16 | [-0.77, +4.78] | 0.4030 |
| 22 | strict | 64 | Equal probability mean | 82.56% | [73.07, 90.59]% | +3.09 | [+0.23, +5.71] | 0.4635 |
| 22 | strict | 64 | Shrinkage FBCSP + LDA | 80.02% | [70.60, 88.04]% | +0.54 | [-2.55, +3.40] | 2.0479 |
| 22 | strict | 64 | FBCSP RBF SVM | 78.70% | [69.75, 86.27]% | -0.77 | [-4.09, +2.39] | 0.5673 |
| 22 | strict | 64 | Joint CSP/tangent SVM | 81.48% | [72.14, 89.12]% | +2.01 | [-1.16, +5.02] | 0.5858 |
| 22 | strict | 64 | Multi-order covariance prototypes | 77.39% | [68.06, 85.49]% | -2.08 | [-4.63, +0.46] | 0.4985 |
| 22 | strict | 64 | Previous intercept (16, strict) | 72.30% | [62.19, 83.33]% | -7.18 | [-14.81, -0.62] | 0.5313 |
| 22 | strict | 64 | Previous fusion (16 sensors, strict) | 79.48% | [71.68, 86.96]% | +0.00 | [+0.00, +0.00] | 0.4809 |
| 22 | strict | 64 | Previous tangent SVM (16, strict) | 79.71% | [70.21, 87.96]% | +0.23 | [-2.24, +2.39] | 0.6211 |
| 22 | strict | 64 | Fisher-selected FBCSP SVM | 80.48% | [70.68, 88.89]% | +1.00 | [-2.31, +4.48] | 0.5233 |
| 22 | strict | 64 | Tangent logistic | 80.48% | [71.06, 88.58]% | +1.00 | [-2.16, +4.32] | 0.4320 |
| 22 | strict | 64 | Tangent RBF SVM | 80.17% | [71.14, 88.12]% | +0.69 | [-2.16, +3.78] | 0.6002 |
| 22 | strict | 144 | Anchored multiview residual | 84.49% | [77.01, 91.51]% | -0.62 | [-2.31, +1.08] | 0.3859 |
| 22 | strict | 144 | Nested donor stack (primary) | 86.50% | [78.24, 93.83]% | +1.39 | [+0.08, +2.85] | 0.3189 |
| 22 | strict | 144 | Equal logit mean | 85.34% | [77.47, 92.44]% | +0.23 | [-0.85, +1.47] | 0.3332 |
| 22 | strict | 144 | Equal probability mean | 85.57% | [77.47, 92.82]% | +0.46 | [-0.54, +1.47] | 0.4120 |
| 22 | strict | 144 | Shrinkage FBCSP + LDA | 82.64% | [74.69, 89.74]% | -2.47 | [-4.78, -0.15] | 0.8405 |
| 22 | strict | 144 | FBCSP RBF SVM | 82.25% | [74.00, 89.58]% | -2.85 | [-5.32, -0.23] | 0.5162 |
| 22 | strict | 144 | Joint CSP/tangent SVM | 82.95% | [74.15, 90.66]% | -2.16 | [-4.32, -0.00] | 0.4953 |
| 22 | strict | 144 | Multi-order covariance prototypes | 80.79% | [71.83, 89.12]% | -4.32 | [-6.10, -2.31] | 0.4304 |
| 22 | strict | 144 | Previous intercept (16, strict) | 72.22% | [61.65, 83.26]% | -12.89 | [-23.15, -4.09] | 0.5317 |
| 22 | strict | 144 | Previous fusion (16 sensors, strict) | 85.11% | [76.93, 92.52]% | +0.00 | [+0.00, +0.00] | 0.4188 |
| 22 | strict | 144 | Previous tangent SVM (16, strict) | 83.64% | [74.61, 91.44]% | -1.47 | [-3.32, +0.23] | 0.5890 |
| 22 | strict | 144 | Fisher-selected FBCSP SVM | 83.64% | [74.15, 91.90]% | -1.47 | [-3.55, +0.39] | 0.4626 |
| 22 | strict | 144 | Tangent logistic | 84.49% | [76.16, 91.59]% | -0.62 | [-2.62, +1.62] | 0.3642 |
| 22 | strict | 144 | Tangent RBF SVM | 84.10% | [73.92, 92.28]% | -1.00 | [-4.63, +2.16] | 0.6068 |

For 22-sensor and batch-unlabeled rows, subtraction from the old 16-sensor strict reference changes the input requirement. It is a pipeline comparison, not an isolated algorithm gain. Intervals resample nine people; they are descriptive and unadjusted for secondary comparisons, overlap in donor cohorts and prior dataset inspection.

## Candidate screen and nested selection

71 fixed configurations were evaluated in 7, 668 donor fits across two sensor conditions, two input conditions, three budgets and nine T donors. Family settings for each outer person exclude that person completely. Meta-training predictions for donor d additionally exclude d from family selection. Stack C uses conditional leave-one-donor-out meta-validation. These internal scores are selection evidence, not an independent accuracy estimate. The 144-label arm emulates 96 labels during donor selection to retain later T validation runs; the final personal fit receives 144.

The screen covers shrinkage FBCSP, nonlinear and Fisher-selected CSP, tangent logistic/SVM, joint CSP/tangent kernels, multi-order covariance prototypes and a source-anchored multiview residual. The donor stack and multiview residual are project-specific combinations of established components; worldwide novelty is not established. Simplified power-mean features are not a complete reproduction of affine-invariant Mean Field. SVM sigmoid scores are uncalibrated surrogates.

Exact mathematics, candidate grid, prior work and limitations: [COVARIANCE_LAB_METHODS.md](../../docs/COVARIANCE_LAB_METHODS.md).

## Reproduction and evidence

```bash
python scripts/freeze_covariance_inputs.py --plan configs/covariance_lab_corrected.yaml
python -m neuroedge.covariance_study all --plan configs/covariance_lab_corrected.yaml --workers 4
python -m pytest -q
```

The complete earlier BNCI inputs must be restored first; see README.md. New methods or changed data require a fresh YAML/output identity. No method is modified after seeing this study's E scores.

Independent verification PASS: 972 checkpoint replays, 972 refits, 108 nested settings, 7, 668 donor candidate metrics and 217, 728 actual E prediction rows.

- `donors/*/predictions.csv`, `scores.csv`, `complete.json`: every donor candidate and split.
- `donor_summary.csv`, `selection.json`, `nested_meta_predictions.csv`: all choices, scores and nested identities.
- `models/`, `fit_receipts.json`: every selected model, support identity and hash.
- `predictions.csv`, `metrics_by_subject.csv`, `summary.csv`: all actual predictions and subject metrics.
- `paired_differences.csv`, `all_pairwise_comparisons.csv`: all retained positive and negative comparisons.
- `protocol.json`, `verification.json`: frozen implementation and independent verification.
The initial diagnostic run in `../covariance_lab` was invalidated after a matrix-exponential eigenvalue-floor error was found in its power-mean feature. Its original implementation and T-screen evidence are retained for audit, while incomplete model output is omitted. All results in this report come from a fresh corrected protocol and recomputed donor/final fits. A constant-SPD regression detects the original error and confirms the corrected matrix identity.
Raw EEG is public attributed data and is not redistributed. This offline study does not establish clinical or real-time performance.
