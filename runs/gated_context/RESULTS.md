# Fixed versus quality-conditioned context fusion

This follow-up was registered after the 63-candidate source screen, before any external EEG acquisition. The parent was paused during reused source validation so both new variants could finish source development and freeze before external data. Previously inspected people remain development data; external results never choose architecture, seed, gate or epoch.

Both networks retain active log-variance features and context contrasts, share BatchNorm across concatenated task/rest time points, and enforce reflection equivariance. Both have the SAME gate architecture and parameter count. The fixed control feeds constant zero conditioning features to the gate, learning population-fixed weights. The quality-conditioned version feeds relative baseline log variance and its temporal dispersion. These are operational quality proxies, not ground-truth artifact labels. Gate weights lie in [0,1]. Both feature paths retain task-only information so noisy context can be downweighted.

The architecture combines known variance, conditioning and symmetry ideas. World-first novelty is not established. All models still require the same 16 electrodes and the preceding 1.5-second context. This is an offline recorded-data study.

Frozen source-selected fusion: `fixed_gate`, reused-validation balanced accuracy 65.84%. Both variants are reported in direct transfer. Only this source-selected variant is refitted in external LOSO, without external reselection.

## Direct transfer

| Pipeline | Balanced accuracy | 95% participant interval | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| CSP reference | 66.63% | [57.52, 76.74]% | 66.63% | 0.6301 |
| EEGNet reference | 66.94% | [57.25, 76.81]% | 66.94% | 0.6434 |
| Selected: trial only | 73.69% | [64.58, 83.41]% | 73.69% | 0.5325 |
| Selected: pre-cue context | 71.84% | [64.47, 80.09]% | 71.84% | 0.5714 |
| Fixed learned gate | 73.30% | [65.62, 81.75]% | 73.30% | 0.5365 |
| Quality-conditioned gate | 69.14% | [62.38, 76.74]% | 69.14% | 0.6419 |
## Frozen-algorithm external LOSO

| Pipeline | Balanced accuracy | 95% participant interval | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| CSP reference | 57.60% | [52.43, 63.81]% | 57.60% | 0.6840 |
| EEGNet reference | 67.59% | [58.26, 76.97]% | 67.59% | 0.6490 |
| Selected: trial only | 73.03% | [64.43, 82.33]% | 73.03% | 0.5405 |
| Selected: pre-cue context | 68.94% | [61.50, 77.31]% | 68.94% | 0.5922 |
| Source-selected fusion | 71.84% | [63.42, 80.67]% | 71.84% | 0.5668 |

## Paired participant comparisons

- external_transfer: quality_gate versus fixed_gate: -4.17 pp [-5.79, -2.39].
- external_transfer: quality_gate versus selected_pre_cue: -2.70 pp [-4.55, -0.81].
- external_transfer: fixed_gate versus selected_pre_cue: +1.47 pp [-0.39, +3.47].
- external_loso: selected_fusion versus selected_pre_cue: +2.89 pp [+0.58, +5.17].
- external_loso: selected_fusion versus selected_zero_calibration: -1.20 pp [-2.89, +1.04].

Intervals are conditional participant bootstraps on nine people; a paired interval crossing zero does not establish a reliable positive gain. The direct-transfer and LOSO protocols have different training conditions. These are not paired accuracy gains over the older PhysioNet confirmation. The gate comparison tests conditioning within this specified architecture; it is not a causal physiological explanation or proof of general novelty.

Independent verification: **PASS**, including reinference of all six source OOF fits, both validation/final variants and every external LOSO fold. No adverse source or external result is omitted. Sources and original methods are in docs/CONTEXT_METHODS.md and the parent report.
