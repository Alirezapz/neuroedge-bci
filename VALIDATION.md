# Validation record

## Completed continuation: current evidence

All five registered studies and the combined report completed independent
verification. Earlier checkpoint sections below are retained as history.

| Check | Completed outcome |
|---|---|
| Software suite | 73 passed, five original ONNX warnings; tests/final_software_test_results.xml and final_software_test_run.txt |
| Source development | 65 base configurations, 195 source group fits; 35 main validation singles/combinations, two matched gates, and 12 source personal-kernel choices |
| Exact input recovery | All five development and all five external array/trial hashes match the frozen acquisition; tests/context_recovery_20261101.json |
| Core external verification | PASS: 11 source constituents and 99 external LOSO constituent fits reinferred; all 9 held people exclude both sessions from their fit |
| Gate verification | PASS: 6 source OOF fits, both validation/final variants and 9 external LOSO fits reinferred |
| Session-disjoint calibration | Both parent studies PASS; chronological 4/8/16 session-T supports and all 144 session-E trials/person checked |
| Personal residual kernels | PASS: 12 source settings, 288 source support fits, two source-only scalers and 144 external method/budget/person checks; same-budget bias predictions match the original calibration |
| Offline report QA | All six reports pass at 1280 px and 390 px: 12 layout/content checks, loaded scientific figures, no page overflow; tests/final_report_browser_qa.json |

Direct-transfer task-ensemble BA is 73.6883% versus 66.9367% for EEGNet; paired
difference 6.7515 pp, [0.8102, 11.2269]. External LOSO is 73.0324% versus 67.5926%;
paired difference 5.4398 pp, [0.6173, 9.3750]. These are whole-pipeline comparisons
on the same people within each protocol; bands, window and ensembling differ.
The two protocols have different training conditions. Intervals are descriptive
participant bootstraps on nine people, not proof of SOTA or clinical usability.

On the same session-E test set, 16-label intercept calibration scores 73.0710%.
The primary hybrid kernel scores 72.9938%, difference −0.0772 pp, [−0.2315, 0.0000].
Secondary geometry/spectral kernels reach 73.9969%/73.7654%; both paired
intervals against intercept-only include zero. Extra kernel accuracy gains are
not established. All negative findings are retained.

Numerical verification fixes retain original predictions, frozen choices,
model/data identities and all metric tolerances. Gated validation metrics now
use their validated archived probabilities; independently reinferred probabilities
retain the original float 32 comparison and exact-decision checks. Core CSV
verification uses round-trip binary 64 parsing, reproducing the extreme-probability
source log loss exactly. Audits and regressions are in tests/gated_verification_numerics_20261101.json,
tests/context_csv_precision_20261101.json and the test suite.

## Historical checkpoints


Completed on 2026-10-29 using Linux, Python 3.12.14, CPU PyTorch 2.9.1, MNE 1.13.2 and ONNX Runtime 1.31.1. Exact dependency pins are in requirements-tested.txt; original hardware and package metadata are retained with each run.

| Check | Observed result |
|---|---|
| Software invariants | 13 passed; see tests/v0_1_test_results.xml and tests/v0_1_test_run.txt |
| Real-data smoke study | 4 participants, 180 epochs, 18 completed model fits |
| Main real-data study | 20 participants, 900 epochs, 75 completed model fits |
| Main outer-test predictions | 13, 500 rows, including model/channel/seed repetitions |
| Independent main verification | PASS for all 75 saved models, including recomputed probabilities and decisions |
| Raw and prepared data integrity | Original EDF and cache checksums verified |
| Preprocessing provenance | Saved neural normalization agrees with training-only recomputation |
| Subject separation | Train, validation and test membership checked; calibration uses training participants only |
| FP 32 export | 15 of 15 models passed the declared numerical/decision agreement checks |
| INT 8 QDQ execution | 15 of 15 variants ran; no speed or accuracy improvement assumed |
| Repeated preparation and resume | Smoke data manifest unchanged after repeat preparation; 18 completed fits resumed without retraining |
| Fresh-run configuration helper | New name/output created without changing cohort or training settings |
| Browser replay | PASS in Chromium 153, desktop 1280×1000 and mobile 390×844; see runs/main/demo/browser_qa.json |
| Plot inspection | Main channel comparison and accuracy/CPU-cost plots visually inspected |

The exporter emitted legacy-export and constant-folding warnings. Exported graphs passed ONNX checks and actual ONNX Runtime agreement/execution checks. The tested dependency versions are pinned; a future exporter migration requires revalidation.

GitHub-hosted Actions and Windows execution were not performed. The provided CI workflow and Windows instructions are setup material, not claims of successful execution on those platforms. The archive excludes raw EDF recordings and large processed caches; the documented prepare command reacquires them. Browser screenshots were internal QA only and are not included in the deliverable.

These checks support the reproducibility and software integrity of this study. They do not establish clinical validity, causal physiological interpretation, superiority to published methods, physical device efficiency, or the absence of every possible defect. Observed predictive performance is limited; see the report and docs/RESULTS.md.


## Accuracy extension — 2026-10-31

The original 75-model verification was rerun and passed with all four frozen training-source hashes unchanged. The current software suite passes 18 tests, including participant separation, training-only BatchNorm recalibration, per-trial covariance normalization, non-transductive tangent inference, and rejection of retrospective source-size control specification. The archived 13-test v0.1 records remain available separately.

The small-cohort accuracy study retained all six development candidates and its negative independent confirmation. All three final saved models reproduced their stored predictions. The subsequent expanded study used 60 training, 10 validation and 10 new confirmation participants; final models refit on 70 development participants. Its independent verification and the prespecified 20-source CSP control verification passed. Negative results and uncertainty are retained in the reports. This is sequential exploratory work with one training seed per method.

The memorization control uses 32 real training trials and is explicitly not reported as generalization performance. New replay pages show stored predictions from their own held-out cohorts. Windows execution and GitHub-hosted CI remain untested here. No new device/energy/clinical claim is made.

## Frontier extension — 2026-10-31

The new methods and original frozen source are separated. No degree level is
imposed. This is a broad but bounded sequential exploratory study, not a new
algorithm or an assertion that every EEG method has been evaluated.

| Check | Observed result |
|---|---|
| Full software suite | 40 passed, 5 original ONNX warnings; tests/frontier_test_results.xml and tests/frontier_test_run.txt |
| End-to-end software fixture | Synthetic arrays exercise source-only selection, combinations, confirmation, independent verification, reporting and rejection of tampered final predictions; never research accuracy |
| Real-signal functional probes | Seven representative candidate families completed three small development-group folds; diagnostic scores are not research results |
| Archived numerical compatibility | All 900 original 8–30 Hz epochs exactly match the new canonical baseline branch, maximum absolute difference 0 uV |
| Archived experiment preservation | All 610 original result files and six frozen scientific source modules remain byte-identical to the input archive |
| Archived main inference | All 75 saved models reverified with the new package source; tests/archive_main_verify.txt |
| Development EEG | 80 actual participants, 3600 task trials; all passed the frozen structural QC |
| Separate resting references | R01 from all 80 development people; no task trial or label used for alignment |
| Base screening | All 224 specified configurations completed 3 participant-group folds, 672 evaluations, zero failed candidates |
| Screening fitted-model receipts | 672 fitted-model hashes and source trial/person receipts checked; 36 neural inner source-only splits checked; 783 metadata/history files preserved |
| Real-data capacity diagnostic | EEGNet, compact shallow and attention models each reached 100% on the SAME 32 source training examples in a separate diagnostic setting; zero held-out examples |
| Label-shuffle diagnostic | Ten within-person permutations with group CV: mean balanced accuracy 50.2696%, range 49.3884–51.2072%; no confirmation data |
| Runtime recovery | All four development-cache files and the separate-resting-reference array reconstructed byte for byte; the previously recorded selection SHA 256 reproduced exactly; frozen identities retained |
| Secondary validation and combinations | 22 individual candidates and 18 combinations completed on 20 separate development participants; the two selected pipelines and their rules are frozen in selection.json |
| New confirmation | 24 valid new participants, 1080 task trials; five requested participants excluded by the frozen structural annotation rule before scoring |
| Final prediction integrity | PASS: all 224 screening and 40 validation tables recomputed; all eight saved final constituents/references reinferred from real EEG; training-only scalers and single-example/batch agreement checked |
| Frozen combination rules | Eight source-OOF weighted/stacking rules refitted and matched; specified validation winners, epoch policies, seeds and final source membership unchanged |
| Primary confirmation BA | Selected zero-calibration pipeline 63.7384% [59.5314, 68.3754]%; CSP reference 57.4753%, EEGNet reference 60.0546%; same 24 people |
| Secondary confirmation BA | Separate-R01-rest-calibrated pipeline 68.7226% [63.4127, 74.3359]%; different acquisition requirement explicitly retained |
| Recovery software checks | 21 frontier unit checks and the isolated end-to-end/tampering fixture passed again after runtime recovery; the full 40-test record remains available |

The primary zero-calibration and secondary separate-rest-calibrated conditions
remain separate. Baseline EEGNet uses the new wrapper's gradient clipping at 5,
so this is a reference configuration based on the old settings rather than
byte-identical reproduction of old neural training. The canonical baseline
input signals and old engine itself are unchanged. Exact methods are in
docs/FRONTIER_METHODS.md. New frontier execution is Linux CPU; the user's
supplied Windows v0.2 result archive is preserved as its own separate run.

## Context/reflection extension — in progress

This new sequential study treats every previously inspected PhysioNet person
as development data. External transfer and frozen-algorithm external LOSO are
fixed before BNCI 2014_001 acquisition; their results are not asserted here yet.

| Check | Observed result |
|---|---|
| Full software suite | 60 passed, 5 original ONNX warnings; tests/context_test_results.xml and tests/context_test_run.txt |
| New software coverage | Reflection laws, per-trial covariance whitening, source role separation, past-only context extraction, checkpoint roundtrip, source risk weighting, and isolated synthetic end-to-end/tampering fixture |
| Real development input | 104 actual people, 4680 task/context pairs |
| Canonical reference compatibility | All 4680 archived 16-electrode 8–30 Hz late inputs match exactly, maximum difference 0 uV |
| Capacity controls | Three new filter-bank variants each fit the SAME 32 balanced source examples at 100% in a separate diagnostic setting; zero held-out examples |
| Label permutations | Ten within-source-person shuffles with group CV: mean BA 50.2855%, range 49.3378–51.3223%; no external or reused-selection people |
| Frozen study | 63 base configurations, three source group folds, fixed epochs/seeds and two distinct external protocols; source and prediction identities retained |
| Research execution | In progress; no new independent accuracy is claimed in this checkpoint |

The expanded current suite passes **67 tests** (five original ONNX warnings),
including the matched gate and session-disjoint calibration fixtures;
see tests/research_test_results.xml and tests/research_test_run.txt. The full
source screen completed all 63 configurations/189 folds. The registered
fixed/quality-conditioned follow-up completed another six OOF fits and both
three-seed reused-validation/final-source variants before external acquisition.
Its validation BA is 65.8418% for fixed gating and 61.8385% for conditioned
gating; these are explicitly previously inspected development people. The
supplementary primary calibration budget of 16 chronological session-T labels
was registered before any external data. New-dataset outcomes are pending.


## Verified direct external transfer and continued evaluation

All 65 base configurations completed 195 source-group fits;35 main validation
singles/combinations and both matched gates were retained. All 11 final source
constituents, including 13 neural repetitions, were fitted on 104 PhysioNet
people and frozen before external acquisition. Direct transfer contains all 9
BNCI people and 2592 real left/right trials. Independent transfer-only
reinference passed: source membership, training-only scalers, all probabilities,
decisions and summary/participant metrics. The selected task ensemble scores
73.6883% BA, versus 66.9367% for the EEGNet reference; paired difference 6.7515 pp,
95% descriptive participant interval[0.8102, 11.2269]. Reference/advanced
window, bands and ensembling differ; this is a whole-pipeline comparison.

The full software suite now passes 73 checks with the five original ONNX
warnings (tests/final_software_test_results.xml). Additional checks cover
scheduler equivalence/cancellation, the analytic kernel-logistic gradient,
positive-semidefinite fused kernels, source-only feature standardization and
invariance of personal fitted coefficients to future-session labels.

A supplementary mathematical residual-calibration specification preceded
external acquisition. Its implementation, 12 source C/feature choices and
80/104-person feature scalers were frozen before the first external scores.
Source selection uses first 16 run 4/8 labels and held run 12 data of previously
inspected development people. Primary hybrid and 16 labels were fixed; no
external outcome chooses them. All 4/8/16 budgets are reported on the same session-E trials. Full external LOSO
and personal evaluation are now complete; see the current evidence at the top.


## English release and expanded calibration (v0.4)

The exploratory session study compares 11 methods at six budgets on the same
1, 296 session-E trials. All 18 BNCI raw hashes and five processed fingerprints
match the original study. Independent verification replayed and refitted 540
saved artifacts, reconstructed 54 donor-only choices and checked 85, 536 actual
prediction rows. All 330 method-pair/budget means were checked against class
recalls. The fixed primary 16-label gain remains uncertain;144-label performance
requires the complete first-session support. See the full latest report.

The local suite passed 82 software checks. All eight report pages passed both
1280 px and 390 px browser checks, including image loading, horizontal layout,
English document language and the 11-method budget filter. All human-readable
project files were scanned for remaining localized script. Archived scientific
modules, binary model/scaler artifacts and numerical CSV tables are byte-exact.
Presentation changes are audited in tests/language_migration.json; current
release evidence is in tests/english_release_validation.json. GitHub-hosted CI
status is separate from these local results.


## v0.5 corrected covariance study

All 94 local software tests passed, including the negative-log-eigenvalue matrix
exponential regression. Independent verification replayed/refitted 972 final
artifacts, reconstructed 108 nested donor choices and recomputed 7, 668 candidate
metrics. Exact 16/22-channel input hashes were frozen before corrected E scoring.
The initial power-mean implementation was invalidated and retained for audit;
no incomplete initial model or score is represented as a verified result.
All scientific tables refer to the freshly refitted corrected output.
