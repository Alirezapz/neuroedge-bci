# NeuroEdge BCI research model card

## Purpose and provenance

NeuroEdge BCI is a reproducible offline motor-imagery EEG research project,
maintained by Alireza Pazouki with Codex assistance. The release preserves
multiple historical studies, exact saved checkpoints, prediction tables,
split manifests, negative results and independent verification receipts.
The archive is version 0.5.0; publication packaging does not retrain its models.

The source datasets are PhysioNet EEGMMIDB 1.0.0 and BNCI2014_001. Source URLs,
file hashes, preprocessing and run-specific acquisition records remain in the
archive. Dataset terms and attributions are in [DATA_LICENSES.md](DATA_LICENSES.md).

## Latest evaluation

The corrected covariance study evaluated 71 fixed configurations in 7,668
donor fits, then saved 972 selected model/stack artifacts. It evaluates the
same 1,296 session-E left/right trials from nine BNCI participants. Personal
training and donor selection use session T. All E labels are excluded from
fitting and parameter selection.

| Pipeline at 144 personal session-T labels | EEG sensors | Unlabeled full E batch used | Balanced accuracy |
|---|---:|---|---:|
| Previous donor-selected fusion | 16 | No | 85.11% |
| Corrected donor stack, primary | 16 | No | 84.95% |
| Corrected donor stack, secondary | 22 | No | 86.50% |
| Equal probability mean, secondary | 22 | Yes | 87.04% |

The matched primary comparison did not improve: -0.15 percentage points,
descriptive interval [-1.31, +1.16]. More electrodes change the input
requirement. The full-batch condition accesses future unlabeled E trials and
is transductive. Every row requires personal labeled calibration; 144 is the
complete first-session binary calibration set. This previously inspected
dataset makes the study exploratory. Secondary intervals are descriptive and
unadjusted for multiple comparisons. Nine people remain the sampling unit.

## Verification and limitations

The archived independent verifier replayed and refitted all 972 final
artifacts, reconstructed 108 nested settings and checked 7,668 donor metrics
and 217,728 actual E prediction rows. The archived software suite passed
94 tests. These completed experiment checks are separate from GitHub CI or
publication-packaging checks. An earlier prototype matrix-exponential error
invalidated an initial diagnostic run; that run is labeled and all final
results were recomputed under the corrected protocol.

Historical zero-calibration, LOSO and personal-calibration conditions are
reported separately. This release establishes no clinical utility, live
brain control, causal real-time accuracy, MCU performance, measured energy
savings or universal state-of-the-art result. The compact adaptations and
project-specific combinations build on credited established methods.

## Access and reuse

Run `python scripts/restore_model_bundle.py` after cloning to restore exact
artifacts. Run with `--verify-only` to check archive bytes without extraction.
The script needs only Python's standard library. Replaying a trained model
also requires the pinned scientific environment and the correct run protocol.
Serialized joblib/pickle and PyTorch models must be loaded only from a trusted
source. The restoration script validates bytes without deserializing models.

MIT covers the project's own contributions. It does not replace rights in
training data, retained third-party feature content or task annotations.
See [DATA_LICENSES.md](DATA_LICENSES.md) for the exact scope and upstream terms.
