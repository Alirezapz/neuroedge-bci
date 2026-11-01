# Archived original benchmark results

The fixed20-person PhysioNet cohort supplies900 three-second left/right imagery
epochs,75 model artifacts and13,500 test prediction rows across models/seeds.
There are20 independent people, not13,500 independent participants.

| Method | 8 channels | 16 channels | 64 channels |
|---|---:|---:|---:|
| Training class-prior reference | 50.0% | 50.0% | 50.0% |
| CSP + LDA | 54.6% | 54.8% | 51.9% |
| EEGNet, mean of three seeds | 50.3% | 51.9% | 50.6% |

Values are participant-mean balanced accuracy. EEGNet remained near chance;
the classical result is insufficient for a high-performing BCI claim. The
largest predefined mean is descriptive, not a test-selected deployment model.
Full intervals and paired comparisons are in `runs/main/REPORT.md` and CSVs;
the16-channel CSP interval is approximately50.7–59.3%. Overlapping intervals
do not establish equivalence between sensor configurations.

All15 predefined first-seed FP32 exports matched PyTorch decisions and all15
INT8 QDQ variants executed with training-only calibration. INT8 was slower on
this CPU. Example8-channel mean model median latency: FP32 approximately0.091ms,
INT8 approximately0.307ms. Its artifact grew from12,209 to approximately13,729
bytes. The64-channel artifact shrank slightly, but inference was still slower.
No separate profiling experiment established the precise cause of overhead.

These timings measure model execution only and omit acquisition of the
three-second EEG window and preprocessing. No MCU or physical EEG hardware
was tested. All75 stored models reproduced probabilities/decisions; the original
software suite passed13 checks. Participant separation, training-only scaling,
raw/data/code hashes and quantization identities were checked. Browser replay
uses stored predictions. These checks are software evidence, not clinical
validation or proof that every possible error is absent.

Later studies are in README.md. They preserve this original limited result,
investigate learning using training/validation only, and keep personal
calibration and independent-dataset evaluation separate.
