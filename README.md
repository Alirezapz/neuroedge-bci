# NeuroEdge

Reproducible offline EEG motor-imagery research: source code and aggregate results.
This GitHub edition excludes model checkpoints, individual-trial records, raw
EEG, scientific image payloads and recorded-signal replays. The complete project
bundle retains those exact artifacts for its owner. Binary/data publication
remains blocked by automated approval review pending explicit payload approval.

## Latest verified experiment

71 configurations were evaluated in 7,668 donor fits, followed by 972 selected
model/stack artifacts. The same 1,296 session-E left/right trials from nine
BNCI2014_001 people were scored. All parameter selection used session T only.
The conditions below all use 144 personal first-session left/right labels.

| Pipeline | EEG sensors | E-batch statistics | Balanced accuracy |
|---|---:|---|---:|
| Previous fusion | 16 | None | 85.11% |
| New donor stack, primary | 16 | None | 84.95% |
| New donor stack | 22 | None | 86.50% |
| New equal-mean ensemble | 22 | Unlabeled full E batch | 87.04% |

The primary comparison did not improve: -0.15pp [-1.31,+1.16]. The 22-sensor
strict stack gains 1.39pp [0.08,2.85] versus the old 16-sensor pipeline; this
also changes the electrode requirement. The 87.04% condition uses future
unlabeled E trials for alignment. All results require personal calibration.
These previously inspected data make this an exploratory study. Intervals
are descriptive and unadjusted for secondary comparisons. Gains remain limited.

- [Latest full aggregate report](runs/covariance_lab_corrected/REPORT.md).
- [All 168 method/input/budget conditions](runs/covariance_lab_corrected/summary.csv).
- [All 1,092 aggregate comparisons](runs/covariance_lab_corrected/all_pairwise_comparisons.csv).
- [Exact methods and limitations](docs/COVARIANCE_LAB_METHODS.md).
- [Earlier calibration report](runs/session_adaptation/REPORT.md).
- [Attribution and licenses](THIRD_PARTY.md).

94 local software tests passed. Independent verification replayed/refitted
all 972 final artifacts, reconstructed 108 nested settings, and checked all
7,668 donor candidate metrics and 217,728 actual prediction rows. Nine people
remain the sampling unit. An initial diagnostic implementation was invalidated
after a matrix-exponential prototype error; all final results were recomputed
under the corrected protocol. Negative findings are preserved.

## Install the source

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

Archived checkpoint replay requires the complete artifact bundle. Fresh source
training uses new named configurations and the attributed public datasets;
see the method guides. Never remove hash guards to force incompatible resumes.
All project text is English. Source code is MIT; dataset terms are separate.
The code was developed with Codex assistance. This offline study establishes no
clinical, real-time, MCU or universal-SOTA claim. See PUBLISHING.md for scope.
