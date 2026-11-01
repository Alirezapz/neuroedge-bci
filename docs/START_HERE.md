# Start here

NeuroEdge is a reproducible offline motor-imagery EEG research project. It
classifies imagined left/right hand movement and studies generalization,
calibration requirements, channel ablation, and measured CPU deployment costs.

Open `runs/covariance_lab_corrected/results.html` for the latest expanded comparison.
The earlier small-label study is in `runs/session_adaptation/results.html`.
Open `runs/research_summary/results.html` for the earlier independent-dataset
transfer, external LOSO and small-label personalization results. Both work
offline without retraining. Every study's `REPORT.md` contains its own methods,
all numerical results, negative findings and limitations.

## Read in this order

1. `README.md`: current findings and reproduction entry points.
2. `docs/COVARIANCE_LAB_METHODS.md`: exact latest hypothesis, input conditions and isolation.
3. `docs/CONTEXT_METHODS.md`: the original independent external evaluation.
4. `docs/RELATED_WORK.md`: prior work and defensible contribution.
5. `docs/SETUP_WINDOWS.md`: local environment setup.

## Explain the work accurately

The contribution is a tested research implementation, participant/session
isolation, explicit source selection, complete evidence and independent replay.
The code was developed with Codex assistance. Be prepared to explain the data
mapping, covariance features, EEGNet, regularization and experimental limits.
Do not describe every component as a new invention.

Be able to explain why random trial splits can overestimate unseen-person
performance, why personal calibration has a different supervision requirement,
why test labels cannot choose model settings, and why seeds are not independent
participants. Explain balanced accuracy as the mean recall across the two
classes; participant-mean scores give each person equal weight.

Archived demonstrations replay stored signals and predictions. They do not
connect to an EEG device or perform new browser inference. Filtering is offline;
CPU model timings omit acquisition and preprocessing. INT8 did not automatically
improve speed or artifact size. These are research results, not clinical or
real-time-device validation. There is no degree-level claim attached to the
number of experiments.
