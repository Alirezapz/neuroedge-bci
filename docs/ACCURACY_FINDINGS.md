# Findings from the archived accuracy investigation

The initial near-chance EEGNet result prompted training/validation diagnostics,
a real-training-data memorization control, longer bounded development and new
participant confirmation. Training accuracy is a diagnostic, not generalization.
All earlier model artifacts and predictions remain available.

Longer training on the small source cohort did not beat matched baselines.
The expanded70-source study gave CSP57.5% with20 source people and61.2% with70
sources on a common holdout; the paired descriptive gain interval included zero.
Its neural baseline scored62.3%. These are limited pilot findings; they do not
support a strong BCI accuracy claim or a causal dataset-size conclusion.

See `runs/accuracy_v2/REPORT.md`, `runs/accuracy_expanded/REPORT.md`, and
`ACCURACY_NEXT.md`. Later frontier and independent-dataset studies broaden the
comparison and retain these adverse results. The latest study separates
personal support requirements explicitly; no single headline percentage
describes all training and evaluation conditions.
