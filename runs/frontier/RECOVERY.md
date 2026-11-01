# Execution recovery

The runtime was reset after selection and during final source-only fitting. The durable archive preserved all 224 screening results, 783 fit-history/receipt files and 13 completed secondary-validation configurations. Temporary raw/processed caches and incomplete final fits were not retained.

The selection metadata observed before the reset were reconstructed from the recorded tool output and checked byte-for-byte against its previously recorded SHA256: 2e98bfdccb7649590f628b60b88a7768980f62296f897fe31ba3ada25edda3d7. They are stored as selection_observed_before_reset.json. This file anchors the already-chosen pipelines; it is not a newly tuned selection.

The four development-cache files and the separate resting-reference array were reconstructed with the unchanged frozen preparation code. All original checksums match exactly; see recovery_data_checks.json and recovery_rest_check.json. The remaining nine rest-calibrated validation fits and all eighteen combinations were regenerated. The resulting selection.json matches the original selection SHA256 exactly, as recorded in selection_recovery_check.json. Existing completed predictions, manifests and frozen source identities are preserved. Final source-only fitting resumes before confirmation acquisition; no confirmation result is used to recover or change selection.
