# Protocol and implementation decisions

The main protocol is an engineering reproduction and controlled comparison, not a newly invented EEG method. The configuration is frozen locally before evaluation; this is not a public preregistration. The first 20 dataset participant identifiers were selected for a bounded CPU experiment before any test results were inspected.

## Data and task

Use EEGMMIDB version 1.0.0 from [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/). Runs 4, 8 and 12 are imagined left/right fist movement. T1 maps to class 0 (left), T2 to class 1 (right), and T0 rest is excluded. The meanings differ in runs 6/10/14, which are deliberately rejected by this implementation. Data retain the original recording provenance and SHA-256 checksums; those checksums record acquired bytes, not independent upstream signatures.

Audit channel names, sample rates, finite signal values, event durations, and class counts before training. Malformed or missing recordings stop the run with an audit reason. Subjects are not silently replaced based on data availability or model performance.

Apply a fixed fourth-order Butterworth band-pass (8–30 Hz), forward/backward zero-phase, to each recording independently. Resample to 128 Hz. Extract exactly one 384-sample epoch, [1,4) seconds after each imagery cue; the initial second is omitted to reduce immediate cue-related effects. The epoch must fit within the annotated task duration. Signals are stored in microvolts. There are no overlapping window augmentations and no learned artifact rejection.

This filter and cue-aligned epoch protocol are offline. They do not demonstrate causal streaming, asynchronous intention decoding, or control of a physical device. Sampling rates and cue/event mappings are audited rather than inferred from filenames alone.

## Predefined electrode comparison

The 8 electrodes are FC3, FC4, C3, Cz, C4, CP3, CPz, CP4. The 16 set adds FC1, FC2, C1, C2, CP1, CP2, C5, C6. These fixed symmetric sets emphasize the sensorimotor region and are nested within the recorded 64-channel set. They are not optimized on test accuracy and are not claimed to be optimal.

Each configuration is average-referenced using only its own available electrodes. Referencing to all 64 electrodes before selecting eight would secretly retain information from unavailable electrodes; the unit tests prevent that error. Band-pass filtering and resampling act independently on each channel. This ablation removes access to stored channels; it does not measure power savings of a physical acquisition system.

## Evaluation

Split unique subject identifiers into five shuffled outer folds with seed 20261029. Each subject is tested exactly once. In every fold, 4 subjects form the outer test set, 4 of the remaining subjects form validation, and 12 form training. This is group five-fold evaluation, not leave-one-subject-out. Both the neural and classical methods use the same 12 training subjects. Validation is reserved for EEGNet early stopping; fixed classical hyperparameters are not optimized there. There is no refit on the validation subjects before test scoring.

Fit all learned preprocessing, CSP spatial filters, LDA parameters, and EEGNet per-channel scaling on the training split only. The common-average reference and fixed filter contain no fitted population statistics. EEGNet validation-loss checkpoint selection never inspects outer-test metrics. Model/channel configurations and epoch budget are fixed before testing.

Report balanced accuracy and macro-F1 per held-out participant, per seed. Average EEGNet repetitions within participant first, then average across participants. The 95% percentile bootstrap resamples participants, not EEG windows or seeds. Paired channel comparisons resample within-participant differences. These are descriptive intervals conditional on the fitted fold models: the procedure does not retrain each bootstrap sample and does not account for every dependence between overlapping cross-validation training sets. They characterize a small fixed cohort. They do not prove equivalence or non-inferiority; no equivalence margin or confirmatory power calculation was specified. Confusion matrices average complete test counts across seeds to avoid implying three times as many independent trials.

## Models

- Dummy: training-class prior, deterministic argmax.
- CSP+LDA: MNE CSP with four components, fixed shrinkage 0.1, log average-power features; LDA uses lsqr and automatic shrinkage. CSP is fitted inside each training fold.
- EEGNet-8,2: 8 temporal filters, depth multiplier 2, 16 separable outputs; depthwise spatial convolution, ELU, average pooling 4 then 8, ordinary dropout 0.5. The input is N×1×C×384, followed by 2 output logits. Cross-entropy handles softmax during training. The 32-sample first kernel follows the authors' shorter-kernel guidance for high-passed SMR data. Batch normalization uses eps=0.001 and PyTorch momentum=0.01, corresponding to a 0.99 running-statistic retention convention. Spatial filter max norm is 1.0; classifier max norm is 0.25, applied after each optimizer step. This is an independently written PyTorch adaptation checked against the authors' layer sequence, not an assertion of identical published training results.

Train EEGNet using Adam at 0.001, PyTorch default beta/epsilon settings, a maximum of 25 epochs, and validation-loss early stopping with patience 8. We use ordinary shuffled trial minibatches and no class weighting. Seeds 7,17,27 are fixed repetitions; deterministic CPU operations are requested. Preprocessing differs from some published protocols, so published accuracy numbers must not be compared directly.

## Deployment

Export only seed 7, selected before evaluation, for every outer fold and electrode configuration. Verify FP32 probability agreement against PyTorch on corresponding held-out trials (maximum absolute error ≤1e-4 and identical argmax). Calibrate per-channel signed INT8 QDQ Conv/Gemm/MatMul quantization on at most 64 training trials, chosen by a fixed permutation. The calibration artifact records exact trial identifiers. Some layers remain floating point.

Benchmark batch-one ONNX Runtime CPUExecutionProvider sessions with one intra-op and one inter-op thread, 30 warm-up calls and 200 timed calls per model. Alternate the variant measurement order across folds. Store every timing and actual file size. These are model-only measurements including Python session.run overhead; they exclude source reading, filtering/resampling, reference/normalization, and signal acquisition. No actual MCU or energy measurements are claimed. Size or speed improvements are not assumed in advance.

## References

See REFERENCES.bib and THIRD_PARTY.md. Nearby prior work includes Wang et al. (2020), MI-BMInet (2024), and Tuimebay et al. (2026). A future claim of novelty requires a separate literature review and a distinctive contribution.

