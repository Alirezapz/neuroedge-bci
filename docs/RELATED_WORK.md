# Related work and research contribution

This is a targeted review of nearby primary work, not a systematic proof that
no similar method exists. Published percentages cannot be compared directly
when participants, classes, channels, windows or target-data access differ.

| Primary work | Connection | Implementation or evaluation distinction |
|---|---|---|
| [Mane et al., FBCNet (2021)](https://arxiv.org/abs/2104.01233) | Filter banks, spatial filtering and variance pooling | This project uses a compact independent adaptation, not an exact reproduction. |
| [Hoshino et al., channel-and-label flip](https://pubmed.ncbi.nlm.nih.gov/40039924/) | Swap left/right channels and labels | Augmentation, equivariance and no-flip controls are compared; flipping is prior art. |
| [Zhang and Wu, MEKT](https://pubmed.ncbi.nlm.nih.gov/32286993/) | Covariance geometry and cross-person adaptation | MEKT uses unlabeled target data; the original direct-transfer arm fits no target-task statistics. MEKT itself was not reproduced here. |
| [Gaur et al., LR-TSTL](https://repository.essex.ac.uk/30780/) | Tangent features and logistic transfer | Tangent logistic classification is established; the original personal kernels instead use source-scaled matrix-log coordinates and a source-logit residual. |
| [Azab et al., weighted transfer learning](https://pubmed.ncbi.nlm.nih.gov/31217122/) | Low-label personal adaptation and source regularization | Their parameters are regularized toward similar-person classifiers; our residual corrects a fixed source ensemble. Their method was not reproduced here. |
| [Wu, weighted adaptation regularization](https://arxiv.org/abs/1702.02897) | Reduced target-label requirements | The cited main task is ERP, not a matched MI benchmark for this project. |
| [Bleuze, Mattout and Congedo (2022)](https://doi.org/10.3389/fnhum.2022.1049985) | Covariance recentering, tangent representation and transfer | The new calibration study uses support-local whitening and donor-only parameter selection; it is not a reproduction of TSA. |

[Arsigny et al. (2006)](https://doi.org/10.1002/mrm.20965) supplies mathematical
background for matrix-log geometry. That paper concerns diffusion tensors,
not EEG results. Standardizing log-covariance coordinates defines a particular
feature metric; it is not the classic affine-invariant distance.

The original study asks whether complementary spectral, spatial and neural
representations generalize across recording datasets with source-only choices.
The pre-cue/gating studies test extra preceding EEG. The personal-kernel study
tests a small nonlinear correction versus the same-budget intercept. A failed
primary comparison remains a failed comparison even when a secondary variant
has a higher observed mean.

The latest extension tests support-local linear tangent/dual residuals and a
donor-selected fusion across six labeled-support budgets. These are proposed
project-specific combinations of established components. The dataset was
previously inspected, so this extension is explicitly exploratory. Increasing
calibration to a complete first session changes the required supervision.

The defensible contribution is a reproducible comparative study with saved
models, actual predictions, exact splits, negative results and independent
checks. The experiments do not establish worldwide algorithmic novelty,
superiority over unreproduced named methods, publication acceptance, or SOTA.
A stronger future claim needs matched implementations of nearby methods and
an independent dataset that has not already guided development.
