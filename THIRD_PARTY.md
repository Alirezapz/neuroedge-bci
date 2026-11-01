# Data, references, and attribution

NeuroEdge source code is independently implemented and released under MIT.
See [DATA_LICENSES.md](DATA_LICENSES.md) for the software/artifact license scope,
full copyright notice and dataset-specific terms. The dataset and its derivatives do not become MIT-licensed by inclusion in this project.

## EEG dataset

- Schalk, G. (2009). EEG Motor Movement/Imagery Dataset, version 1.0.0. PhysioNet. DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P).
- Original acquisition system: Schalk et al. (2004), BCI2000. DOI: [10.1109/TBME.2004.827072](https://doi.org/10.1109/TBME.2004.827072).
- PhysioNet platform citation requested on the current dataset page: Pollard et al. (2026), PhysioNet as a global platform for biomedical research. DOI: [10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).
- Dataset page: [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/).
- Dataset license: [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).

Raw EDF files are downloaded from the official public physionet-open S3 bucket or the canonical PhysioNet host and are omitted from the source deliverable. Historical owner-delivered demo pages embedded filtered/resampled/selected/average-referenced derivative epochs with dataset attribution. Recorded-signal replay pages are excluded from this GitHub publication. Dataset rights and attribution obligations remain applicable to any locally reconstructed derivatives.

## Architecture and software references

The layer sequence was checked against [Lawhern's official EEGModels.py](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py) and the EEGNet paper. This is an independent PyTorch adaptation, not a copy of the Keras source and not validated by the original authors. See docs/METHODS.md for differences. The upstream repository uses its own CC0/Apache licensing terms.

Other primary implementation references:

- [MNE CSP motor-imagery example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html).
- [MNE EEGBCI run meanings](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html).
- [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

Scientific dependencies retain their respective upstream licenses. Installed versions are recorded with each run. No publication, affiliation, clinical validation, or state-of-the-art result is implied.

## Frontier extension references

- Schirrmeister et al. (2017), *Deep learning with convolutional neural networks
  for EEG decoding and visualization*, DOI 10.1002/hbm.23730,
  https://arxiv.org/abs/1703.05051.
- Song et al. (2023), *EEG Conformer: Convolutional Transformer for EEG Decoding
  and Visualization*, DOI 10.1109/TNSRE.2022.3230250,
  https://github.com/eeyhsong/EEG-Conformer.
- He and Wu (2020), *Transfer Learning for Brain–Computer Interfaces: A Euclidean
  Space Data Alignment Approach*, https://arxiv.org/abs/1808.05464.
- pyRiemann's official TangentSpace/MDM implementations and documentation:
  https://pyriemann.readthedocs.io/.

The new shallow and attention networks are independently implemented compact
adaptations, with depthwise spatial convolutions and reduced attention depth;
they are not copies or exact reproductions of the upstream networks. The
secondary alignment condition adapts inverse-covariance whitening to a separate
eyes-open baseline and explicitly records its additional calibration requirement.

## Context and reflection extension

The independent compact filter-bank variance network is inspired by Mane et al.,
FBCNet (https://arxiv.org/abs/2104.01233). It uses six bands, eight spatial
filters/band, four variance segments and explicit task/pre-cue sharing; it is
not an exact reproduction of the published architecture or training study.
Channel-and-label reflection follows prior MI augmentation work, including
Hoshino et al. (2024), https://doi.org/10.1109/EMBC53108.2024.10782028 .
ERD/ERS contrasts build on Pfurtscheller and Lopes da Silva (1999),
https://doi.org/10.1016/S1388-2457(99)00141-8 . Source participant risk weighting
is inspired by Sagawa et al., https://arxiv.org/abs/1911.08731 . These ideas
are credited rather than claimed as new inventions.

The external dataset BNCI2014_001 (BCI Competition IV dataset 2a) is provided
by the Institute for Knowledge Discovery, Graz University of Technology under
CC BY-ND 4.0: https://bnci-horizon-2020.eu/database/data-sets . The original
description is Brunner, Leeb, Mueller-Putz, Schloegl and Pfurtscheller,
https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf .
The MATLAB timing, units and channel mapping are checked against the official
MOABB loader documentation. The binary offline transfer/LOSO experiment is
different from the original four-class competition and its causal evaluation.
Raw and processed external EEG are not redistributed in the archive. Source
URLs/checksums, model artifacts and research predictions are retained. The
dataset is not relicensed under the source code's MIT license.

# Personal residual kernels

The supplementary personal-calibration code independently implements a
regularized kernel-logistic residual around the frozen source ensemble.
Gaussian kernels, logistic regularization and matrix logarithms of SPD
covariances are established tools. Arsigny et al. (2006),
https://doi.org/10.1002/mrm.20965, provides background on Log-Euclidean matrix
geometry in diffusion imaging; it is not an EEG accuracy comparator. The
present feature-wise source standardization defines its own coordinate metric.
No claim of inventing these principles or reproducing that paper's experiments
is made. NumPy/SciPy supply linear algebra and optimization; original dependency
licenses remain applicable.

Related EEG work is credited in [the English related-work note](docs/RELATED_WORK.md):
Zhang and Wu's MEKT (10.1109/TNSRE.2020.2985996), Gaur et al.'s LR-TSTL
(10.1109/TCDS.2021.3099988), Azab et al.'s source-regularized logistic transfer
(10.1109/TNSRE.2019.2923315), and Wu's wAR (10.1109/THMS.2016.2608931).
The source-prior and covariance ideas have prior art. Those four algorithms
were not copied or reproduced as external comparators in this package; their
published percentages are not directly compared with NeuroEdge's different
protocols. wAR's cited experiments are ERP, not this binary MI evaluation.

