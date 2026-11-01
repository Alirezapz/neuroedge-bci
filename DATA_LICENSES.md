# License scope and dataset attribution

Copyright (c) 2026 Alireza Pazouki and NeuroEdge contributors.

The full [MIT license](LICENSE) applies to the project's original source code,
documentation, figures, learned parameter contributions and original research
outputs, to the extent that the project holds the relevant rights. It does
not replace licenses or other rights in third-party material. The package
metadata's `MIT` identifier describes the software, not an EEG dataset license.

| Material | Terms and attribution |
|---|---|
| Original NeuroEdge code and original project contributions | MIT; retain the copyright and permission notice |
| PhysioNet EEG Motor Movement/Imagery Dataset 1.0.0 | ODC Attribution 1.0; Gerwin Schalk and the credited acquisition team |
| BNCI2014_001 / BCI Competition IV dataset 2a | CC BY-ND 4.0; Institute for Knowledge Discovery, Graz University of Technology |
| Dependency code | Each dependency's upstream license; see the pinned environment and THIRD_PARTY.md |
| Retained third-party content within a model or result table | The original applicable rights and dataset terms remain; the project cannot grant additional rights in that content |

## PhysioNet

Schalk, G. (2009). *EEG Motor Movement/Imagery Dataset*, version 1.0.0.
PhysioNet. DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P).

Dataset: https://physionet.org/content/eegmmidb/1.0.0/

Database license: https://opendatacommons.org/licenses/by/1-0/

The database is made available under the Open Data Commons Attribution License
v1.0. Keep the source attribution and database-license notice with public
research outputs produced from it. Cite Schalk et al. (2004), BCI2000,
DOI 10.1109/TBME.2004.827072, and the current PhysioNet platform citation listed
in [THIRD_PARTY.md](THIRD_PARTY.md). This project filters, resamples, selects
electrodes and references recordings locally; processed recording caches and
recorded-signal replay pages are excluded from this GitHub publication.

## BNCI / Graz

Dataset and licensor notice:
https://bnci-horizon-2020.eu/database/data-sets

Original dataset description:
https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf

License and legal text:
https://creativecommons.org/licenses/by-nd/4.0/
https://creativecommons.org/licenses/by-nd/4.0/legalcode.en

The dataset remains CC BY-ND 4.0. That license permits sharing the licensed
material with attribution but does not grant permission to share Adapted
Material. NeuroEdge publishes independently implemented analysis code and its
research evidence; it does not redistribute the original MATLAB recordings
or filtered/resampled recording arrays. Its binary left/right tasks and
offline preprocessing differ from the original four-class competition.

Model serialization can retain support features, covariance summaries or other
training-derived material; a model file is not automatically free of dataset
rights. Neither the root MIT license nor this notice relicenses EEG samples,
source annotations, retained feature arrays or other third-party content.
Obtain any additional permission required by the upstream rights holder for a
proposed redistribution of such material. No upstream endorsement is implied.

## Artifact provenance

The archive manifest identifies each restored model, prediction table and
audit record by size and SHA-256. These are the original saved experiment
bytes, not newly generated predictions. Participant identifiers are the
public datasets' pseudonymous research identifiers. Prediction tables may
retain task labels for scientific verification; those labels are not
relabeled as MIT dataset content.

See [ARTIFACTS.md](ARTIFACTS.md), [MODEL_CARD.md](MODEL_CARD.md),
[THIRD_PARTY.md](THIRD_PARTY.md) and [REFERENCES.bib](REFERENCES.bib).
