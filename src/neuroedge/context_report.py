"""Read-only derived reports for the predeclared context/symmetry study."""
from __future__ import annotations
import base64
import html
import json
import numpy as np
import pandas as pd
from .config import digest
from .context_protocol import freeze, grid, interval, subject_scores
NAMES = {'csp_reference': 'CSP reference', 'eegnet_reference': 'EEGNet reference', 'selected_zero_calibration': 'Selected: trial only', 'selected_pre_cue': 'Selected: pre-cue context'}
FA = NAMES.copy()

def report(plan):
    out, identity = freeze(plan)
    verification = json.loads((out / 'verification.json').read_text())
    if verification['status'] != 'PASS':
        raise RuntimeError('Independently verify before reporting.')
    selection = json.loads((out / 'selection.json').read_text())
    screen = pd.read_csv(out / 'screening_summary.csv')
    validation = pd.read_csv(out / 'validation_summary.csv')
    comparisons = []
    summaries = {}
    per_subject = {}
    lines = ['# Pre-cue contrast, reflection and source-only fusion', '', 'A hypothesis-driven sequential exploratory study. The previously inspected PhysioNet people are development data. Neither their cross-validation nor reused validation is an independent new confirmation. All methods and external protocols were fixed before external acquisition.', '', f"{len(screen)} base configurations completed {len(screen) * plan['search']['folds']} participant-group source fits. Source development: {len(plan['source_subjects'])} people; reused selection: {len(plan['selection_subjects'])} people. {len(validation)} singles/combinations were compared for selection. Three fixed seeds are averaged for advanced neural validation/final fits; the historical neural reference retains seed 7.", '', '## Input conditions', '', 'Every model uses the same 16 specified sensorimotor electrodes and their own common average reference. The trial-only condition requires [0.5,4) seconds of imagery after the cue. The context condition additionally requires [-1.75,-0.25) seconds of preceding rest from that same trial, not a task-batch estimate or an R01 recording. Task EEG uses the historical offline zero-phase bandpass. Pre-cue EEG uses TWO forward causal fourth-order Butterworth passes, matching the squared magnitude response, followed by resampling of the bounded past segment only. Its phase differs from task filtering. A software perturbation check ensures that task/future samples cannot change extracted context. The complete decoder is still offline.', '', 'The CSP and EEGNet references retain the 8–30 Hz, [1,4)-second canonical input. Advanced methods use [0.5,4) seconds and different bands/representations. Matched electrodes do not isolate architecture from these other pipeline changes.', '', '## Frozen choices', '']
    for arm, winner in selection['selected'].items():
        details = winner.get('spec', winner.get('combination'))
        lines += [f"- {arm}: `{winner['name']}`, reused-development BA {winner['balanced_accuracy'] * 100:.2f}%.", '  `' + json.dumps(details, ensure_ascii=False, sort_keys=True) + '`']
        if winner['kind'] == 'single':
            lines.append(f"  Source-OOF probability temperature: {winner['temperature']:.6g}.")
    lines += ['', '## Two separately defined external evaluations', '', 'External dataset: BNCI2014_001 / BCI Competition IV-2a, all nine people and both sessions. Only left/right hand trials are retained, using the supplied class mapping 1→left and 2→right; feet/tongue are outside this binary task. The MATLAB trial index is 1-based trial start, with the imagery cue two seconds later. The 16 matching EEG channels are selected from the known 22-channel montage. EOG channels never enter a model. All finite left/right trials are retained, including flagged artifacts; flags are preserved for descriptive error analysis. This is not a reproduction of the original four-class competition or its causal scoring procedure.', '', '1. **Direct transfer:** base models are fitted on all 104 PhysioNet development people; no BNCI task data or label fits a transform or model.', '2. **Frozen-algorithm external LOSO:** the already-selected base algorithms refit on the other eight BNCI people. Both sessions of the held person are excluded. Hyperparameters, epochs, seeds, ensemble coefficients and temperature remain fixed from PhysioNet; BNCI scores never select a new pipeline. The training condition is different from direct transfer.', '']
    for protocol, title, title_fa in [('external_transfer', 'Direct cross-dataset transfer', 'Direct transfer from PhysioNet to the external dataset'), ('external_loso', 'Frozen-algorithm external leave-one-person-out', 'External LOSO: train on the other eight people')]:
        summary = pd.read_csv(out / protocol / 'summary.csv')
        per = pd.read_csv(out / protocol / 'metrics_by_subject.csv')
        summaries[protocol] = summary
        per_subject[protocol] = per
        frames = pd.read_csv(out / protocol / 'predictions.csv')
        lines += ['## ' + title, '', '| Pipeline | People | Balanced accuracy | 95% participant interval | Accuracy | Log loss |', '|---|---:|---:|---:|---:|---:|']
        for row in summary.itertuples():
            lines.append(f'| {NAMES[row.model]} | {row.subjects} | {row.balanced_accuracy * 100:.2f}% | [{row.ci_low * 100:.2f}, {row.ci_high * 100:.2f}]% | {row.accuracy * 100:.2f}% | {row.log_loss:.4f} |')
        pivot = per.pivot(index='subject', columns='model', values='balanced_accuracy')
        for selected, baseline in [('selected_zero_calibration', 'csp_reference'), ('selected_zero_calibration', 'eegnet_reference'), ('selected_pre_cue', 'selected_zero_calibration')]:
            delta = pivot[selected] - pivot[baseline]
            low, high = interval(delta, plan)
            comparisons.append({'protocol': protocol, 'selected': selected, 'baseline': baseline, 'subjects': len(delta), 'mean_difference': float(delta.mean()), 'ci_low': low, 'ci_high': high, 'same_input_requirement': selected != 'selected_pre_cue'})
        from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
        errors = []
        for (model, person), group in frames.groupby(['model', 'subject']):
            matrix = confusion_matrix(group.label, group.prediction, labels=[0, 1])
            errors.append({'model': model, 'subject': person, 'left_recall': matrix[0, 0] / matrix[0].sum(), 'right_recall': matrix[1, 1] / matrix[1].sum(), 'roc_auc': roc_auc_score(group.label, group.prob_right), 'brier_score': brier_score_loss(group.label, group.prob_right), 'single_class_prediction': group.prediction.nunique() == 1})
        pd.DataFrame(errors).to_csv(out / protocol / 'error_analysis_by_subject.csv', index=False)
        lines += ['', f'Actual external task trials: {frames.trial_id.nunique()}; people: {frames.subject.nunique()}. Intervals resample people, not sessions, windows or seeds. LOSO intervals are conditional descriptions and do not account for every dependence from overlapping training cohorts.', '']
    pd.DataFrame(comparisons).to_csv(out / 'paired_differences.csv', index=False)
    lines += ['## Paired participant differences', '', '| Protocol | Selected | Reference | Difference (pp) | 95% paired interval | Same input |', '|---|---|---|---:|---:|---|']
    for row in comparisons:
        lines.append(f"| {row['protocol']} | {row['selected']} | {row['baseline']} | {row['mean_difference'] * 100:+.2f} | [{row['ci_low'] * 100:+.2f}, {row['ci_high'] * 100:+.2f}] | {row['same_input_requirement']} |")
    ablations = []
    specs = grid()
    for field in ['symmetry', 'context', 'dro', 'mode']:
        pairs = {}
        for spec in specs:
            if field not in spec:
                continue
            ignored = {'name', 'arm', field} | ({'context'} if field == 'mode' else set())
            key = digest({k: v for k, v in spec.items() if k not in ignored})
            pairs.setdefault(key, []).append(spec)
        for group in pairs.values():
            group = sorted(group, key=lambda s: {'none': 0, 'flip': 1, 'equivariant': 2}[s[field]] if field == 'symmetry' else str(s[field]))
            if len(group) < 2:
                continue
            base = group[0]
            for other in group[1:]:
                scores = []
                for spec in [base, other]:
                    f = pd.read_csv(out / 'screening' / spec['name'] / 'predictions.csv')
                    scores.append(subject_scores(f, np.column_stack([1 - f.prob_right, f.prob_right])).set_index('subject').balanced_accuracy)
                delta = scores[1] - scores[0]
                low, high = interval(delta, plan)
                ablations.append({'factor': field, 'base': base['name'], 'alternative': other['name'], 'base_setting': base[field], 'alternative_setting': other[field], 'base_balanced_accuracy': float(scores[0].mean()), 'alternative_balanced_accuracy': float(scores[1].mean()), 'mean_difference': float(delta.mean()), 'ci_low': low, 'ci_high': high})
    pd.DataFrame(ablations).to_csv(out / 'matched_development_ablations.csv', index=False)
    controls = out / 'diagnostics/controls.json'
    if controls.exists():
        d = json.loads(controls.read_text())
        lines += ['', '## Source-only learning diagnostics', '', 'These scores are not held-out accuracy. Three filter-bank variants (task-only, context, and equivariant context) use a separate diagnostic setting: the SAME 32 balanced source examples from four source people, 150 epochs, dropout 0, learning rate .003.']
        for r in d['memorization']:
            lines.append(f"- context={r['context']}, symmetry={r['symmetry']}: {r['training_accuracy'] * 100:.1f}% training accuracy, zero held-out examples.")
        lines += ['', f"Ten within-source-person label permutations with participant-group CV average {d['permuted_group_cv_mean_balanced_accuracy'] * 100:.2f}% BA. These controls test the learning path and loss of predictable signal under shuffling; they do not prove absence of every leakage."]
        for r in d['memorization']:
            pass
    baseline_controls = out / 'diagnostics/baseline_only_controls.json'
    if baseline_controls.exists():
        d = json.loads(baseline_controls.read_text())
        rest = d['preceding_rest_only']['balanced_accuracy']
        order = d['run_and_cue_ordinal_only']['balanced_accuracy']
        lines += ['', f'Source-only negative controls: preceding-rest-only BA {100 * rest:.2f}%; run/cue-ordinal-only BA {100 * order:.2f}%. The former is slightly above chance; these checks do not prove the absence of every possible confound. Neither external labels nor task EEG enter these control features.', '']
    canonical = out / 'diagnostics/canonical_input_check.json'
    if canonical.exists():
        d = json.loads(canonical.read_text())
        lines += ['', f"Archived canonical task-input check: {d['participants']} people, {d['trials']} epochs; maximum absolute difference {d['maximum_absolute_difference_uv']:g} microvolts for the matched 16-electrode, 8–30 Hz late input. New pre-cue features are additional inputs, not a hidden change to the reference.", '']
    lines += ['', '## Interpretation and provenance', '', 'The original 24-person confirmation has been reused as development here, so its new selection scores are not new independent evidence. The external protocols change both dataset and, in LOSO, training cohort. Do not compare their raw percentages to a previous PhysioNet test as if they were a paired accuracy gain. A paired interval crossing zero does not demonstrate a reliable positive gain. Context-versus-trial comparisons change the input requirement as well as the selected algorithm.', '', f'{len(ablations)} matched source-development comparisons retain adverse effects. These are exploratory, share overlapping source folds, and are not multiplicity-adjusted causal claims. Temperature scaling uses source OOF labels only and leaves the 0.5 decision boundary unchanged. Any remaining probability-quality problem must be reported rather than calibrated on the external holdout.', '', 'The contribution is a tested combination/hypothesis and a reproducible evaluation, not a claim that filter banks, ERD, mirroring or robust group weighting were invented here. The network is a compact independent FBCNet-inspired adaptation. Group weighting is inspired by Group DRO, with source participant groups and fixed eta; it is not a complete reproduction of the cited optimization studies. No degree, publication, SOTA, clinical or real-time-hardware claim follows from these experiments.', '', '**Independent verification: PASS.** All final transfer and external LOSO fits were reinferred from real EEG. Saved OOF predictions, source trial/person identities, source-fitted temperatures and combination coefficients were checked. The archived frontier study is retained unchanged.', '', '## Primary references', '', '- FBCNet: https://arxiv.org/abs/2104.01233', '- Channel- and label-flip augmentation: https://pubmed.ncbi.nlm.nih.gov/40039924/', '- ERD/ERS principles: https://pubmed.ncbi.nlm.nih.gov/10576479/', '- Group DRO: https://arxiv.org/abs/1911.08731', '- Original external dataset description: https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf', '- MATLAB loader/channel documentation: https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = list(NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout='constrained', sharey=True)
    for ax, (protocol, title) in zip(axes, [('external_transfer', 'Direct transfer'), ('external_loso', 'Frozen-algorithm external LOSO')]):
        table = summaries[protocol].set_index('model').loc[order]
        for i, (model, row) in enumerate(table.iterrows()):
            ax.errorbar(i, row.balanced_accuracy * 100, yerr=[[100 * (row.balanced_accuracy - row.ci_low)], [100 * (row.ci_high - row.balanced_accuracy)]], fmt='o', capsize=4, color='#d97706' if model == 'selected_pre_cue' else '#2563eb')
        ax.axhline(50, color='#64748b', linestyle='--')
        ax.set(ylim=(0, 100), xticks=np.arange(4), xticklabels=['CSP', 'EEGNet', 'Selected task', 'Selected context'], title=title)
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.grid(axis='y', alpha=0.15)
    axes[0].set_ylabel('Participant-mean balanced accuracy (%)')
    fig.suptitle('Nine external people · different training conditions shown separately')
    fig.savefig(out / 'external_accuracy.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, (protocol, title) in zip(axes, [('external_transfer', 'Direct transfer'), ('external_loso', 'External LOSO')]):
        p = per_subject[protocol].pivot(index='subject', columns='model', values='balanced_accuracy')[order]
        image = ax.imshow(p.to_numpy() * 100, vmin=0, vmax=100, cmap='viridis', aspect='auto')
        ax.set(xticks=np.arange(4), xticklabels=['CSP', 'EEGNet', 'Selected task', 'Selected context'], yticks=np.arange(len(p)), yticklabels=[f'A{s - 1000:02d}' for s in p.index], title=title)
        ax.tick_params(axis='x', rotation=15, labelsize=9)
    fig.colorbar(image, ax=axes, label='Balanced accuracy (%)')
    fig.savefig(out / 'external_participants.png', dpi=180)
    plt.close(fig)
    images = ''.join(('<img alt="' + name + '" src="data:image/png;base64,' + base64.b64encode((out / name).read_bytes()).decode() + '">' for name in ['external_accuracy.png', 'external_participants.png']))
    tables = ''.join(('<h2>' + html.escape(protocol) + '</h2>' + table.to_html(index=False, float_format=lambda v: f'{v:.4f}') for protocol, table in summaries.items()))
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuroEdge context/symmetry study</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:24px;background:#f8fafc;color:#0f172a}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #cbd5e1}.tables{overflow:auto}p{line-height:1.7}</style><h1>NeuroEdge: pre-cue contrast and reflection</h1><p>63 base configurations; source-only development. Nine new dataset participants. Direct transfer and frozen-algorithm LOSO have different training conditions. Context requires the preceding 1.5-second rest segment. See REPORT.md for methods, uncertainty and limits.</p>' + images + '<div class="tables">' + tables + '</div></html>'
    (out / 'results.html').write_text(page, encoding='utf-8')
    print(out / 'REPORT.md', flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
