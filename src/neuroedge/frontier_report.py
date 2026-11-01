"""Derived statistics, figures and readable reports for frozen frontier results."""
from __future__ import annotations
import base64
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import digest, json_write
from .frontier_protocol import arm, bootstrap, freeze, subject_scores
from .frontier_features import candidates
from .frontier_data import BANDS, WINDOWS

def describe(spec):
    bands = ', '.join((f'{BANDS[b][0]}–{BANDS[b][1]}' for b in spec['bands']))
    start, stop = WINDOWS[spec['window']]
    parts = [spec.get('architecture', spec['family']), f"{spec['channels']} electrodes", f'{bands} Hz', f'[{start:g},{stop:g}) s']
    for field in ['classifier', 'augmentation', 'components', 'reg', 'metric', 'normalize', 'select']:
        if field in spec:
            parts.append(f'{field}={spec[field]}')
    return '; '.join(parts)

def report(plan):
    out, identity = freeze(plan)
    verification = json.loads((out / 'verification.json').read_text())
    if verification['status'] != 'PASS':
        raise RuntimeError('Verify actual final predictions before writing the report.')
    selection = json.loads((out / 'selection.json').read_text())
    summary = pd.read_csv(out / 'confirmation/summary.csv')
    per = pd.read_csv(out / 'confirmation/metrics_by_subject.csv')
    pivot = per.pivot(index='subject', columns='model', values='balanced_accuracy')
    comparisons = []
    for selected, baseline in [('selected_zero_calibration', 'csp16_baseline'), ('selected_zero_calibration', 'eegnet16_baseline'), ('selected_rest_calibrated', 'selected_zero_calibration')]:
        difference = pivot[selected] - pivot[baseline]
        low, high = bootstrap(difference, plan)
        comparisons.append({'selected': selected, 'baseline': baseline, 'subjects': len(difference), 'mean_difference': float(difference.mean()), 'ci_low': low, 'ci_high': high, 'same_operating_condition': selected != 'selected_rest_calibrated'})
    pd.DataFrame(comparisons).to_csv(out / 'confirmation/paired_differences.csv', index=False)
    audit = json.loads((out / 'confirmation_data/audit.json').read_text())
    screen = pd.read_csv(out / 'screening_summary.csv')
    validation = pd.read_csv(out / 'validation_summary.csv')
    validation_details = {p.parent.name: json.loads(p.read_text()) for p in (out / 'validation').glob('*/complete.json')}
    specs = candidates()

    def matched_key(spec):
        return digest({k: v for k, v in spec.items() if k not in {'name', 'arm'}})
    uncalibrated = {matched_key(s): s for s in specs if arm(s) == 'zero_calibration'}
    ablation = []
    for calibrated in [s for s in specs if arm(s) == 'rest_calibrated']:
        base = uncalibrated[matched_key(calibrated)]
        frames = [pd.read_csv(out / 'screening' / s['name'] / 'predictions.csv') for s in [base, calibrated]]
        scores = [subject_scores(f, np.column_stack([1 - f.prob_right, f.prob_right])).set_index('subject').balanced_accuracy for f in frames]
        delta = scores[1] - scores[0]
        lo, hi = bootstrap(delta, plan)
        ablation.append({'base': base['name'], 'rest_calibrated': calibrated['name'], 'family': base['family'], 'discovery_subjects': len(delta), 'base_balanced_accuracy': float(scores[0].mean()), 'rest_balanced_accuracy': float(scores[1].mean()), 'difference': float(delta.mean()), 'ci_low': lo, 'ci_high': hi, 'scope': 'Development OOF representation ablation, not final confirmation'})
    pd.DataFrame(ablation).to_csv(out / 'rest_alignment_development_ablation.csv', index=False)
    from .frontier_ablation import development_ablations
    factor_ablation = development_ablations(plan)
    lines = ['# NeuroEdge frontier study', '', 'A bounded, broad sequential exploratory search, followed by one new participant holdout. This is not an exhaustive search of every EEG method or a claim of a new algorithm.', '', f"Discovery: {len(plan['discovery_subjects'])} participants, {plan['search']['folds']}-fold participant-group cross-validation. Secondary validation: {len(plan['validation_subjects'])} different participants. Previously inspected participants 71–80 are explicitly development data now. Only the secondary validation chooses the final pipelines; confirmation does not choose a model, seed, weight or epoch.", '', f"Base configurations completed: {len(screen)}. Secondary validation configurations, including combinations: {len(validation)}. Confirmation requested {len(plan['confirmation_subjects'])} participants; {len(audit['included_subjects'])} pass the frozen structural-data QC rule.", '', '## Operating conditions', '', "The primary zero-calibration arm uses no EEG or statistic from a new participant before that task trial. The secondary rest-calibrated arm additionally requires that participant's separate R01 eyes-open baseline. Its covariance whitening uses no task/test trial or label, but its extra calibration requirement must be reported. All other learned population transforms, CSP, tangent references, scaling, classifiers and combination weights use source participants only.", '', '## Frozen development selections', '']
    for current_arm, winner in selection['selected'].items():
        lines.append(f"- {current_arm}: `{winner['name']}`; validation balanced accuracy {winner['balanced_accuracy'] * 100:.2f}%; kind {winner['kind']}.")
        if winner['kind'] == 'ensemble':
            lines.append(f"  Members: {', '.join(winner['combination']['members'])}; rule: {winner['combination']['kind']}.")
            if 'weights' in winner['combination']:
                lines.append('  Frozen source-OOF vote weights (same member order): ' + ', '.join((f'{w:.8g}' for w in winner['combination']['weights'])) + '.')
            for member in winner['combination']['members']:
                spec = next((s for s in specs if s['name'] == member))
                details = validation_details[member]
                neural_policy = f"; fixed epochs={details['refit_epochs']}; seeds={details['seeds']}" if spec['family'] == 'neural' else ''
                lines.append(f'  `{member}`: {describe(spec)}{neural_policy}.')
            sensors = max((next((s for s in specs if s['name'] == member))['channels'] for member in winner['combination']['members']))
            calibration_text = ', including its separate resting reference' if current_arm == 'rest_calibrated' else ''
            lines.append(f'  The complete pipeline requires access to {sensors} electrodes{calibration_text}.')
        else:
            lines.append('  ' + describe(winner['spec']) + f"; fixed epochs={winner.get('refit_epochs')}; seeds={winner.get('seeds')}.")
    lines += ['', '## Prospective confirmation', '', 'All final models are refitted using the 80 development participants before confirmation data acquisition. Advanced neural candidates average the three fixed training seeds; the original EEGNet baseline retains its fixed seed and 25 epochs. Intervals below bootstrap participants, not windows; they are conditional descriptive intervals, not guarantees.', '', '| Pipeline | Condition | Participants | Balanced accuracy | 95% participant interval | Accuracy | Log loss |', '|---|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.model} | {r.arm} | {r.subjects} | {r.balanced_accuracy * 100:.2f}% | [{r.ci_low * 100:.2f}, {r.ci_high * 100:.2f}]% | {r.accuracy * 100:.2f}% | {r.log_loss:.4f} |')
    primary = summary.loc[summary.model == 'selected_zero_calibration'].iloc[0]
    if primary.log_loss > np.log(2):
        lines += ['', f'The selected zero-calibration pipeline has log loss {primary.log_loss:.4f}, worse than {np.log(2):.4f} for constant 0.5 probabilities, despite better-than-chance decisions. Its probability confidence is therefore a remaining limitation under log loss. No probability calibration is fitted using confirmation data.']
    lines += ['', '## Paired participant differences', '', '| Compared pipeline | Reference | Difference (percentage points) | 95% paired interval | Same calibration requirement |', '|---|---|---:|---:|---|']
    for r in comparisons:
        lines.append(f"| {r['selected']} | {r['baseline']} | {r['mean_difference'] * 100:+.2f} | [{r['ci_low'] * 100:+.2f}, {r['ci_high'] * 100:+.2f}] | {r['same_operating_condition']} |")
    lines += ['', 'An interval crossing zero does not establish a reliable positive gain. The selected zero-calibration pipeline also changes sensor count, window and preprocessing relative to the references; its comparison is a complete-pipeline comparison, not an isolated architecture effect. The rest-versus-zero comparison changes the calibration requirement as well as the selected pipeline; it cannot be interpreted as an isolated architectural improvement.', '', '## Structural exclusions', '']
    if audit['exclusions']:
        lines.extend((f"- S{e['subject']:03d}: {e['reason']}" for e in audit['exclusions']))
    else:
        lines.append('None.')
    lines += ['', '## Matched development alignment ablation', '', 'All fifteen rest-calibrated configurations are paired with the same uncalibrated hyperparameters on the same discovery group folds. This is development evidence and its intervals do not include all dependence from overlapping source folds. No configuration is omitted for a negative effect. See rest_alignment_development_ablation.csv.', '', '| Family | Uncalibrated configuration | Base BA | Rest-calibrated BA | Difference (pp) |', '|---|---|---:|---:|---:|']
    for row in ablation:
        lines.append(f"| {row['family']} | {row['base']} | {row['base_balanced_accuracy'] * 100:.2f}% | {row['rest_balanced_accuracy'] * 100:.2f}% | {row['difference'] * 100:+.2f} |")
    lines += ['', f'An additional {len(factor_ablation)} matched development comparisons isolate specified channel, window, band, normalization, augmentation or shrinkage changes while keeping the other candidate fields fixed. All rows, including adverse effects, are retained in matched_development_ablations.csv. These exploratory comparisons do not drive a new confirmation selection or establish a multiplicity-adjusted effect.']
    controls = out / 'diagnostics/controls.json'
    if controls.exists():
        control = json.loads(controls.read_text())
        lines += ['', '## Source-only learning controls', '', 'These controls do not change selection and are not held-out performance results. With dropout disabled and a separate diagnostic learning setting, each neural architecture is trained for 120 epochs on 32 balanced real discovery examples.', '']
        for row in control['memorization']:
            lines.append(f"- {row['architecture']}: {row['training_accuracy'] * 100:.1f}% accuracy on the same 32 training examples; no held-out examples.")
        lines += ['', f"Ten within-participant label-shuffle controls, retaining the subject-group split, average {control['permuted_group_cv_mean_balanced_accuracy'] * 100:.2f}% balanced accuracy (range {control['permuted_group_cv_min'] * 100:.2f}–{control['permuted_group_cv_max'] * 100:.2f}%). This is consistent with a working learning path and loss of predictable signal after label permutation; it does not itself establish generalization or prove that every possible leakage is absent."]
    lines += ['', 'Exclusions follow the frozen whole-participant structural rule, before any confirmation model score. No participant is dropped for low accuracy. The task and cohort differ from public within-person benchmarks; published accuracy numbers cannot be substituted as matched baselines.', '', '## Search and verification scope', '', 'Classical methods cover regularized CSP/LDA, CSP/RBF-SVM, six-band CSP with train-only feature selection, single/multiband Riemannian tangent logistic models, minimum-distance-to-mean, spectral power with linear, RBF and tree classifiers. Neural methods include EEGNet, independent compact shallow log-power and convolution/attention adaptations, two sensor sets, mixup and same-class segment recombination. They are not exact reproductions of every cited network. Mean voting, diverse mean voting, nonnegative source-OOF-weighted voting and source-OOF logistic stacking are evaluated on secondary validation. Neural source-only inner participant validation chooses epoch counts; a full source refit then makes every discovery OOF prediction. Final epoch policies and seeds are fixed before confirmation. Failed or negative methods remain in the screen table.', '', f"Independent verification: **{verification['status']}**. All screening and validation metrics are recomputed from saved predictions. Final selected constituents and baselines are reinferred from real EEG; scalers are recomputed from source data, and individual/batch predictions are compared.", '', 'These are offline cue-aligned experiments. No causal online, clinical, real-time hardware, MCU, energy, state-of-the-art or degree-level novelty claim follows from a larger model search. More test-informed iteration requires another untouched dataset/cohort.', '', '## Reproduction', '', '`python -m neuroedge.frontier all --plan configs/frontier.yaml`', '', 'For a new host, first create a fresh plan/output name with scripts/new_run.py. Raw EDF, large covariance caches and temporary screening feature/model caches are omitted from the archive; final models, predictions, audit receipts and exact source hashes remain. Reacquire data to rerun inference verification.', '', '## Method references', '', '- EEGMMIDB: https://physionet.org/content/eegmmidb/1.0.0/', '- EEGNet (Lawhern et al.): https://arxiv.org/abs/1611.08024', '- Shallow ConvNet ideas (Schirrmeister et al.): https://arxiv.org/abs/1703.05051', '- EEG Conformer ideas (Song et al.): https://github.com/eeyhsong/EEG-Conformer', '- Euclidean alignment (He and Wu): https://arxiv.org/abs/1808.05464', '- Train-fitted tangent mapping: https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, brier_score_loss
    stored = pd.read_csv(out / 'confirmation/predictions.csv')
    errors = []
    for (name, subject), group in stored.groupby(['model', 'subject']):
        matrix = confusion_matrix(group.label, group.prediction, labels=[0, 1])
        errors.append({'model': name, 'subject': int(subject), 'left_recall': float(matrix[0, 0] / matrix[0].sum()), 'right_recall': float(matrix[1, 1] / matrix[1].sum()), 'macro_f1': float(f1_score(group.label, group.prediction, average='macro', zero_division=0)), 'roc_auc': float(roc_auc_score(group.label, group.prob_right)), 'brier_score': float(brier_score_loss(group.label, group.prob_right)), 'predicted_right_fraction': float(group.prediction.mean()), 'single_class_prediction': bool(group.prediction.nunique() == 1)})
    pd.DataFrame(errors).to_csv(out / 'confirmation/error_analysis_by_subject.csv', index=False)
    runs = []
    for (name, subject, run), group in stored.groupby(['model', 'subject', 'run']):
        p = np.column_stack([1 - group.prob_right, group.prob_right])
        value = subject_scores(group, p).iloc[0].to_dict()
        value['subject'] = int(value['subject'])
        value['trials'] = int(value['trials'])
        runs.append({'model': name, 'run': int(run), **value})
    pd.DataFrame(runs).to_csv(out / 'confirmation/metrics_by_run_subject.csv', index=False)
    display_names = {'csp16_baseline': 'CSP + LDA reference, 16 electrodes', 'eegnet16_baseline': 'EEGNet reference, 25 epochs', 'selected_zero_calibration': 'Selected zero-calibration model', 'selected_rest_calibrated': 'Selected resting-calibrated model'}
    for row in summary.itertuples():
        condition = 'Requires a separate R01 resting recording' if row.arm == 'rest_calibrated' else 'No new-person calibration'
    if primary.log_loss > np.log(2):
        pass
    for row in comparisons:
        pass
    for current_arm, winner in selection['selected'].items():
        if winner['kind'] == 'ensemble':
            if 'weights' in winner['combination']:
                pass
            for member in winner['combination']['members']:
                spec = next((s for s in specs if s['name'] == member))
                details = validation_details[member]
                neural_policy = f"; fixed epochs: {details['refit_epochs']}; seeds: {details['seeds']}" if spec['family'] == 'neural' else ''
            sensors = max((next((s for s in specs if s['name'] == member))['channels'] for member in winner['combination']['members']))
            calibration_text = ', for task and resting-reference data,' if current_arm == 'rest_calibrated' else ''
    if controls.exists():
        for row in control['memorization']:
            pass
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['CSP baseline', 'EEGNet baseline', 'Selected: zero calibration', 'Selected: resting calibration']
    fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
    y = summary.balanced_accuracy.to_numpy() * 100
    errors = np.stack([y - summary.ci_low.to_numpy() * 100, summary.ci_high.to_numpy() * 100 - y])
    for i in range(len(summary)):
        ax.errorbar(i, y[i], yerr=errors[:, i:i + 1], fmt='o', capsize=6, color='#d97706' if summary.iloc[i].arm == 'rest_calibrated' else '#2563eb')
    ax.axhline(50, color='#64748b', linestyle='--', label='Chance reference')
    ax.set(xticks=np.arange(len(summary)), xticklabels=labels, ylabel='Participant-mean balanced accuracy (%)', title=f'Prospective unseen participants · n={len(pivot)}')
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.15)
    ax.legend(frameon=False)
    fig.savefig(out / 'confirmation_accuracy.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.8), layout='constrained')
    zero = screen[screen.arm == 'zero_calibration']
    families = sorted(zero.family.unique())
    rng = np.random.default_rng(43)
    for i, family in enumerate(families):
        values = zero[zero.family == family].balanced_accuracy.to_numpy() * 100
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(values)), values, s=15, alpha=0.55, color='#2563eb')
        ax.scatter([i], [values.max()], marker='D', s=45, color='#0f172a')
    ax.axhline(50, color='#94a3b8', linestyle='--')
    ax.set(xticks=np.arange(len(families)), xticklabels=families, ylabel='Discovery OOF balanced accuracy (%)', title=f'Zero-calibration discovery search · {len(zero)} configurations\nCalibrated counterparts are reported separately')
    ax.grid(axis='y', alpha=0.15)
    fig.savefig(out / 'discovery_search.png', dpi=180)
    plt.close(fig)
    display = pivot[['csp16_baseline', 'eegnet16_baseline', 'selected_zero_calibration', 'selected_rest_calibrated']]
    fig, ax = plt.subplots(figsize=(8, max(5, 0.24 * len(display))), layout='constrained')
    heat = ax.imshow(display.to_numpy() * 100, vmin=0, vmax=100, cmap='viridis', aspect='auto')
    ax.set(xticks=np.arange(4), xticklabels=['CSP', 'EEGNet', 'Selected zero', 'Selected rest'], yticks=np.arange(len(display)), yticklabels=[f'S{s:03d}' for s in display.index], title='Each row is the same unseen participant')
    fig.colorbar(heat, ax=ax, label='Balanced accuracy (%)')
    fig.savefig(out / 'participant_accuracy.png', dpi=180)
    plt.close(fig)
    images = []
    for name in ['confirmation_accuracy.png', 'discovery_search.png', 'participant_accuracy.png']:
        images.append('<img style="max-width:100%;height:auto" src="data:image/png;base64,' + base64.b64encode((out / name).read_bytes()).decode() + '" alt="' + name + '">')
    table = summary[['model', 'arm', 'subjects', 'balanced_accuracy', 'ci_low', 'ci_high', 'log_loss']].to_html(index=False, float_format=lambda v: f'{v:.4f}')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuroEdge frontier results</title><style>body{font:16px system-ui;background:#f1f5f9;color:#0f172a;max-width:1100px;margin:40px auto;padding:24px}section{background:white;padding:24px;margin:20px 0;border-radius:14px}section{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e2e8f0}p{line-height:1.7}</style><h1>NeuroEdge · frontier study</h1><p>Frozen participant-disjoint development and one prospective confirmation. Zero-calibration and separate-resting-calibration conditions are shown explicitly.</p><section><h2>Verified results</h2>' + table + '</section>' + ''.join(('<section>' + i + '</section>' for i in images)) + '<section><h2>Interpretation</h2><p>Intervals resample participants. Rest-calibrated pipelines require an additional unlabeled resting recording. These are offline research results; the full method, exclusions, comparisons and limits are in REPORT.md.</p></section></html>'
    (out / 'results.html').write_text(page, encoding='utf-8')
    print(out / 'REPORT.md', flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
