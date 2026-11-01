"""Matched-session reports for the predeclared residual personal calibration."""
import argparse
import base64
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .context_protocol import interval
from .residual_study import load_plan, register
NAMES = {'bias': 'Intercept only', 'spectral': 'Spectral kernel', 'geometry': 'Covariance-geometry kernel', 'hybrid': 'Equal kernel fusion (primary)'}
FA = NAMES.copy()

def report(plan):
    out, identity = register(plan)
    if json.loads((out / 'verification.json').read_text())['status'] != 'PASS':
        raise RuntimeError('Verify residual calibration before reporting.')
    summary = pd.read_csv(out / 'summary.csv')
    per = pd.read_csv(out / 'metrics_by_subject.csv')
    source = pd.read_csv(out / 'source_summary.csv')
    selection = json.loads((out / 'source_selection.json').read_text())
    budget = plan['primary_budget']
    primary = plan['primary_kernel']
    bootstrap = {'evaluation': {'bootstrap_iterations': plan['bootstrap_iterations'], 'bootstrap_seed': plan['bootstrap_seed']}}
    differences = []
    baseline = per[(per.model == 'bias') & (per.support_budget == budget)].set_index('subject').balanced_accuracy
    zero = per[(per.model == 'bias') & (per.support_budget == 0)].set_index('subject').balanced_accuracy
    for kind in plan['kernels']:
        scores = per[(per.model == kind) & (per.support_budget == budget)].set_index('subject').balanced_accuracy
        for reference, values in [('same_budget_bias', baseline), ('unadapted_same_session', zero)]:
            delta = scores - values
            lo, hi = interval(delta, bootstrap)
            differences.append({'model': kind, 'reference': reference, 'support_budget': budget, 'subjects': len(delta), 'mean_difference': float(delta.mean()), 'ci_low': lo, 'ci_high': hi})
    pd.DataFrame(differences).to_csv(out / 'paired_differences.csv', index=False)
    primary_row = summary[(summary.model == primary) & (summary.support_budget == budget)].iloc[0]
    bias_row = summary[(summary.model == 'bias') & (summary.support_budget == budget)].iloc[0]
    difference = next((r for r in differences if r['model'] == primary and r['reference'] == 'same_budget_bias'))
    lines = ['# Nonlinear personal residual calibration', '', f"Primary hybrid, {budget} labeled support trials: BA {100 * primary_row.balanced_accuracy:.2f}%, compared with {100 * bias_row.balanced_accuracy:.2f}% for intercept-only calibration on the SAME session-E trials. Paired difference {100 * difference['mean_difference']:+.2f} pp [{100 * difference['ci_low']:+.2f}, {100 * difference['ci_high']:+.2f}].", '', 'This is a supplementary sequential source-development hypothesis. Its mathematical specification was recorded before external acquisition. Implementation and source-only C selection were frozen after raw acquisition but before the first external transfer scores. The raw external EEG or its labels never fit a population transform or select C, kernel, budget or primary hypothesis.', '', 'All methods inherit the SAME source-selected task-only five-component ensemble. They use the same chronological session-T support and all 144 session-E test trials per person. No deep weight is fine-tuned. Unlike zero-calibration decoding, these methods require personal labels.', '', '## Method', '', 'The source ensemble logit is retained as a prior. A regularized support-kernel residual and one intercept correct it. Spectral features are relative log variances in full/two half windows across eight bands and sixteen sensors. Geometry features are upper-triangle vectors of the matrix logarithm of trace-normalized, 0.1-shrunk full-window covariance. Off-diagonal entries receive sqrt(2) weighting. Feature means/std come from source participants only. Gaussian kernels use mean squared standardized feature distance; the primary hybrid is the equal mean of the two kernels.', '', 'Population scaling uses people 1–80 during source selection, then all 104 development people for external evaluation. Each kernel chooses C from [0.01,0.1,1,10] using first16 chronological source run4/8 labels and held-out run12 trials of the previously inspected 24 development people. Those scores are development evidence. All 12 choices are retained.', '', 'For new people, support is the first chronological left/right session-T trials. Budgets4/8/16 are all reported; 16 and the hybrid kernel were fixed as primary. If support contains only one class, keep the source decoder. All features use task EEG only and the same sixteen available electrodes. The complete signal pipeline is offline.', '', '| Method | Source-selected C | Source-development run12 BA |', '|---|---:|---:|']
    for kind, r in selection['selected'].items():
        lines.append(f"| {NAMES[kind]} | {r['C']:g} | {100 * r['balanced_accuracy']:.2f}% |")
    lines += ['', '## All external conditions', '', '| Method | Session-T support | Session-E BA | 95% participant interval | Log loss |', '|---|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {NAMES[row.model]} | {row.support_budget} | {100 * row.balanced_accuracy:.2f}% | [{100 * row.ci_low:.2f}, {100 * row.ci_high:.2f}]% | {row.log_loss:.4f} |')
    lines += ['', '## Paired differences', '', '| Method | Reference | Difference (pp) | 95% paired interval |', '|---|---|---:|---:|']
    for r in differences:
        lines.append(f"| {NAMES[r['model']]} | {r['reference']} | {100 * r['mean_difference']:+.2f} | [{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}] |")
    limits = 'Intervals resample people, not trials or seeds. They are descriptive, conditional on these fitted models and not adjusted for all secondary comparisons. '
    limits += 'An interval crossing zero does not establish a reliable positive improvement. Kernels, covariance logarithms and logistic regularization are established tools; this experiment does not prove worldwide algorithmic novelty, SOTA, clinical use or a degree level.'
    lines += ['', limits, '', 'Independent verification checks all12 source choices, both source-only scalers, chronological support identities and all external method/budget/person predictions.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    colors = {'bias': '#64748b', 'spectral': '#2563eb', 'geometry': '#7c3aed', 'hybrid': '#d97706'}
    for kind, marker in zip(['bias'] + plan['kernels'], ['o', 's', '^', 'D']):
        s = summary[summary.model == kind].sort_values('support_budget')
        axes[0].errorbar(s.support_budget, 100 * s.balanced_accuracy, yerr=np.array([100 * (s.balanced_accuracy - s.ci_low), 100 * (s.ci_high - s.balanced_accuracy)]), marker=marker, capsize=3, color=colors[kind], linewidth=2 if kind == primary else 1, label=NAMES[kind])
    axes[0].set(xlabel='Labeled session-T support trials', ylabel='Session-E participant-mean balanced accuracy (%)', ylim=(0, 100), xticks=[0, 4, 8, 16], title='Same unseen session-E trials at every budget')
    axes[0].axhline(50, color='#94a3b8', linestyle='--')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.15)
    comparisons = [r for r in differences if r['reference'] == 'same_budget_bias']
    for i, r in enumerate(comparisons):
        axes[1].errorbar(100 * r['mean_difference'], i, xerr=[[100 * (r['mean_difference'] - r['ci_low'])], [100 * (r['ci_high'] - r['mean_difference'])]], fmt='o', capsize=4, color=colors[r['model']])
    axes[1].set(yticks=np.arange(len(comparisons)), yticklabels=[NAMES[r['model']] for r in comparisons], xlabel='Difference vs intercept-only calibration (pp)', title='16 identical labels: paired participant differences')
    axes[1].axvline(0, color='#94a3b8', linestyle='--')
    axes[1].grid(axis='x', alpha=0.15)
    fig.savefig(out / 'personal_kernels.png', dpi=180)
    plt.close(fig)
    image = base64.b64encode((out / 'personal_kernels.png').read_bytes()).decode()
    display = summary.copy()
    display['method'] = display.model.map(NAMES)
    intro_html = re.sub('\\[[+-]\\d+\\.\\d+, [+-]\\d+\\.\\d+\\]|[+-]\\d+\\.\\d+', lambda m: "<bdi dir='ltr'>" + m.group(0) + '</bdi>', lines[2])
    from .english_reports import portable_page
    portable_page(out, 'Source-prior nonlinear personal calibration', lines[2], {'All method and support conditions': display}, ['personal_kernels.png'])
    page = (out / 'results.html').read_text()
    (out / 'results.html').write_text(page, encoding='utf-8')
    print(out / 'RESULTS.md', flush=True)
    return differences
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='configs/residual_personal.yaml')
    args = parser.parse_args()
    report(load_plan(args.plan))
