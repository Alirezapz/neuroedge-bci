"""Report every calibration budget; sixteen labels is the prespecified primary."""
import base64
import json
import numpy as np
import pandas as pd
from .context_protocol import interval
from .context_report import NAMES, FA
from .personalization import bind_selection
NAMES = {**NAMES, 'fixed_gate': 'Fixed population gate', 'quality_gate': 'Trial quality gate'}
FA = NAMES.copy()

def report(plan):
    out, identity = bind_selection(plan)
    if json.loads((out / 'verification.json').read_text())['status'] != 'PASS':
        raise RuntimeError('Verify calibration before reporting.')
    summary = pd.read_csv(out / 'summary.csv')
    per = pd.read_csv(out / 'metrics_by_subject.csv')
    primary = plan['primary_budget']
    table = summary[summary.support_budget.isin([0, primary])]
    lines = ['# Session-disjoint personal intercept calibration', '', 'A supplementary protocol registered before external acquisition. This requires labeled examples from the new person and is a different input/training condition from zero-calibration decoding. No network or feature weight is fine-tuned.', '', f'The first {primary} chronological left/right trials of session T are the prespecified primary support set. All 144 session-E trials per person remain out of the adaptation fit. The SAME test trials are scored with and without calibration. Budgets 4/8/16 are all reported; no external score chooses the budget, prior or source algorithm.', '', 'Only one log-odds intercept is fitted with a class-balanced likelihood and a fixed zero-mean Gaussian prior (standard deviation 4). If the chronological support contains just one class, keep the source model without looking ahead or excluding the person. Source pipelines/ensembles are inherited from the parent PhysioNet-only selection.', '', '| Pipeline | Session-T labeled support | Session-E balanced accuracy | 95% participant interval | Accuracy |', '|---|---:|---:|---:|---:|']
    for r in table.itertuples():
        lines.append(f'| {NAMES[r.model]} | {r.support_budget} | {r.balanced_accuracy * 100:.2f}% | [{r.ci_low * 100:.2f}, {r.ci_high * 100:.2f}]% | {r.accuracy * 100:.2f}% |')
    comparisons = []
    bootstrap = {'evaluation': {'bootstrap_seed': plan['bootstrap_seed'], 'bootstrap_iterations': plan['bootstrap_iterations']}}
    for model, group in per.groupby('model'):
        pivot = group.pivot(index='subject', columns='support_budget', values='balanced_accuracy')
        delta = pivot[primary] - pivot[0]
        lo, hi = interval(delta, bootstrap)
        comparisons.append({'model': model, 'support_budget': primary, 'subjects': len(delta), 'mean_difference': float(delta.mean()), 'ci_low': lo, 'ci_high': hi})
    pd.DataFrame(comparisons).to_csv(out / 'paired_calibration_differences.csv', index=False)
    lines += ['', '## Paired calibration effect', '']
    for r in comparisons:
        lines.append(f"- {NAMES[r['model']]}: {r['mean_difference'] * 100:+.2f} percentage points; 95% paired interval [{r['ci_low'] * 100:+.2f}, {r['ci_high'] * 100:+.2f}].")
    lines += ['', 'A paired interval crossing zero does not establish a reliable positive calibration gain. These nine-person conditional intervals do not include support-set resampling or every source-training uncertainty. The public dataset supplies the labeled support; this is a recorded-data calibration simulation, not a live user study. The adaptation is an existing type of calibration idea being tested for the observed offset hypothesis, not a new general algorithm.', '', 'Independent verification: **PASS**. First-session chronology, test-session separation, all budgets, intercept fits, predictions and metrics were recomputed.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for model, group in summary.groupby('model'):
        group = group.sort_values('support_budget')
        ax.plot(group.support_budget, group.balanced_accuracy * 100, marker='o', label=NAMES[model])
    ax.axvline(primary, color='#64748b', linestyle=':', label='Prespecified primary budget')
    ax.axhline(50, color='#94a3b8', linestyle='--')
    ax.set(xticks=[0] + plan['support_budgets'], ylim=(0, 100), xlabel='Labeled first-session support trials', ylabel='Second-session participant-mean balanced accuracy (%)', title='Personal intercept adaptation · same nine people and test trials')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=0.15)
    fig.savefig(out / 'calibration_curve.png', dpi=180)
    plt.close(fig)
    image = base64.b64encode((out / 'calibration_curve.png').read_bytes()).decode()
    (out / 'results.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Personal calibration</title><style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:20px}img{max-width:100%}table{font-size:13px;border-collapse:collapse}.table{max-width:100%;overflow-x:auto}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>Session-disjoint personal calibration</h1><p>Requires labeled examples from the new person. Sixteen support trials is the prespecified primary condition; all budgets are retained. Same session-E test trials with and without adaptation. See REPORT.md for paired participant intervals and limits.</p><img alt="Calibration support budget curve" src="data:image/png;base64,' + image + '">' + '<div class="table">' + summary.to_html(index=False, float_format=lambda x: f'{x:.4f}') + '</div></html>', encoding='utf-8')
    print(out / 'REPORT.md', flush=True)
