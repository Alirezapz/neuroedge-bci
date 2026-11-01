"""Compare both prespecified gate variants and the frozen source-selected one."""
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import json_write
from .context_protocol import interval
from .context_report import NAMES
from .gated_study import freeze

def report(plan):
    out, identity = freeze(plan)
    parent = Path(plan['_parent']['_root']) / plan['_parent']['output_dir']
    if json.loads((out / 'verification.json').read_text())['status'] != 'PASS' or json.loads((parent / 'verification.json').read_text())['status'] != 'PASS':
        raise RuntimeError('Verify both real-EEG studies before comparing them.')
    selection = json.loads((out / 'selection.json').read_text())
    names = {**NAMES, 'fixed_gate': 'Fixed learned gate', 'quality_gate': 'Quality-conditioned gate', 'selected_fusion': 'Source-selected fusion'}
    lines = ['# Fixed versus quality-conditioned context fusion', '', 'This follow-up was registered after the 63-candidate source screen, before any external EEG acquisition. The parent was paused during reused source validation so both new variants could finish source development and freeze before external data. Previously inspected people remain development data; external results never choose architecture, seed, gate or epoch.', '', 'Both networks retain active log-variance features and context contrasts, share BatchNorm across concatenated task/rest time points, and enforce reflection equivariance. Both have the SAME gate architecture and parameter count. The fixed control feeds constant zero conditioning features to the gate, learning population-fixed weights. The quality-conditioned version feeds relative baseline log variance and its temporal dispersion. These are operational quality proxies, not ground-truth artifact labels. Gate weights lie in [0,1]. Both feature paths retain task-only information so noisy context can be downweighted.', '', 'The architecture combines known variance, conditioning and symmetry ideas. World-first novelty is not established. All models still require the same 16 electrodes and the preceding 1.5-second context. This is an offline recorded-data study.', '', f"Frozen source-selected fusion: `{selection['selected']['variant']}`, reused-validation balanced accuracy {selection['selected']['balanced_accuracy'] * 100:.2f}%. Both variants are reported in direct transfer. Only this source-selected variant is refitted in external LOSO, without external reselection.", '']
    comparisons = []
    tables = {}
    bootstrap = {'evaluation': {'bootstrap_seed': plan['bootstrap_seed'], 'bootstrap_iterations': plan['bootstrap_iterations']}}
    for protocol, title, fa_title in [('external_transfer', 'Direct transfer', 'Direct transfer without external task training'), ('external_loso', 'Frozen-algorithm external LOSO', 'Train on eight other people; test the ninth')]:
        original = pd.read_csv(parent / protocol / 'summary.csv')
        new = pd.read_csv(out / protocol / 'summary.csv')
        table = pd.concat([original, new], ignore_index=True)
        tables[protocol] = table
        combined = pd.concat([pd.read_csv(parent / protocol / 'metrics_by_subject.csv'), pd.read_csv(out / protocol / 'metrics_by_subject.csv')], ignore_index=True)
        lines += ['## ' + title, '', '| Pipeline | Balanced accuracy | 95% participant interval | Accuracy | Log loss |', '|---|---:|---:|---:|---:|']
        for r in table.itertuples():
            lines.append(f'| {names[r.model]} | {r.balanced_accuracy * 100:.2f}% | [{r.ci_low * 100:.2f}, {r.ci_high * 100:.2f}]% | {r.accuracy * 100:.2f}% | {r.log_loss:.4f} |')
        pivot = combined.pivot(index='subject', columns='model', values='balanced_accuracy')
        pairs = [('quality_gate', 'fixed_gate'), ('quality_gate', 'selected_pre_cue'), ('fixed_gate', 'selected_pre_cue')] if protocol == 'external_transfer' else [('selected_fusion', 'selected_pre_cue'), ('selected_fusion', 'selected_zero_calibration')]
        for proposed, reference in pairs:
            delta = pivot[proposed] - pivot[reference]
            lo, hi = interval(delta, bootstrap)
            comparisons.append({'protocol': protocol, 'proposed': proposed, 'reference': reference, 'subjects': len(delta), 'mean_difference': float(delta.mean()), 'ci_low': lo, 'ci_high': hi, 'same_input_requirement': reference != 'selected_zero_calibration'})
    pd.DataFrame(comparisons).to_csv(out / 'paired_differences.csv', index=False)
    lines += ['', '## Paired participant comparisons', '']
    for r in comparisons:
        text = f"{r['protocol']}: {r['proposed']} versus {r['reference']}: {r['mean_difference'] * 100:+.2f} pp [{r['ci_low'] * 100:+.2f}, {r['ci_high'] * 100:+.2f}]."
        lines.append('- ' + text)
    lines += ['', 'Intervals are conditional participant bootstraps on nine people; a paired interval crossing zero does not establish a reliable positive gain. The direct-transfer and LOSO protocols have different training conditions. These are not paired accuracy gains over the older PhysioNet confirmation. The gate comparison tests conditioning within this specified architecture; it is not a causal physiological explanation or proof of general novelty.', '', 'Independent verification: **PASS**, including reinference of all six source OOF fits, both validation/final variants and every external LOSO fold. No adverse source or external result is omitted. Sources and original methods are in docs/CONTEXT_METHODS.md and the parent report.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained', sharey=True)
    for ax, (protocol, title) in zip(axes, [('external_transfer', 'Direct transfer'), ('external_loso', 'External LOSO')]):
        table = tables[protocol]
        for i, r in enumerate(table.itertuples()):
            ax.errorbar(i, r.balanced_accuracy * 100, yerr=[[100 * (r.balanced_accuracy - r.ci_low)], [100 * (r.ci_high - r.balanced_accuracy)]], fmt='o', capsize=4)
        ax.axhline(50, color='#64748b', linestyle='--')
        ax.set(ylim=(0, 100), xticks=np.arange(len(table)), xticklabels=[names[m] for m in table.model], title=title)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.grid(axis='y', alpha=0.15)
    axes[0].set_ylabel('Participant-mean balanced accuracy (%)')
    fig.savefig(out / 'gated_comparison.png', dpi=180)
    plt.close(fig)
    image = base64.b64encode((out / 'gated_comparison.png').read_bytes()).decode()
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quality-conditioned context</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:20px}img{max-width:100%}table{border-collapse:collapse;font-size:13px}.table{max-width:100%;overflow-x:auto}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>Fixed versus quality-conditioned context</h1><p>Source-only hypothesis and selection, frozen before new-dataset acquisition. Both variants retain task features and use the preceding rest. Different training conditions are shown separately. See REPORT.md for paired intervals and limitations.</p><img alt="External gated comparison" src="data:image/png;base64,' + image + '">'
    for protocol, table in tables.items():
        html += '<h2>' + protocol + '</h2><div class="table">' + table.to_html(index=False, float_format=lambda x: f'{x:.4f}') + '</div>'
    (out / 'results.html').write_text(html + '</html>', encoding='utf-8')
    print(out / 'REPORT.md', flush=True)
