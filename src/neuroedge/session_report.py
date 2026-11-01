"""English, self-contained comparison with all calibration budgets and failures."""
import base64
from html import escape
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import sha256,json_write
from .session_study import register

LABELS={'source_ensemble':'Frozen source ensemble','source_eegnet':'Frozen source EEGNet',
    'intercept':'Source + intercept','csp_lda':'Personal CSP + LDA','fbcsp_lda':'Personal filter-bank CSP + LDA',
    'spectral_logistic':'Personal spectral logistic','tangent_logistic':'Personal tangent logistic',
    'tangent_rbf_svm':'Personal tangent RBF SVM','anchored_tangent':'Anchored tangent residual',
    'anchored_dual':'Anchored dual residual (primary)','donor_selected_fusion':'Donor-selected fusion'}


def md_table(frame):
    values=frame.astype(str)
    return '\n'.join(['| '+' | '.join(values.columns)+' |','| '+' | '.join(['---']*len(values.columns))+' |']+
        ['| '+' | '.join(row)+' |' for row in values.itertuples(index=False,name=None)])


def report(plan):
    out,_=register(plan);verified=json.loads((out/'verification.json').read_text())
    if verified['status']!='PASS':raise RuntimeError('Verify before reporting')
    summary=pd.read_csv(out/'summary.csv');per=pd.read_csv(out/'metrics_by_subject.csv');paired=pd.read_csv(out/'paired_differences.csv')
    from .session_comparisons import save_pairwise
    all_pairs=save_pairwise(plan)
    primary=summary[(summary.method==plan['primary_method'])&(summary.support_budget==plan['primary_budget'])].iloc[0]
    primary_delta=paired[(paired.method==plan['primary_method'])&(paired.support_budget==plan['primary_budget'])].iloc[0]
    complete=json.loads((out/'complete.json').read_text());donor=json.loads((out/'donor_complete.json').read_text())
    intro=(f'The fixed primary hypothesis, anchored dual with {plan["primary_budget"]} first-session labels, scores '
        f'{100*primary.balanced_accuracy:.2f}% balanced accuracy. Its paired difference versus same-budget intercept calibration is '
        f'{100*primary_delta.mean_difference:+.2f} percentage points '
        f'[{100*primary_delta.ci_low:+.2f}, {100*primary_delta.ci_high:+.2f}]. ')
    intro+=('This does not establish a reliable positive gain.' if primary_delta.ci_low<=0 else
            'The descriptive participant interval is positive; this remains exploratory evidence.')
    full=summary[(summary.method=='donor_selected_fusion')&(summary.support_budget==144)].iloc[0]
    full_finding=(f'In the prespecified secondary 144-label condition, donor-selected fusion scores {100*full.balanced_accuracy:.2f}% '
        f'[{100*full.ci_low:.2f}, {100*full.ci_high:.2f}]%. It requires the complete first-session left/right support. '
        'It is the highest observed mean in this screen; that observation does not turn it into an independently selected confirmatory winner.')
    context='All values below use the SAME 1,296 session-E trials from nine people, sixteen electrodes and left/right classes. All 144 trials per person are retained. Labeled support is taken chronologically from the first session only. More labels change the calibration requirement.'
    limits='This dataset and earlier results were previously inspected. The new study is exploratory, not an untouched external confirmation. E labels never choose the method, C, mixture weight or budget. All settings use only other people\'s session T. Nine-person bootstrap intervals are descriptive and are not adjusted for multiple secondary comparisons or all shared-donor dependencies.'
    grid=summary.pivot(index='method',columns='support_budget',values='balanced_accuracy').reindex(LABELS)
    display=(100*grid).map(lambda v:f'{v:.2f}%');display.index=display.index.map(LABELS);display=display.reset_index(names='Method')
    display.columns=[str(c) for c in display.columns]
    lines=['# NeuroEdge: expanded cross-session calibration','',intro,'',full_finding,'',context,'',limits,'',
        '## Matched comparison: balanced accuracy','',md_table(display),'',
        'Columns are the number of labeled session-T support trials. Frozen source rows are repeated controls and use no personal labels. '
        'All target-only and anchored methods have identical support identities within each column. '
        'The 144-label arm is a full first-session supervised condition, not a zero-calibration result.','',
        '## All method/budget metrics and paired intervals','',
        '| Method | Labels | Balanced accuracy | 95% participant interval | Difference vs intercept (pp) | 95% paired interval (pp) | Macro F1 | Log loss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        d=paired[(paired.method==r.method)&(paired.support_budget==r.support_budget)].iloc[0]
        lines.append(f'| {LABELS[r.method]} | {r.support_budget} | {100*r.balanced_accuracy:.2f}% | [{100*r.ci_low:.2f}, {100*r.ci_high:.2f}]% | {100*d.mean_difference:+.2f} | [{100*d.ci_low:+.2f}, {100*d.ci_high:+.2f}] | {r.macro_f1:.4f} | {r.log_loss:.4f} |')
    lines+=['','## Fusion versus stronger personal controls at 144 labels','',
        '| Reference | Fusion difference (pp) | Descriptive paired interval (pp) |','|---|---:|---:|']
    for ref in ['fbcsp_lda','tangent_logistic','tangent_rbf_svm','anchored_tangent','anchored_dual']:
        row=all_pairs[(all_pairs.support_budget==144)&(
            ((all_pairs.method_a=='donor_selected_fusion')&(all_pairs.method_b==ref))|
            ((all_pairs.method_b=='donor_selected_fusion')&(all_pairs.method_a==ref)))].iloc[0]
        sign=1 if row.method_a=='donor_selected_fusion' else -1
        delta=sign*row.difference_a_minus_b;lo,hi=(row.ci_low,row.ci_high) if sign==1 else (-row.ci_high,-row.ci_low)
        lines.append(f'| {LABELS[ref]} | {100*delta:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines+=['','The small differences against stronger personal controls do not establish reliable superiority when their intervals include zero. '
        'All 330 unordered method-pair/budget comparisons, including adverse results, are in `all_pairwise_comparisons.csv`; intervals are not multiplicity-adjusted.','']
    lines+=['','## Selection, algorithms and limits','',
        'Each outer person is excluded from every donor model/parameter choice. For each of the other eight people, pseudo-support uses the first n T trials, '
        'and validation uses strictly later T runs. For the 144-label arm, selection deliberately emulates 96 labels because no later T run would remain. '
        'The outer person still receives exactly 144 support labels in the final fit. This selection-budget mismatch is a limitation.','',
        'Controls include CSP, filter-bank CSP, spectral logistic regression, tangent logistic regression and tangent RBF SVM. '
        'The source ensemble and EEGNet are the archived PhysioNet-only models. The proposed anchored residuals use support-local whitening and '
        'linear tangent/spectral kernels to correct that fixed source logit. Donor-selected fusion blends the intercept source with an equal mean of three personal decoders. '
        'These are project-specific combinations of established ideas, not a proven first-in-literature algorithm.','',
        'Tangent SVM uses sigmoid decision scores as uncalibrated probability surrogates; its log loss is reported with that limitation. '
        'Single-class supports retain the source for anchored adaptation and use the observed support prior for target-only models. '
        'Offline cue-aligned preprocessing does not establish real-time or clinical utility.','',
        'Exact equations, prior work, input requirements and selection rules: [SESSION_ADAPTATION_METHODS.md](../../docs/SESSION_ADAPTATION_METHODS.md).','',
        '## Historical conditions','',
        'Earlier direct transfer: selected ensemble 73.69%, EEGNet 66.94%, evaluated on both BNCI sessions (2,592 trials). '
        'Earlier external LOSO: 73.03% versus 67.59%, trained on the other eight people\'s two sessions. '
        'Earlier small-label calibration used the same E session as this new study; its source ensemble was 72.15%, intercept/16 was 73.07%, '
        'hybrid Gaussian kernel/16 was 72.99%, and geometry kernel/16 was 74.00%. '
        'These historical conditions have different training or test sets; do not subtract their headline percentages to claim a matched gain. '
        'The complete archived reports and all adverse results remain included.','',
        '## Reproduction and saved evidence','',
        '```bash','python scripts/restore_external_data.py','python -m neuroedge.session_study all --plan configs/session_adaptation.yaml','python -m pytest -q','```','',
        'The restoration command requires every original raw and processed input fingerprint to match. For a changed method or environment that changes bytes, '
        'copy the YAML to a new name and output directory; never remove identity guards or overwrite this study.','',
        f'Donor development: {donor["fits"]} fit/evaluation jobs; {donor["seconds"]:.1f} seconds on this host. '
        f'Final evaluation: {complete["model_fits"]} saved fit/evaluation artifacts plus 54 deterministic fusion results; {complete["seconds"]:.1f} seconds. '
        'These counts include unchanged source-control evaluations and are not counts of distinct new algorithms.','',
        f'Independent verification: **PASS**, {verified["checkpoint_replays"]} checkpoint replays and support refits, '
        f'{verified["donor_choices_reconstructed"]} donor-only parameter choices, {verified["final_prediction_rows"]:,} actual prediction rows.','',
        '- `donor_predictions.csv`: every C and donor validation trial.',
        '- `donor_splits.json`: support/validation IDs and the 144-to-96 selection rule.',
        '- `selection.json`: every outer-person C/weight and all donor scores.',
        '- `fit_receipts.json` and `models/`: all personal support IDs and saved models.',
        '- `predictions.csv`: all actual session-E predictions.',
        '- `metrics_by_subject.csv`, `summary.csv`, `paired_differences.csv`: all participant results and uncertainty.',
        '- `all_pairwise_comparisons.csv`: all 330 method-pair/budget comparisons, including stronger controls.',
        '- `protocol.json`, `environment.json`, `verification.json`: exact evidence identities.','']
    (out/'REPORT.md').write_text('\n'.join(lines));(out/'RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#64748b','#1d4ed8','#a21caf','#0f766e','#ea580c','#6d28d9','#be123c']
    chosen=['intercept','fbcsp_lda','tangent_logistic','tangent_rbf_svm','anchored_tangent','anchored_dual','donor_selected_fusion']
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    for method,color in zip(chosen,colors):
        g=summary[summary.method==method].sort_values('support_budget');ax.plot(g.support_budget,100*g.balanced_accuracy,'o-',label=LABELS[method],color=color)
    ax.set(xscale='log',xticks=plan['support_budgets'],xticklabels=plan['support_budgets'],ylim=(40,100),
        xlabel='Labeled first-session support trials (log scale)',ylabel='Session-E participant-mean balanced accuracy (%)',
        title='Same nine people and same second-session trials at every budget')
    ax.axhline(50,color='#94a3b8',ls='--');ax.grid(alpha=.15);ax.legend(fontsize=8,loc='best')
    fig.savefig(out/'calibration_comparison.png',dpi=180);plt.close(fig)
    d=paired[(paired.support_budget==plan['primary_budget'])&(~paired.method.isin(['source_eegnet','source_ensemble','intercept']))]
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    x=100*d.mean_difference.to_numpy();ax.errorbar(x,np.arange(len(d)),xerr=np.stack([x-100*d.ci_low,100*d.ci_high-x]),fmt='o',capsize=4,color='#1d4ed8')
    ax.set(yticks=np.arange(len(d)),yticklabels=d.method.map(LABELS),xlabel='Difference vs same-budget intercept (percentage points)',title='16 labels: descriptive paired participant intervals');ax.axvline(0,color='#94a3b8',ls='--');ax.grid(axis='x',alpha=.15)
    fig.savefig(out/'paired_16.png',dpi=180);plt.close(fig)
    g=per[per.support_budget==144].pivot(index='subject',columns='method',values='balanced_accuracy').reindex(columns=chosen)
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained');im=ax.imshow(100*g,aspect='auto',vmin=40,vmax=100,cmap='viridis')
    ax.set(xticks=np.arange(len(chosen)),xticklabels=[LABELS[m].replace(' ','\n',1) for m in chosen],yticks=np.arange(9),yticklabels=[f'A{p-1000:02d}' for p in g.index],title='144-label condition: every held-out second-session participant')
    ax.tick_params(axis='x',labelsize=8)
    for i in range(len(g)):
        for j in range(len(chosen)):ax.text(j,i,f'{100*g.iloc[i,j]:.1f}',ha='center',va='center',color='white' if g.iloc[i,j]<.7 else '#0f172a',fontsize=9)
    fig.colorbar(im,ax=ax,label='Balanced accuracy (%)');fig.savefig(out/'participants_144.png',dpi=180);plt.close(fig)
    figures=''.join('<figure><img alt="'+name.replace('_',' ')+'" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'"></figure>' for name in ['calibration_comparison.png','paired_16.png','participants_144.png'])
    detailed=summary.merge(paired,on=['method','support_budget','subjects'],suffixes=('','_difference'));detailed['method']=detailed.method.map(LABELS)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuroEdge - session adaptation</title><style>body{font:16px system-ui;line-height:1.65;max-width:1240px;margin:36px auto;padding:0 20px;color:#17243b;background:#f3f6fb}section{background:white;padding:24px;margin:20px 0;border:1px solid #dce4ef;border-radius:14px}h1{font-size:34px;line-height:1.2}h2{font-size:23px}img{max-width:100%;height:auto}figure{margin:24px 0}.table{overflow-x:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:9px;text-align:right;border-bottom:1px solid #dde4ee}th:first-child,td:first-child{text-align:left}select{font:inherit;padding:8px;border:1px solid #bbc9d9;border-radius:6px}.note{border-left:4px solid #d97706;padding-left:14px}a{color:#1d4ed8}</style><h1>NeuroEdge: cross-session adaptation</h1>'''
    page+='<section><h2>Primary and secondary findings</h2><p>'+escape(intro)+'</p><p>'+escape(full_finding)+'</p><p>'+escape(context)+'</p><p class="note">'+escape(limits)+'</p></section>'
    page+='<section><h2>All methods, matched test trials</h2><p>Columns show labeled support budgets. Values are balanced accuracy.</p><div class="table">'+display.to_html(index=False,border=0)+'</div></section>'
    page+='<section><h2>Measured comparisons</h2>'+figures+'</section>'
    page+='<section><h2>All raw summary metrics</h2><label for="budget">Support budget: </label><select id="budget"><option value="all">All</option>'+''.join(f'<option>{b}</option>' for b in plan['support_budgets'])+'</select><p>Accuracy and interval fields below are fractions. Paired intervals are descriptive.</p><div class="table" id="detail">'+detailed.to_html(index=False,border=0,float_format=lambda x:f'{x:.5f}')+'</div></section>'
    page+='<section><h2>Evidence and interpretation</h2><p>Independent verification PASS: all 540 saved artifacts replayed and refitted. See <a href="REPORT.md">the complete English report</a> for methods, all data, negative results and limits. Larger calibration budgets are a separate supervision condition.</p><p><a href="../research_summary/results.html">Earlier research</a> · <a href="../../docs/SESSION_ADAPTATION_METHODS.md">Exact new methods</a></p></section>'
    page+='''<script>const s=document.getElementById('budget');const t=document.querySelector('#detail table');const j=Array.from(t.tHead.rows[0].cells).findIndex(c=>c.textContent==='support_budget');s.onchange=()=>{for(const r of t.tBodies[0].rows)r.style.display=s.value==='all'||r.cells[j].textContent.trim()===s.value?'':'none'};</script></html>'''
    (out/'results.html').write_text(page)
    json_write(out/'presentation_verification.json',{'status':'PASS','report_source_sha256':sha256(__file__),
        'scientific_verification_sha256':sha256(out/'verification.json'),'summary_sha256':sha256(out/'summary.csv'),
        'language':'English','all_method_budget_conditions':len(summary)})
    print(out/'REPORT.md',flush=True)
