"""English report of the entire frozen covariance screen, including failures."""
import base64
from html import escape
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import json_write,sha256
from .covariance_study import register
from .session_study import interval

LABELS={'fbcsp_lda':'Shrinkage FBCSP + LDA','fbcsp_svm':'FBCSP RBF SVM','selected_fbcsp':'Fisher-selected FBCSP SVM',
    'tangent_logistic':'Tangent logistic','tangent_svm':'Tangent RBF SVM','joint_svm':'Joint CSP/tangent SVM',
    'power_means':'Multi-order covariance prototypes','anchored_multiview':'Anchored multiview residual',
    'donor_stack':'Nested donor stack (primary)','equal_probability':'Equal probability mean','equal_logit':'Equal logit mean',
    'reference_previous_fusion':'Previous fusion (16 sensors, strict)','reference_previous_tangent':'Previous tangent SVM (16, strict)',
    'reference_intercept':'Previous intercept (16, strict)'}


def table_md(frame):
    f=frame.astype(str)
    return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(row)+' |' for row in f.itertuples(index=False,name=None)])


def report(plan):
    out,_=register(plan);verification=json.loads((out/'verification.json').read_text())
    if verification['status']!='PASS':raise RuntimeError('Verify before reporting')
    summary=pd.read_csv(out/'summary.csv');per=pd.read_csv(out/'metrics_by_subject.csv');paired=pd.read_csv(out/'paired_differences.csv');donor=pd.read_csv(out/'donor_summary.csv');all_pairs=[]
    for (sensors,condition,budget),g in per.groupby(['sensors','condition','support_budget']):
        pivot=g.pivot(index='subject',columns='method',values='balanced_accuracy')
        for a,b in combinations(sorted(pivot.columns),2):
            d=pivot[a]-pivot[b];lo,hi=interval(d,plan)
            all_pairs.append({'sensors':int(sensors),'condition':condition,'support_budget':int(budget),'method_a':a,'method_b':b,'difference':float(d.mean()),'ci_low':lo,'ci_high':hi})
    pd.DataFrame(all_pairs).to_csv(out/'all_pairwise_comparisons.csv',index=False)
    primary=summary[(summary.sensors==plan['primary_sensors'])&(summary.condition==plan['primary_condition'])&(summary.support_budget==plan['primary_budget'])&(summary.method==plan['primary_method'])].iloc[0]
    d=paired[(paired.sensors==plan['primary_sensors'])&(paired.condition==plan['primary_condition'])&(paired.support_budget==plan['primary_budget'])&(paired.method==plan['primary_method'])].iloc[0]
    finding=(f'The fixed primary donor stack, 16 sensors, strict prediction and 144 T labels, scores {100*primary.balanced_accuracy:.2f}% '
        f'balanced accuracy [{100*primary.ci_low:.2f}, {100*primary.ci_high:.2f}]%. '
        f'The paired difference versus the previous same-budget strict fusion is {100*d.difference:+.2f} pp '
        f'[{100*d.ci_low:+.2f}, {100*d.ci_high:+.2f}]. ')
    finding+=('This interval does not establish reliable positive improvement.' if d.ci_low<=0 else 'The descriptive interval is positive; independent confirmation remains necessary.')
    intro='Previously inspected dataset; exploratory continuation. Nine people, binary left/right imagery, the same 1,296 E trials. Sensor and unlabeled-batch requirements are reported separately. All parameters and combinations were frozen using T data only before new E scoring.'
    best=summary[~summary.method.str.startswith('reference_')].sort_values('balanced_accuracy',ascending=False).iloc[0]
    best_note=f'The highest observed secondary mean is {100*best.balanced_accuracy:.2f}% for {LABELS[best.method]}, {best.sensors} sensors, {best.condition}, {best.support_budget} T labels. This is a post-evaluation descriptive maximum, not a newly selected confirmatory winner.'
    lines=['# NeuroEdge: expanded covariance and donor-stack study','',finding,'',intro,'',best_note,'',
        '## Requirements and fair interpretation','',
        '- Strict: every fitted transform uses T support only. Prediction on one prepared epoch uses no statistics from other E epochs. Fixed signal filtering still remains offline.',
        '- Batch-unlabeled: the complete unlabeled E batch supplies a covariance center. This uses future unlabeled trials and is not strict, causal, zero-calibration or independent-trial performance.',
        '- 22 sensors: all EEG electrodes, with a 22-channel common average; EOG is excluded. This changes the sensor requirement from the previous16-channel study.',
        '- All methods require the indicated personal T labels. Reference rows always retain their actual16-sensor, strict requirement even in a22-sensor or batch panel.',
        '- The previous fusion benchmark is85.11% at144 labels. Its values use exactly the same E trials as this report.','']
    sections={}
    for (sensors,condition),g in summary.groupby(['sensors','condition']):
        pivot=g.pivot(index='method',columns='support_budget',values='balanced_accuracy').reindex(LABELS)
        display=(100*pivot).map(lambda v:f'{v:.2f}%');display.index=display.index.map(LABELS);display=display.reset_index(names='Method');display.columns=[str(c) for c in display.columns]
        title=f'{sensors} sensors / {condition}'
        sections[title]=display;lines+=['## '+title,'','Columns are first-session labeled support counts. Values are participant-mean balanced accuracy.','',table_md(display),'']
    lines+=['## All intervals, losses and matched reference differences','',
        '| Sensors | Condition | Labels | Method | BA | 95% participant interval | Difference vs previous fusion (pp) | 95% paired interval (pp) | Log loss |',
        '|---:|---|---:|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        d=paired[(paired.sensors==r.sensors)&(paired.condition==r.condition)&(paired.support_budget==r.support_budget)&(paired.method==r.method)].iloc[0]
        lines.append(f'| {r.sensors} | {r.condition} | {r.support_budget} | {LABELS[r.method]} | {100*r.balanced_accuracy:.2f}% | [{100*r.ci_low:.2f}, {100*r.ci_high:.2f}]% | {100*d.difference:+.2f} | [{100*d.ci_low:+.2f}, {100*d.ci_high:+.2f}] | {r.log_loss:.4f} |')
    lines+=['','For22-sensor and batch-unlabeled rows, subtraction from the old16-sensor strict reference changes the input requirement. It is a pipeline comparison, not an isolated algorithm gain. Intervals resample nine people; they are descriptive and unadjusted for secondary comparisons, overlap in donor cohorts and prior dataset inspection.','',
        '## Candidate screen and nested selection','',
        f'{donor.candidate.nunique()} fixed configurations were evaluated in {len(donor):,} donor fits across two sensor conditions, two input conditions, three budgets and nine T donors. '
        'Family settings for each outer person exclude that person completely. Meta-training predictions for donor d additionally exclude d from family selection. '
        'Stack C uses conditional leave-one-donor-out meta-validation. These internal scores are selection evidence, not an independent accuracy estimate. '
        'The144-label arm emulates96 labels during donor selection to retain later T validation runs; the final personal fit receives144.','',
        'The screen covers shrinkage FBCSP, nonlinear and Fisher-selected CSP, tangent logistic/SVM, joint CSP/tangent kernels, multi-order covariance prototypes and a source-anchored multiview residual. '
        'The donor stack and multiview residual are project-specific combinations of established components; worldwide novelty is not established. '
        'Simplified power-mean features are not a complete reproduction of affine-invariant Mean Field. SVM sigmoid scores are uncalibrated surrogates.','',
        'Exact mathematics, candidate grid, prior work and limitations: [COVARIANCE_LAB_METHODS.md](../../docs/COVARIANCE_LAB_METHODS.md).','',
        '## Reproduction and evidence','',
        '```bash','python scripts/freeze_covariance_inputs.py --plan configs/covariance_lab_corrected.yaml','python -m neuroedge.covariance_study all --plan configs/covariance_lab_corrected.yaml --workers 4','python -m pytest -q','```','',
        'The complete earlier BNCI inputs must be restored first; see README.md. New methods or changed data require a fresh YAML/output identity. No method is modified after seeing this study\'s E scores.','',
        f'Independent verification PASS: {verification["checkpoint_replays"]} checkpoint replays, '
        f'{verification["independent_support_meta_refits"]} refits, {verification["nested_settings_checked"]} nested settings, '
        f'{verification["donor_candidate_metrics_checked"]:,} donor candidate metrics and {verification["prediction_rows"]:,} actual E prediction rows.','',
        '- `donors/*/predictions.csv`, `scores.csv`, `complete.json`: every donor candidate and split.',
        '- `donor_summary.csv`, `selection.json`, `nested_meta_predictions.csv`: all choices, scores and nested identities.',
        '- `models/`, `fit_receipts.json`: every selected model, support identity and hash.',
        '- `predictions.csv`, `metrics_by_subject.csv`, `summary.csv`: all actual predictions and subject metrics.',
        '- `paired_differences.csv`, `all_pairwise_comparisons.csv`: all retained positive and negative comparisons.',
        '- `protocol.json`, `verification.json`: frozen implementation and independent verification.',
        'The initial diagnostic run in `../covariance_lab` was invalidated after a matrix-exponential eigenvalue-floor error was found in its power-mean feature. '
        'Its original implementation and T-screen evidence are retained for audit, while incomplete model output is omitted. '
        'All results in this report come from a fresh corrected protocol and recomputed donor/final fits. '
        'A constant-SPD regression detects the original error and confirms the corrected matrix identity.',
        'Raw EEG is public attributed data and is not redistributed. This offline study does not establish clinical or real-time performance.','']
    (out/'REPORT.md').write_text('\n'.join(lines));(out/'RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=['reference_previous_fusion','fbcsp_svm','selected_fbcsp','tangent_svm','joint_svm','anchored_multiview','donor_stack','equal_probability']
    colors=['#64748b','#2563eb','#9333ea','#0f766e','#f97316','#7c3aed','#be123c','#0891b2']
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained',sharey=True)
    for ax,(sensors,condition) in zip(axes.ravel(),[(16,'strict'),(22,'strict'),(16,'batch_unlabeled'),(22,'batch_unlabeled')]):
        for method,color in zip(selected,colors):
            g=summary[(summary.sensors==sensors)&(summary.condition==condition)&(summary.method==method)].sort_values('support_budget');ax.plot(g.support_budget,100*g.balanced_accuracy,'o-',label=LABELS[method],color=color)
        ax.set(xlabel='Personal session-T labels',ylabel='Session-E balanced accuracy (%)',ylim=(45,100),xticks=[16,64,144],title=f'{sensors} sensors / {condition}');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7,loc='lower right');fig.suptitle('Same nine people and E trials; input requirements differ between panels')
    fig.savefig(out/'condition_comparison.png',dpi=160);plt.close(fig)
    d=paired[(paired.sensors==16)&(paired.condition=='strict')&(paired.support_budget==144)&(~paired.method.str.startswith('reference_'))]
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained');x=100*d.difference.to_numpy();ax.errorbar(x,np.arange(len(d)),xerr=np.stack([x-100*d.ci_low,100*d.ci_high-x]),fmt='o',capsize=4)
    ax.axvline(0,color='#94a3b8',ls='--');ax.set(yticks=np.arange(len(d)),yticklabels=d.method.map(LABELS),xlabel='Difference vs previous fusion (percentage points)',title='Matched primary requirement: 16 sensors, strict, 144 labels');ax.grid(axis='x',alpha=.15)
    fig.savefig(out/'paired_primary.png',dpi=160);plt.close(fig)
    images=''.join('<figure><img alt="'+name.replace('_',' ')+'" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'"></figure>' for name in ['condition_comparison.png','paired_primary.png'])
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuroEdge - expanded covariance experiments</title><style>body{font:16px system-ui;line-height:1.65;max-width:1280px;margin:30px auto;padding:20px;color:#17243b;background:#f4f7fb}section{background:white;padding:22px;margin:20px 0;border:1px solid #dce4ef;border-radius:12px}h1{line-height:1.2}img{max-width:100%;height:auto}figure{margin:20px 0}.table{overflow-x:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dce4ef;text-align:right}td:first-child,th:first-child{text-align:left}.note{border-left:4px solid #d97706;padding-left:14px}select{font:inherit;padding:7px}a{color:#1d4ed8}</style><h1>NeuroEdge: expanded covariance study</h1>'
    page+='<section><h2>Frozen primary finding</h2><p>'+escape(finding)+'</p><p>'+escape(intro)+'</p><p>'+escape(best_note)+'</p><p class="note">Batch-unlabeled rows use the entire unlabeled E batch. Twenty-two sensors require more electrodes. Neither may be presented as strict sixteen-sensor performance.</p></section>'
    for title,table in sections.items():page+='<section><h2>'+escape(title)+'</h2><p>Columns are personal T labels; old reference rows always use16 sensors and strict prediction.</p><div class="table">'+table.to_html(index=False,border=0)+'</div></section>'
    page+='<section><h2>Measured comparisons</h2>'+images+'</section><section><h2>All numerical metrics</h2><p>Accuracy and interval fields below are fractions. All original CSV tables and actual predictions are included.</p><div class="table">'+summary.to_html(index=False,border=0,float_format=lambda x:f'{x:.5f}')+'</div></section>'
    page+='<section><h2>Methods and evidence</h2><p>Independent verification PASS. All candidates, negative outcomes, support IDs, checkpoints and descriptive paired intervals are retained. See <a href="REPORT.md">the complete English report</a> and <a href="../../docs/COVARIANCE_LAB_METHODS.md">exact methods</a>.</p><p><a href="../session_adaptation/results.html">Previous calibration study</a> · <a href="../research_summary/results.html">Original research comparison</a></p></section></html>'
    (out/'results.html').write_text(page);json_write(out/'presentation_verification.json',{'status':'PASS','language':'English','conditions':len(summary),
        'paired_comparisons':len(all_pairs),'source_sha256':sha256(__file__),'scientific_verification_sha256':sha256(out/'verification.json')})
    print(out/'REPORT.md',flush=True)
