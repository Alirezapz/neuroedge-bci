"""English summary of frozen source choices and separate operating conditions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import json_write,sha256
from .context_protocol import interval
from .english_reports import portable_page
FA={'csp_reference':'CSP reference','eegnet_reference':'EEGNet reference','selected_zero_calibration':'Selected task ensemble','selected_pre_cue':'Selected pre-cue ensemble'}
TITLES={'external_transfer':'Direct transfer from PhysioNet','external_loso':'External LOSO on eight other people','personal_16':'Session E after 16 first-session labels'}
ORDER=list(FA)

def report(context, gated, personal, personal_gated, output='runs/research_summary'):
    root = Path(context['_root'])
    out = root / output
    parent = root / context['output_dir']
    child = root / gated['output_dir']
    pdir = root / personal['output_dir']
    pgdir = root / personal_gated['output_dir']
    protocol = json.loads((out / 'protocol.json').read_text())
    selection = json.loads((out / 'selection.json').read_text())
    if not protocol['registered_before_external_acquisition'] or not selection['frozen_before_external_acquisition'] or selection['external_used_for_selection']:
        raise AssertionError('Overall choice must precede external acquisition.')
    p = json.loads((parent / 'selection.json').read_text())
    g = json.loads((child / 'selection.json').read_text())
    if selection['parent_selection_sha256'] != sha256(parent / 'selection.json') or selection['gated_selection_sha256'] != sha256(child / 'selection.json'):
        raise AssertionError('Source choice dependencies changed.')
    choices = [{'origin': 'context', 'model': 'selected_pre_cue', 'result': p['selected']['pre_cue']}, {'origin': 'gated', 'model': g['selected']['variant'], 'result': g['selected']}]
    expected = sorted(choices, key=lambda r: (-r['result']['balanced_accuracy'], r['result']['log_loss'], r['result']['name']))[0]
    if selection['pre_cue'] != expected or selection['zero_calibration']['result'] != p['selected']['zero_calibration']:
        raise AssertionError('Overall source-choice rule differs.')
    dependencies = {}
    for label, directory in [('context', parent), ('gated', child), ('personal', pdir), ('gated_personal', pgdir)]:
        if json.loads((directory / 'verification.json').read_text())['status'] != 'PASS':
            raise RuntimeError('Verify every study first.')
        dependencies[label] = sha256(directory / 'verification.json')
    tables = {}
    paired = []
    bootstrap = {'evaluation': context['evaluation']}
    for role in TITLES:
        rows = []
        scores = {}
        for model in ORDER:
            chosen = selection['pre_cue'] if model == 'selected_pre_cue' else {'origin': 'context', 'model': model}
            origin = chosen['origin']
            source_model = chosen['model']
            if role == 'personal_16':
                directory = pgdir if origin == 'gated' else pdir
                frame = pd.read_csv(directory / 'summary.csv')
                frame = frame[frame.support_budget == 16]
                per = pd.read_csv(directory / 'metrics_by_subject.csv')
                per = per[per.support_budget == 16]
            else:
                directory = child if origin == 'gated' else parent
                frame = pd.read_csv(directory / role / 'summary.csv')
                per = pd.read_csv(directory / role / 'metrics_by_subject.csv')
                if origin == 'gated' and role == 'external_loso':
                    source_model = 'selected_fusion'
            row = frame[frame.model == source_model].iloc[0].to_dict()
            row.update(model=model, origin=origin, source_model=source_model, protocol=role)
            rows.append(row)
            scores[model] = per[per.model == source_model].set_index('subject').balanced_accuracy
        tables[role] = pd.DataFrame(rows)
        for selected, baseline in [('selected_zero_calibration', 'eegnet_reference'), ('selected_pre_cue', 'selected_zero_calibration')]:
            delta = scores[selected] - scores[baseline]
            lo, hi = interval(delta, bootstrap)
            paired.append(dict(protocol=role, selected=selected, reference=baseline, mean_difference=float(delta.mean()), ci_low=lo, ci_high=hi, subjects=len(delta), same_context_requirement=selected != 'selected_pre_cue'))
    summary = pd.concat(tables.values(), ignore_index=True)
    summary.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(paired).to_csv(out / 'paired_differences.csv', index=False)
    nval = len(pd.read_csv(parent / 'validation_summary.csv'))
    english=['# NeuroEdge: frozen-source research summary','','65 base configurations and 195 group fits were evaluated on source development. All 104 previously inspected PhysioNet participants are development data. The original external choices were frozen before BNCI acquisition. All methods use 16 channels.','', 'Direct transfer, external LOSO, pre-cue context and labeled personal calibration have different training/input requirements. Do not treat their headline percentages as matched gains. Intervals resample people and are descriptive.','']
    for role,table in tables.items():
        english += ['## '+TITLES[role],'','| Model | Balanced accuracy | 95% participant interval | Accuracy | Log loss |','|---|---:|---:|---:|---:|']
        for r in table.itertuples():
            english.append(f'| {FA[r.model]} | {100*r.balanced_accuracy:.2f}% | [{100*r.ci_low:.2f}, {100*r.ci_high:.2f}]% | {100*r.accuracy:.2f}% | {r.log_loss:.4f} |')
        english.append('')
    english += ['## Paired differences','']
    for r in paired:
        english.append(f"- {r['protocol']}, {r['selected']} versus {r['reference']}: {100*r['mean_difference']:+.2f} pp [{100*r['ci_low']:+.2f}, {100*r['ci_high']:+.2f}].")
    english += ['','Quality-conditioned gating did not outperform its matched fixed-gate control; all negative findings and budgets are preserved. An interval including zero does not establish a reliable gain. Filter banks, covariance geometry, ERD, mirroring and residual calibration have prior work; no worldwide novelty, SOTA, clinical or real-time claim is made.','', 'Complete methods and data: ../context_fusion/REPORT.md, ../gated_context/REPORT.md, ../personalization/REPORT.md, ../personalization_gated/REPORT.md. New exploratory calibration expansion: ../session_adaptation/REPORT.md.','']
    for name in ['REPORT.md','RESULTS.md']:(out/name).write_text('\n'.join(english),encoding='utf-8')
    portable_page(out,'NeuroEdge: research comparison','Frozen-source choices. Training and calibration requirements differ between protocols.',{TITLES[k]:v for k,v in tables.items()},['overall_accuracy.png'])
    json_write(out/'verification.json',dict(status='PASS',underlying_verification_sha256=dependencies,overall_source_selection_recomputed=True,external_used_for_overall_selection=False,source_selection_sha256=sha256(out/'selection.json'),summary_sha256=sha256(out/'summary.csv'),report_source_sha256=sha256(__file__)))
    return tables


def attach_residual(context,residual,output='runs/research_summary'):
    root=Path(context['_root']);out=root/output;directory=root/residual['output_dir']
    if json.loads((directory/'verification.json').read_text())['status']!='PASS':raise RuntimeError('Verify residual study first')
    table=pd.read_csv(directory/'summary.csv');table=table[table.support_budget.isin([0,residual['primary_budget']])]
    table.to_csv(out/'personal_kernels_summary.csv',index=False)
    primary=residual['primary_kernel'];diff=pd.read_csv(directory/'paired_differences.csv')
    d=diff[(diff.model==primary)&(diff.reference=='same_budget_bias')].iloc[0]
    marker='<!-- RESIDUAL_PERSONAL -->'
    section=f"## Supplementary source-prior kernels\n\nPrimary hybrid/16: {100*d.mean_difference:+.2f} pp versus the same-budget intercept, [{100*d.ci_low:+.2f}, {100*d.ci_high:+.2f}]. This does not establish a reliable positive gain. All methods use the same E trials. Geometry and spectral secondary kernels are reported in ../residual_personal/REPORT.md with all budgets, controls and negative findings.\n"
    for name in ['REPORT.md','RESULTS.md']:
        p=out/name;p.write_text(p.read_text().split(marker)[0].rstrip()+'\n\n'+marker+'\n\n'+section)
    summary=pd.read_csv(out/'summary.csv');tables={TITLES[k]:g for k,g in summary.groupby('protocol')};tables['Supplementary kernels']=table
    portable_page(out,'NeuroEdge: complete research summary','Separate training, input and calibration requirements; all results and failures retained.',tables,['overall_accuracy.png'])
    receipt=json.loads((out/'verification.json').read_text());receipt['underlying_verification_sha256']['residual']=sha256(directory/'verification.json');receipt['personal_kernels_summary_sha256']=sha256(out/'personal_kernels_summary.csv');receipt['report_source_sha256']=sha256(__file__)
    json_write(out/'verification.json',receipt);return table
