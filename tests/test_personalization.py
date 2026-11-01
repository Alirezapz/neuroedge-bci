"""Synthetic session-disjoint calibration checks, never research accuracy."""
import copy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from neuroedge.personalization import adapt,experiment,fit_intercept,load_plan,register,verify

ROOT=Path(__file__).resolve().parents[1]


def test_personal_intercept_corrects_offset_without_reversing_rank():
    p=load_plan(ROOT/'configs/personalization.yaml');y=np.asarray([0,1]*8)
    probabilities=np.where(y,.99,.9);b,status=fit_intercept(probabilities,y,p['adaptation'])
    corrected=adapt(probabilities,b,p['adaptation'])
    assert b<0 and np.mean(corrected.argmax(axis=1)==y)==1
    assert corrected[y==1,1].min()>corrected[y==0,1].max()
    b,status=fit_intercept(probabilities,np.ones(16),p['adaptation'])
    assert b==0 and status=='single_class_support_keep_source'


def fixture_run(tmp_path):
    root=tmp_path;cfg=root/'configs';cfg.mkdir();parent=root/'runs/parent';parent.mkdir(parents=True)
    (cfg/'parent.yaml').write_text(yaml.safe_dump({'external':{'raw_dir':'data/NEVER_DOWNLOADED_SYNTHETIC'}}))
    (parent/'protocol.json').write_text(json.dumps({'sha256':'synthetic_parent_fixture_only'}))
    (parent/'selection.json').write_text(json.dumps({'synthetic_source_selection':True}))
    (parent/'verification.json').write_text(json.dumps({'status':'PASS','scope':'synthetic software fixture, no real study'}))
    out=parent/'external_transfer';out.mkdir();records=[]
    for model in ['csp_reference','eegnet_reference']:
        for subject in [1001,1002]:
            for session,count in [(1,32),(2,144)]:
                for trial in range(count):
                    label=trial%2;probability=.99 if label else .9
                    records.append({'trial_id':f'SYNTHETIC_{subject}_{session}_{trial:03d}',
                        'subject':subject,'session':session,'run':trial//16,'onset':trial*8.,'label':label,
                        'model':model,'arm':'zero_calibration','prob_right':probability,'prediction':1})
    pd.DataFrame(records).to_csv(out/'predictions.csv',index=False)
    plan=load_plan(ROOT/'configs/personalization.yaml');plan['_root']=str(root)
    plan['parent_output']='runs/parent';plan['parent_plan']='configs/parent.yaml';plan['output_dir']='runs/personal'
    plan['bootstrap_iterations']=20
    return plan,parent


def test_personal_session_split_chronology_metrics_and_tamper(tmp_path):
    plan,parent=fixture_run(tmp_path);register(plan);experiment(plan)
    assert verify(plan)['status']=='PASS'
    from neuroedge.personalization_verify import verify as independent_verify
    from neuroedge.personalization_report import report
    assert independent_verify(plan)['summary_metrics_and_participant_intervals_recomputed']
    report(plan)
    out=tmp_path/'runs/personal';frame=pd.read_csv(out/'predictions.csv')
    assert set(frame.session)=={2} and frame.trial_id.str.contains('SYNTHETIC').all()
    receipts=json.loads((out/'adaptation_receipts.json').read_text())
    assert all(all('_1_' in t for t in r['support_trial_ids']) for r in receipts)
    with (out/'predictions.csv').open('a') as f:f.write('\n')
    with pytest.raises(AssertionError,match='predictions or receipts changed'):verify(plan)


def test_personal_registration_rejects_changed_methods_and_late_registration(tmp_path):
    plan,parent=fixture_run(tmp_path);register(plan);changed=copy.deepcopy(plan)
    changed['adaptation']['prior_standard_deviation']=2
    with pytest.raises(RuntimeError,match='Frozen personal-calibration'):register(changed)
    plan['output_dir']='runs/too_late';(parent/'external_data').mkdir();(parent/'external_data/manifest.json').write_text('{}')
    with pytest.raises(RuntimeError,match='before external acquisition'):register(plan)
