"""Standalone SYNTHETIC software fixture for full selection/confirmation/verify.

Creates temporary arrays only. No PhysioNet download, real confirmation access,
or synthetic research-result substitution is permitted here.
"""
from pathlib import Path
import copy
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from neuroedge.config import json_write,sha256
from neuroedge.data import CHANNELS_16
import neuroedge.frontier as engine
from neuroedge.frontier_protocol import load_plan,freeze
from neuroedge.frontier_data import Cohort,empirical_covariances,load_cohort
from neuroedge.frontier_alignment import load_rest
from neuroedge.frontier_verify import verify
from neuroedge.frontier_report import report


def main():
    root_project=Path(__file__).resolve().parents[1]
    plan=load_plan(root_project/'configs/frontier.yaml')
    with tempfile.TemporaryDirectory() as scratch:
        root=Path(scratch)
        plan.update(_root=str(root),name='synthetic_software_fixture',output_dir='runs/software_fixture',
                    scope='SYNTHETIC SOFTWARE TEST ONLY; never research EEG evidence',
                    discovery_subjects=[1,2,3,4,5,6],validation_subjects=[7,8],confirmation_subjects=[81,82])
        plan['search']['threads']=2
        plan['quality']['minimum_confirmation_subjects']=2
        plan['confirmation']['bootstrap_iterations']=100
        plan['neural']['final_seeds']=[7]
        out=root/plan['output_dir']
        specs=[
            {'family':'raw_csp','channels':16,'window':'late','bands':[1],'name':'csp16_baseline'},
            {'family':'neural','channels':16,'window':'late','bands':[1],'name':'eegnet16_baseline',
             'architecture':'eegnet_baseline','augmentation':'none','dropout':.5,'fixed_epochs':25},
            {'family':'power','channels':16,'window':'late','bands':list(range(2,8)),
             'classifier':'logistic','C':1.,'name':'power_software_fixture'}]
        calibrated=copy.deepcopy(specs[-1]);calibrated.update(arm='rest_calibrated',name='power_rest_software_fixture')
        specs.append(calibrated)
        names=CHANNELS_16+[f'X{i:02d}' for i in range(48)]
        rng=np.random.default_rng(37)
        for role,subjects in [('development_data',list(range(1,9))),('confirmation_data',[81,82])]:
            folder=root/'fixture_arrays'/role;folder.mkdir(parents=True)
            n=len(subjects)*6
            labels=np.tile([0,1],n//2)
            X=rng.normal(size=(n,64,448)).astype('float32')
            X[:,0:3]*=(1+labels[:,None,None]*.5)
            C=empirical_covariances(X).astype('float32')
            cov=np.broadcast_to(C[:,None,None],(n,2,8,64,64)).copy()
            np.save(folder/'wide.npy',X);np.save(folder/'smr.npy',X);np.save(folder/'cov.npy',cov)
            meta=pd.DataFrame({'trial_id':[f'S{s:03d}_R04_T{t:03d}' for s in subjects for t in range(6)],
                'subject':np.repeat(subjects,6),'run':4,'onset':np.tile(np.arange(6)*8.,len(subjects)),
                'label':labels})
            meta.to_csv(folder/'trials.csv',index=False)
            manifest={'subjects':subjects,'n_trials':n,'channels':names,
                'data_source_sha256':sha256(root_project/'src/neuroedge/frontier_data.py'),
                'files':{name:{'path':str((folder/name).relative_to(root)),'sha256':sha256(folder/name)}
                         for name in ['wide.npy','smr.npy','cov.npy','trials.csv']}}
            directory=out/role;directory.mkdir(parents=True)
            json_write(directory/'manifest.json',manifest)
            rest=np.broadcast_to(np.eye(64),(len(subjects),8,64,64)).copy()
            np.save(folder/'rest.npy',rest)
            json_write(directory/'rest_manifest.json',{'subjects':subjects,'path':str((folder/'rest.npy').relative_to(root)),
                'sha256':sha256(folder/'rest.npy'),'source_sha256':sha256(root_project/'src/neuroedge/frontier_alignment.py')})
            json_write(directory/'audit.json',{'requested_subjects':subjects,'included_subjects':subjects,'exclusions':[]})
        def fake_prepare(unused_plan,subjects,role):
            return load_cohort(plan,role)
        def fake_rest(unused_plan,cohort,role):
            return load_rest(plan,cohort,role)
        with patch('neuroedge.frontier.candidates',return_value=specs), \
             patch('neuroedge.frontier_protocol.candidates',return_value=specs), \
             patch('neuroedge.frontier_verify.candidates',return_value=specs), \
             patch('neuroedge.frontier_report.candidates',return_value=specs), \
             patch('neuroedge.frontier_ablation.candidates',return_value=specs), \
             patch('neuroedge.frontier.prepare_cohort',side_effect=fake_prepare), \
             patch('neuroedge.frontier.prepare_rest',side_effect=fake_rest):
            _,identity=freeze(plan)
            engine._worker_init(plan)
            records=[engine._screen_candidate(spec) for spec in specs]
            assert all(r['status']=='complete' for r in records),records
            pd.DataFrame([{k:v for k,v in r.items() if k not in {'spec','folds'}} for r in records]).to_csv(out/'screening_summary.csv',index=False)
            json_write(out/'screening_complete.json',{'tested':len(specs),'failures':[],
                                                     'protocol_sha256':identity['sha256']})
            engine.select(plan)
            engine.confirm(plan)
            result=verify(plan)
            assert result['status']=='PASS'
            report(plan)
            assert (out/'results.html').exists()
            assert (out/'rest_alignment_development_ablation.csv').exists()
            assert (out/'confirmation/paired_differences.csv').exists()
            # Corrupting a completed prediction must fail independently.
            predictions=out/'confirmation/predictions.csv'
            predictions.write_bytes(predictions.read_bytes()+b'\n')
            try:verify(plan)
            except AssertionError as error:assert 'predictions changed' in str(error)
            else:raise AssertionError('Tampered final predictions were accepted.')
        print('SYNTHETIC END-TO-END SOFTWARE FIXTURE: PASS (no research results produced).')


if __name__=='__main__':main()
