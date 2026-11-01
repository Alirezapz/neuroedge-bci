"""Source-only negative controls: preceding rest alone and cue-position metadata."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from neuroedge.config import json_write,sha256
from neuroedge.context_protocol import load_plan,freeze,splits,predictions,score
from neuroedge.context_data import load_cohort


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',default='configs/context_fusion.yaml')
    args=parser.parse_args();plan=load_plan(args.plan);out,identity=freeze(plan);cohort=load_cohort(plan,'development_data')
    baseline=np.log(np.maximum(np.diagonal(np.asarray(cohort.baseline_cov[:,2:],dtype=float),axis1=-2,axis2=-1),1e-8)).reshape(len(cohort.meta),-1)
    ordinal=cohort.meta.trial_id.str.extract(r'_T(\d+)$')[0].astype(int).to_numpy();records=[];features=[]
    with threadpool_limits(limits=1):
        for split in splits(plan):
            train=np.flatnonzero(cohort.meta.subject.isin(split['train_subjects']));held=np.flatnonzero(cohort.meta.subject.isin(split['validation_subjects']))
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=7)).fit(baseline[train],cohort.meta.iloc[train].label)
            p=model.predict_proba(baseline[held]);frame=predictions(cohort.meta.iloc[held].reset_index(drop=True),p);frame['fold']=split['fold'];features.append(frame)
            labels=cohort.meta.iloc[train].label.to_numpy();run=cohort.meta.iloc[train].run.to_numpy();right=[]
            for index in held:
                mask=(run==cohort.meta.iloc[index].run)&(ordinal[train]==ordinal[index])
                right.append(float(labels[mask].mean()) if mask.any() else float(labels.mean()))
            right=np.asarray(right);frame=predictions(cohort.meta.iloc[held].reset_index(drop=True),np.column_stack([1-right,right]));frame['fold']=split['fold'];records.append(frame)
    result={'protocol_sha256':identity['sha256'],'script_sha256':sha256(__file__),
            'scope':'Already-inspected source people only; negative controls, not new confirmation or a deployed decoder.'}
    for name,frames in [('preceding_rest_only',features),('run_and_cue_ordinal_only',records)]:
        frame=pd.concat(frames,ignore_index=True);frame.to_csv(out/'diagnostics'/(name+'.csv'),index=False)
        result[name]=score(frame,np.column_stack([1-frame.prob_right,frame.prob_right]))
    json_write(out/'diagnostics/baseline_only_controls.json',result);print(result,flush=True)


if __name__=='__main__':main()
