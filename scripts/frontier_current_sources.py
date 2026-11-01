"""Record reacquired EDF provenance without replacing the initial receipt."""
from pathlib import Path
import argparse
import json

from neuroedge.config import json_write,sha256
from neuroedge.frontier_protocol import load_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='configs/frontier.yaml')
    args=parser.parse_args();plan=load_plan(args.plan)
    root=Path(plan['_root']);records=[]
    for subject in plan['discovery_subjects']+plan['validation_subjects']+plan['confirmation_subjects']:
        for run in [1]+plan['dataset']['runs']:
            path=root/plan['dataset']['raw_dir']/f'S{subject:03d}/S{subject:03d}R{run:02d}.edf'
            sidecar=path.with_suffix('.source.json')
            if not path.exists() or not sidecar.exists():continue
            source=json.loads(sidecar.read_text())
            if source['sha256']!=sha256(path) or source['bytes']!=path.stat().st_size:
                raise RuntimeError('Reacquired EDF provenance differs from its sidecar.')
            records.append({'subject':subject,'run':run,'path':str(path.relative_to(root)),**source})
    json_write(root/plan['output_dir']/'acquisition_sources_recovered.json',{
        'dataset':'EEGMMIDB1.0.0','files':records,'n_files':len(records),
        'source_sha256':sha256(__file__),
        'note':'Current real EDF sources after runtime recovery; the initial acquisition receipt is retained separately. No download or model training occurs here.'})
    print('CURRENT EDF SOURCE HASHES CHECKED',len(records))


if __name__=='__main__':main()
