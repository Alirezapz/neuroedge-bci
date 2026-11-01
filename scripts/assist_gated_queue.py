"""Optional execution-only tail assistance for the already-frozen gate study."""
import json
import os
from pathlib import Path
import time
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')

from neuroedge.config import json_write,sha256
from neuroedge.context_protocol import now
from neuroedge.gated_study import load_plan,freeze
from run_context_cpu import call,dispatch

plan=load_plan('configs/gated_context.yaml');out,identity=freeze(plan);root=Path(plan['_root'])
json_write(out/'scheduling/auxiliary_plan.json',{
    'registered_utc':now(),'scope':'CPU scheduling only, after scientific methods and source selection were already frozen.',
    'protocol_sha256':identity['sha256'],'worker_sha256':sha256(Path(__file__).with_name('gated_fold_worker.py')),
    'subjects':[1008,1009],'maximum_extra_processes':2,
    'trigger':'After the distant context-fold9 job finishes, use spare memory for temporary complete gate folds; atomic no-replacement publication protects the main writer.',
    'method_parameters_epochs_seeds_and_people_unchanged':True})
while not (root/plan['_parent']['output_dir']/'external_loso/folds/S1009/complete.json').exists():
    log=root.parent.parent/'context_cpu_all.log'
    # The normal main driver is authoritative; auxiliary work is never needed
    # to turn an errored primary study into a successful result.
    if log.exists() and 'Traceback' in log.read_text():raise RuntimeError('Primary driver needs attention; do not start auxiliary work.')
    time.sleep(10)
tasks=[]
for person in [1009,1008]:
    if not (out/'external_loso/folds'/f'S{person:04d}').exists():
        tasks.append((f'gate auxiliary {person}',call,([str(Path(__file__).with_name('gated_fold_worker.py')),'--subject',str(person)],out/f'auxiliary_{person}.log')))
while tasks:
    memory=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    anon=int(memory['anon'])+int(memory['kernel'])
    if anon<4.8*2**30:break
    if all((out/'external_loso/folds'/f'S{person:04d}').exists() for person in [1008,1009]):tasks=[];break
    time.sleep(10)
if tasks:dispatch(tasks,2)
print('Optional gated-queue assistance finished.',flush=True)
