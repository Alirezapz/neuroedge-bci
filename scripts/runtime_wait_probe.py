"""Synthetic execution probe only: never EEG accuracy or a device benchmark.

Run in separate Python processes with OMP_WAIT_POLICY unset, ACTIVE, and
PASSIVE, retaining four requested threads, and compare final state hashes.
"""
import time
import hashlib
import json
import os
import torch
from neuroedge.models import EEGNet

torch.set_num_threads(4);torch.manual_seed(7);torch.use_deterministic_algorithms(True)
a=torch.randn(64,1,16,448);y=torch.arange(64)%2;m=EEGNet(16,448,dropout=.5)
o=torch.optim.AdamW(m.parameters(),lr=.001)
for i in range(65):
    if i==5:start=time.monotonic()
    o.zero_grad(set_to_none=True);z=m(a);loss=torch.nn.functional.cross_entropy(z,y)
    loss.backward();o.step();m.constrain()
h=hashlib.sha256()
for k,v in m.state_dict().items():h.update(v.numpy().tobytes())
print(json.dumps({'scope':'synthetic software-runtime probe, never research accuracy or device timing',
    'policy':os.environ.get('OMP_WAIT_POLICY'),'seconds_for_60_batches':time.monotonic()-start,'state_sha256':h.hexdigest()}))
