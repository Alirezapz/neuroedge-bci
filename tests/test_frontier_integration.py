"""Isolated process checks the complete research-engine software workflow."""
from pathlib import Path
import os
import subprocess
import sys


def test_full_workflow_and_tampering_detection_with_synthetic_fixture():
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy()
    env['PYTHONPATH']=str(root/'src')
    env['OPENBLAS_NUM_THREADS']='2'
    env['OMP_NUM_THREADS']='2'
    completed=subprocess.run([sys.executable,str(root/'tests/frontier_integration_probe.py')],
        cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    assert completed.returncode==0,completed.stdout[-14000:]
    assert 'SYNTHETIC END-TO-END SOFTWARE FIXTURE: PASS' in completed.stdout
