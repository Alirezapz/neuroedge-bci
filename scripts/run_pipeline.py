"""Run all real-data stages sequentially; stop immediately on a failed stage."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/main.yaml")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for command in ["prepare", "run", "deploy", "report", "demo", "verify"]:
    print(f"\nNeuroEdge stage: {command}", flush=True)
    subprocess.run([sys.executable, "-m", "neuroedge", command, "--config", args.config], cwd=root, check=True)

