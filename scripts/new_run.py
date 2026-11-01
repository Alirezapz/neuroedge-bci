"""Create a fresh run configuration without overwriting archived evidence."""
import argparse
from pathlib import Path
import re
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--name", required=True)
parser.add_argument('--parent-plan',help='For child studies, bind to this fresh parent configuration.')
parser.add_argument('--parent-output',help='Override the child prediction source output, for example runs/gated_local.')
args = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
    parser.error("Use only letters, numbers, underscores and hyphens in a run name.")
source = Path(args.config).resolve()
destination = source.parent / (args.name + ".yaml")
if destination.exists():
    raise SystemExit(f"Configuration already exists; left unchanged: {destination}")
cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
cfg["name"] = args.name
cfg["output_dir"] = "runs/" + args.name
if args.parent_plan:
    if 'parent_plan' not in cfg:parser.error('This configuration has no parent plan.')
    parent=Path(args.parent_plan).resolve()
    cfg['parent_plan']=str(parent.relative_to(source.parent.parent))
    if 'parent_output' in cfg:cfg['parent_output']=yaml.safe_load(parent.read_text(encoding='utf-8'))['output_dir']
if args.parent_output:
    if 'parent_output' not in cfg:parser.error('This configuration has no parent output.')
    cfg['parent_output']=args.parent_output
if cfg.get('study_type')=='pre_cue_symmetry':
    cfg['data']['cache_dir']='data/'+args.name
destination.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(destination)

