"""Create a portable source/results archive without raw EDF, large caches, or venv.

Can checkpoint an ongoing run: completed results and hash-verified finished
neural seeds are included. PACKAGE_STATUS.json makes unfinished runs explicit.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
destination = Path(args.output).resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
excluded_parts = {".venv", "__pycache__", ".pytest_cache", ".git", ".ruff_cache"}
status = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs": {},
          "raw_edf_included": False, "processed_cache_included": False,
          "note": "Raw data can be reacquired; see README. Incomplete results are excluded; finished neural seeds with matching receipts may be retained for exact resume."}

seed_checks={}
def finished_seed(rel):
    for i,part in enumerate(rel.parts[:-1]):
        if part.startswith('seed_'):
            directory=root.joinpath(*rel.parts[:i+1])
            if directory not in seed_checks:
                receipt=directory/'seed_receipt.json';model=directory/'model.pt'
                seed_checks[directory]=(receipt.exists() and model.exists() and
                    hashlib.sha256(model.read_bytes()).hexdigest()==json.loads(receipt.read_text())['model_sha256'])
            return seed_checks[directory]
    return False
for progress in sorted((root / "runs").glob("*/progress.json")):
    status["runs"][progress.parent.name] = json.loads(progress.read_text())
frontier_labels=set()
gated_labels=set()
for frozen in sorted((root/"runs").glob("*/protocol.json")):
    directory=frozen.parent
    registration=json.loads(frozen.read_text());spec=registration.get("plan")
    if not isinstance(spec,dict):
        if 'binder_sha256' in registration and 'parent_protocol_sha256' in registration:
            item={'status':'overall_rule_registered','external_used_for_selection':False}
            if (directory/'selection.json').exists():item['status']='source_selection_frozen'
            if (directory/'verification.json').exists():item['status']='verified' if json.loads((directory/'verification.json').read_text())['status']=='PASS' else 'verification_failed'
            status['runs'][directory.name]=item
        continue
    if spec.get('study_type')=='pre_cue_symmetry':
        frontier_labels.add(directory.name)
        item={'status':'implementation_ready','base_candidates':len(json.loads((directory/'candidates.json').read_text())),
              'source_subjects':len(spec['source_subjects']),'reused_selection_subjects':len(spec['selection_subjects']),
              'external_dataset':spec['external']['dataset'],'external_subjects':spec['external']['subjects']}
        if (directory/'screening_progress.json').exists():
            item['screening']=json.loads((directory/'screening_progress.json').read_text());item['status']='screening_in_progress'
        if (directory/'screening_complete.json').exists():item['status']='validation_in_progress'
        if (directory/'selection.json').exists():
            item['status']='selection_frozen'
            selected=json.loads((directory/'selection.json').read_text());names=set(selected['reference_names'])
            for winner in selected['selected'].values():names.update([winner['name']] if winner['kind']=='single' else winner['combination']['members'])
            item['final_constituents_required']=len(names)
            item['final_constituents_completed']=sum((directory/'final_models'/name/'model_info.json').exists() for name in names)
            item['finished_final_neural_seeds']=len(list((directory/'final_models').glob('*/seed_*/seed_receipt.json')))
        if (directory/'final_models_frozen.json').exists():item['status']='source_models_frozen'
        if (directory/'external_transfer/complete.json').exists():item['status']='external_loso_in_progress'
        if (directory/'external_loso/complete.json').exists():item['status']='external_complete'
        if (directory/'verification.json').exists():item['status']='verified' if json.loads((directory/'verification.json').read_text())['status']=='PASS' else 'verification_failed'
        status['runs'][directory.name]=item
    elif "discovery_subjects" in spec:
        frontier_labels.add(directory.name)
        grid=directory/"candidates.json"
        item={"status":"screening_in_progress","base_candidates":len(json.loads(grid.read_text())) if grid.exists() else 0,
              "discovery_subjects":len(spec["discovery_subjects"]),
              "validation_subjects":len(spec["validation_subjects"]),
              "confirmation_requested":spec["confirmation_subjects"]}
        if (directory/"screening_progress.json").exists():
            item["screening"]=json.loads((directory/"screening_progress.json").read_text())
        if (directory/"screening_complete.json").exists():item["status"]="validation_in_progress"
        if (directory/"selection.json").exists():item["status"]="selection_frozen"
        if (directory/"confirmation/complete.json").exists():item["status"]="confirmation_complete"
        if (directory/"verification.json").exists():
            item["status"]="verified" if json.loads((directory/"verification.json").read_text())["status"]=="PASS" else "verification_failed"
        status["runs"][directory.name]=item
    elif 'parent_plan' in spec and 'variants' in spec and 'gate_hidden_units' in spec:
        gated_labels.add(directory.name)
        item={'status':'source_development','base_candidates':len(spec['variants']),
              'source_variants_completed':len(list((directory/'candidates').glob('*/source_complete.json')))}
        if (directory/'selection.json').exists():item['status']='selection_frozen'
        if (directory/'external_transfer/complete.json').exists():item['status']='external_loso_in_progress'
        if (directory/'external_loso/complete.json').exists():item['status']='external_complete'
        if (directory/'verification.json').exists():item['status']='verified' if json.loads((directory/'verification.json').read_text())['status']=='PASS' else 'verification_failed'
        status['runs'][directory.name]=item
    elif 'parent_output' in spec and 'support_budgets' in spec:
        item={'status':'protocol_registered','parent_output':spec['parent_output'],
              'primary_budget':spec['primary_budget'],'support_budgets':spec['support_budgets']}
        if (directory/'parent_selection.json').exists():item['status']='source_selection_bound'
        if (directory/'complete.json').exists():item['status']='complete'
        if (directory/'verification.json').exists():item['status']='verified' if json.loads((directory/'verification.json').read_text())['status']=='PASS' else 'verification_failed'
        status['runs'][directory.name]=item
    elif "development_subjects" in spec:
        item={"status":"development_in_progress","development_subjects":len(spec["development_subjects"]),
              "confirmation_subjects":spec["confirmation_subjects"]}
        if (directory/"selection.json").exists():item["status"]="selection_frozen"
        if (directory/"confirmation/complete.json").exists():item["status"]="complete"
        status["runs"][directory.name]=item
paths = []
invalidated_runs={p.parent.name for p in (root/'runs').glob('*/INVALIDATED.json')}
for name in invalidated_runs:
    status['runs'][name]={'status':'invalidated_diagnostic','scientific_results':False,
        'note':'See INVALIDATED.json and implementation_snapshot. Incomplete model output is omitted.'}
for specification in sorted((root/'runs').glob('*/specification.json')):
    directory=specification.parent;definition=json.loads(specification.read_text());plan=definition.get('plan',{})
    if 'kernels' not in plan or 'source_selection_budget' not in plan:continue
    item={'status':'mathematical_specification_registered','source_calibration_choices':len(plan['kernels'])*len(plan['C_grid']),
          'primary_kernel':plan['primary_kernel'],'primary_budget':plan['primary_budget']}
    if (directory/'implementation.json').exists():item['status']='implementation_frozen'
    if (directory/'source_selection.json').exists():item['status']='source_selection_and_scalers_frozen'
    if (directory/'complete.json').exists():item['status']='external_complete'
    if (directory/'verification.json').exists():item['status']='verified' if json.loads((directory/'verification.json').read_text())['status']=='PASS' else 'verification_failed'
    status['runs'][directory.name]=item
for path in sorted(root.rglob("*")):
    rel = path.relative_to(root)
    if rel == Path("PACKAGE_STATUS.json"):
        continue
    if not path.is_file() or any(p in excluded_parts or p.endswith(".egg-info") for p in rel.parts):
        continue
    if rel.parts[0] == "data" or path.suffix in {".log", ".tmp", ".part", ".zip", ".pyc"}:
        continue
    if len(rel.parts)>2 and rel.parts[0]=='runs' and rel.parts[1] in invalidated_runs and rel.parts[2]=='models':
        continue
    if len(rel.parts)>3 and rel.parts[0]=='runs' and rel.parts[1] in gated_labels:
        model_directory=None
        if rel.parts[2]=='final_models':model_directory=root.joinpath(*rel.parts[:4])
        elif 'source_folds' in rel.parts:
            index=rel.parts.index('source_folds');model_directory=root.joinpath(*rel.parts[:index+2])
        elif 'validation_models' in rel.parts:
            index=rel.parts.index('validation_models');model_directory=root.joinpath(*rel.parts[:index+1])
        elif len(rel.parts)>5 and rel.parts[2]=='external_loso' and rel.parts[3]=='folds' and rel.parts[5]=='model':
            model_directory=root.joinpath(*rel.parts[:6])
        if model_directory is not None and not (model_directory/'model_info.json').exists() and not finished_seed(rel):continue
        if rel.parts[2:4]==('external_loso','folds') and len(rel.parts)>5:
            fold_directory=root.joinpath(*rel.parts[:5])
            if rel.parts[5]!='model' and not (fold_directory/'complete.json').exists():continue
    # Save finished constituent fits during a long LOSO run, but never copy an
    # in-flight PyTorch write or treat incomplete fold predictions as evidence.
    if len(rel.parts)>5 and rel.parts[0]=='runs' and rel.parts[1] in frontier_labels and rel.parts[2:4]==('external_loso','folds'):
        fold_directory=root.joinpath(*rel.parts[:5])
        if rel.parts[5]=='models' and len(rel.parts)>7:
            if not (root.joinpath(*rel.parts[:7])/'model_info.json').exists() and not finished_seed(rel):continue
        elif not (fold_directory/'complete.json').exists():continue
    # Validation constituent models are reproducible temporary fits. Preserve
    # every validation prediction/spec and all stacking meta-classifiers; the
    # actual selected final constituents and baseline models remain packaged.
    if len(rel.parts)>3 and rel.parts[0]=="runs" and rel.parts[1] in frontier_labels and rel.parts[2]=="validation":
        if (root.joinpath(*rel.parts[:4])/'complete.json').exists() and path.name in {"model.joblib","model.pt"}:
            continue
    if len(rel.parts)>3 and rel.parts[0]=='runs' and rel.parts[1] in frontier_labels and rel.parts[2]=='screening':
        if path.name in {'model.joblib','model.pt'}:
            continue
    if len(rel.parts)>3 and rel.parts[0]=="runs" and rel.parts[1] in frontier_labels:
        if rel.parts[2] in {"screening","validation"}:
            if not (root.joinpath(*rel.parts[:4])/"complete.json").exists() and not (rel.parts[2]=='validation' and finished_seed(rel)):
                continue
        if rel.parts[2]=="final_models":
            if not (root.joinpath(*rel.parts[:4])/"model_info.json").exists() and not finished_seed(rel):
                continue
    if "experiments" in rel.parts:
        exp_idx = rel.parts.index("experiments")
        experiment = root.joinpath(*rel.parts[:exp_idx + 4])
        if not (experiment / "complete.json").exists():
            continue
    paths.append(path)
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.writestr("neuroedge/PACKAGE_STATUS.json", json.dumps(status, indent=2))
    for path in paths:
        z.write(path, Path("neuroedge") / path.relative_to(root))
with zipfile.ZipFile(destination) as z:
    bad = z.testzip()
    if bad:
        raise RuntimeError(f"Archive CRC check failed: {bad}")
print(json.dumps({"path": str(destination), "bytes": destination.stat().st_size, "files": len(paths)+1, "status": status}, indent=2))

