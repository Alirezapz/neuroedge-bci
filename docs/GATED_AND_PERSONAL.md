# Matched gates and session-disjoint personalization

Two source-developed hypotheses extend the context study. A fixed learned gate
and a quality-conditioned gate have matched network capacity and both retain
task/pre-cue features. The quality model conditions its mixture on rest quality;
it did not outperform its fixed control. Source and external adverse findings
remain in `runs/gated_context/REPORT.md`.

The separate personal-calibration study fits one balanced log-odds intercept
with a zero-mean Gaussian prior of standard deviation 4. The primary support
is the first16 chronological session-T left/right labels. All session-E trials
are held out; all 4/8/16 budgets are reported. No feature or neural weight is
fine-tuned. If support has one class, keep the source model without looking
ahead or excluding the person. This is recorded-data simulation, not a live
user study. Results are in `runs/personalization` and `personalization_gated`.

## Fresh complete reproduction

Create every child configuration and register source choices before new external
acquisition. The following commands work with `python` in an activated virtual
environment; Windows can use `.\.venv\Scripts\python.exe` instead.

```bash
python scripts/new_run.py --config configs/context_fusion.yaml --name context_local
python scripts/new_run.py --config configs/gated_context.yaml --name gated_local --parent-plan configs/context_local.yaml
python scripts/new_run.py --config configs/personalization.yaml --name personal_local --parent-plan configs/context_local.yaml
python scripts/new_run.py --config configs/personalization_gated.yaml --name personal_gated_local --parent-plan configs/context_local.yaml --parent-output runs/gated_local
python scripts/new_run.py --config configs/residual_personal.yaml --name residual_local --parent-plan configs/context_local.yaml
python scripts/run_research.py --context-plan configs/context_local.yaml --gated-plan configs/gated_local.yaml --personal-plan configs/personal_local.yaml --gated-personal-plan configs/personal_gated_local.yaml --residual-plan configs/residual_local.yaml --summary-output runs/research_local --jobs 3 --external-jobs 3
```

For the four-study path without nonlinear personal kernels, use
`scripts/run_context_research.py` with the first four named plans. The CPU
dispatcher `scripts/run_context_parallel.py` supports `--jobs 3`. These scripts
parallelize independent fits; resource availability determines a suitable count.

For individual gate reproduction after proper registration/source development:

```bash
python -m neuroedge.gated_study source --variant fixed_gate --plan configs/gated_context.yaml
python -m neuroedge.gated_study source --variant quality_gate --plan configs/gated_context.yaml
python -m neuroedge.gated_study freeze-selection --plan configs/gated_context.yaml
python -m neuroedge.gated_study external --plan configs/gated_context.yaml
python -m neuroedge.gated_study verify --plan configs/gated_context.yaml
python -m neuroedge.gated_study report --plan configs/gated_context.yaml
```

Do not combine incompatible run folders or change source-frozen settings using
the external scores. `runs/research_summary/protocol.json` defines the overall
choice: the task branch inherits the context winner, and the context branch
compares source-developed candidates only. Supplementary personal kernels are
documented in `PERSONAL_KERNELS.md`; the latest larger-budget study has its own
`SESSION_ADAPTATION_METHODS.md` and does not replace the original evidence.
