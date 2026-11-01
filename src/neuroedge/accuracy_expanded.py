"""Reuse the frozen accuracy engine with an expanded participant cohort.

The model-fitting algorithm is unchanged. This wrapper makes cohort-specific
report wording reflect the expanded plan rather than the v2 pilot's fixed IDs.
"""
import argparse
from pathlib import Path
from .accuracy_v2 import load_plan, develop, confirm, verify


def correct_report(plan):
    path = Path(plan["_root"])/plan["output_dir"]/"REPORT.md"
    text = path.read_text(encoding="utf-8")
    old = ("The v0.1 20-participant cohort is now development data. All candidate choices used subjects 1–16 for fitting and 17–20 for validation. "
           "Subjects 21–30 were reserved for confirmation and acquired/scored only after selection was frozen. "
           "This is a new participant holdout, not a rerun of the v0.1 five-fold benchmark.")
    new = (f"This expanded-cohort study follows the retained small-cohort v2 confirmation, which did not show a reliable benefit of longer training. "
           f"Development training participants: {plan['train_subjects']}. Validation participants: {plan['validation_subjects']}. "
           f"Confirmation participants: {plan['confirmation_subjects']}. "
           "The expanded plan was frozen before acquiring or scoring its new confirmation cohort. "
           "This is a sequential exploratory study; previous negative results remain part of the evidence. "
           "Its different cohorts do not permit a causal attribution to dataset size alone.")
    text = text.replace(old,new)
    text = text.replace("# NeuroEdge v0.2 — prospective confirmation", "# NeuroEdge — expanded-cohort confirmation")
    text = text.replace("All final models refit using the 20 development participants.",
                        f"All final models refit using the {len(plan['development_subjects'])} development participants.")
    text = text.replace("configs/accuracy_v2.yaml", "configs/accuracy_expanded.yaml")
    text = text.replace("python -m neuroedge.accuracy_v2 develop", "python -m neuroedge.accuracy_expanded develop")
    path.write_text(text,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["develop","confirm","verify"])
    parser.add_argument("--plan",default="configs/accuracy_expanded.yaml")
    args=parser.parse_args()
    plan=load_plan(args.plan)
    if args.command=="confirm":
        from .accuracy_size_control import freeze_control
        freeze_control(plan)
    {"develop":develop,"confirm":confirm,"verify":verify}[args.command](plan)
    if args.command=="confirm":
        correct_report(plan)
        from .accuracy_size_control import run_control
        run_control(plan)
    if args.command=="verify":
        from .accuracy_size_control import verify_control
        verify_control(plan)


if __name__=="__main__":
    main()
