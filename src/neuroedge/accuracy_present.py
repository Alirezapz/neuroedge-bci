"""Present frozen v2 results; no fitting, selection, or new test evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .accuracy_v2 import load_plan, get_data
from .data import select_reference, CHANNELS_8
from .demo import TEMPLATE
from .report import bootstrap_subjects


def present(plan):
    out = Path(plan["_root"]) / plan["output_dir"]
    verification = json.loads((out / "verification.json").read_text())
    if verification["status"] != "PASS":
        raise RuntimeError("Verify the confirmation artifacts before presenting them.")
    chosen = json.loads((out / "selection.json").read_text())["selected"]
    per_subject = pd.read_csv(out / "confirmation/metrics_by_subject.csv")
    pivot = per_subject.pivot(index="subject", columns="model", values="balanced_accuracy")
    paired = []
    for baseline in ["csp_lda", "eegnet_old_settings_25epochs"]:
        if baseline == chosen["name"]:
            continue
        delta = pivot[chosen["name"]] - pivot[baseline]
        lo, hi = bootstrap_subjects(delta)
        paired.append({"selected":chosen["name"], "baseline":baseline, "subjects":len(delta),
                       "mean_difference":float(delta.mean()), "ci_low":lo, "ci_high":hi})
    pd.DataFrame(paired).to_csv(out / "confirmation/paired_differences.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.4), layout="constrained")
    for spec in plan["neural_candidates"]:
        directory = out / "development" / spec["name"]
        history = pd.read_csv(directory / "history.csv")
        result = json.loads((directory / "result.json").read_text())
        axes[0].plot(history.epoch,history.train_balanced_accuracy*100,label=spec["name"])
        axes[1].plot(history.epoch,history.validation_balanced_accuracy*100,label=spec["name"])
        best = history[history.epoch==result["best_epoch"]].iloc[0]
        axes[1].scatter([best.epoch],[best.validation_balanced_accuracy*100],s=25)
    descriptions = [f"Training participants · n={len(plan['train_subjects'])}",
                    f"Validation participants · n={len(plan['validation_subjects'])}"]
    for ax, title in zip(axes,descriptions):
        ax.set(xlabel="Epoch",ylabel="Balanced accuracy (%)",title=title)
        ax.axhline(50,color="#94a3b8",linestyle="--",linewidth=1)
        ax.grid(alpha=.15)
    axes[1].legend(fontsize=7)
    fig.suptitle("Development curves only · markers select minimum validation log loss")
    fig.savefig(out / "development_learning_curves.png",dpi=180)
    plt.close(fig)
    control_file=out/"source_size_control/metrics_by_subject.csv"
    if control_file.exists():
        small=pd.read_csv(control_file).set_index("subject")
        full=per_subject[per_subject.model=="csp_lda"].set_index("subject")
        joint=small[["balanced_accuracy"]].join(full[["balanced_accuracy"]],lsuffix="_20",rsuffix="_70")
        fig,ax=plt.subplots(figsize=(6,4.4),layout="constrained")
        for row in joint.itertuples():
            ax.plot([20,70],[row.balanced_accuracy_20*100,row.balanced_accuracy_70*100],color="#94a3b8",alpha=.55,marker="o",linewidth=1)
        ax.plot([20,70],[joint.balanced_accuracy_20.mean()*100,joint.balanced_accuracy_70.mean()*100],
                color="#2563eb",linewidth=3,marker="o",label="Participant mean")
        ax.set(xticks=[20,70],xlabel="Source participants used to train the same CSP+LDA pipeline",
               ylabel="Held-out balanced accuracy (%)",title="Same ten confirmation participants\nEach grey line is one person")
        ax.legend(frameon=False)
        ax.grid(axis="y",alpha=.15)
        fig.savefig(out/"source_size_control/paired_accuracy.png",dpi=180)
        plt.close(fig)
    X, meta, names = get_data(plan,plan["confirmation_subjects"],"confirmation_data")
    signals = select_reference(X,names,plan["channels"])
    predictions = pd.read_csv(out / "confirmation/predictions.csv")
    frame = predictions[predictions.model==chosen["name"]].sort_values(["subject","run","onset"])
    first_subjects = sorted(frame.subject.unique())[:4]
    frame = frame[frame.subject.isin(first_subjects)].groupby("subject",sort=True).head(3)
    indices = {r.trial_id:i for i,r in meta.iterrows()}
    trials = []
    for r in frame.itertuples():
        trials.append({"id":r.trial_id,"subject":int(r.subject),"run":int(r.run),"truth":int(r.label),
                       "prediction":int(r.prediction),"prob_right":float(r.prob_right),
                       "signals":np.round(signals[indices[r.trial_id],:8],3).tolist()})
    score = per_subject[per_subject.model==chosen["name"]].balanced_accuracy.mean()
    payload = {"trials":trials,"channels":CHANNELS_8,"sfreq":128,"duration":3,
               "balanced_accuracy":float(score),"subjects":int(per_subject.subject.nunique()),
               "seeds":[plan["seed"]],"scope":"fresh unseen-participant confirmation; 16 electrodes, 8 displayed",
               "experiment":plan["name"]}
    html = TEMPLATE.replace("__PAYLOAD__",json.dumps(payload,separators=(",",":")))
    html = html.replace("SAVED EEGNET PREDICTION", "SAVED "+chosen["name"].upper().replace("_"," ")+" PREDICTION")
    html = html.replace("Subject-mean balanced accuracy · 8 channels","Subject-mean balanced accuracy · 16 channels")
    html = html.replace('<div class="value">8 / 64</div>','<div class="value">16 / 64</div>')
    html = html.replace("Explore eight sensorimotor channels", "Explore eight displayed channels of a sixteen-electrode model")
    html = html.replace("Softmax scores are model outputs", "Class probabilities are model outputs")
    directory = out / "demo"
    directory.mkdir(exist_ok=True)
    (directory/"replay.html").write_text(html,encoding="utf-8")
    extra = ["", "## Derived comparisons and visualizations", "",
             "Paired descriptive differences below compare the selected model with baselines on the same ten participants.", "",
             "| Baseline | Mean balanced-accuracy difference | Descriptive 95% participant interval |",
             "|---|---:|---:|"]
    for r in paired:
        extra.append(f"| {r['baseline']} | {r['mean_difference']*100:+.2f} percentage points | [{r['ci_low']*100:+.2f}, {r['ci_high']*100:+.2f}] |")
    extra += ["", "![Development learning curves](development_learning_curves.png)", "",
              "[Recorded-data replay](demo/replay.html) shows the first three chronological trials of the first four confirmation participants, "
              "without selection by correctness. Sixteen channels feed the model; eight are displayed. Predictions are stored outputs.", ""]
    report = out / "REPORT.md"
    base = report.read_text(encoding="utf-8").split("\n## Derived comparisons and visualizations\n")[0]
    report.write_text(base+"\n"+"\n".join(extra),encoding="utf-8")
    print(directory/"replay.html",flush=True)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",default="configs/accuracy_v2.yaml")
    args=parser.parse_args()
    present(load_plan(args.plan))

