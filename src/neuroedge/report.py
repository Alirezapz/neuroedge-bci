"""Participant-level summaries, paired intervals, real result plots, and report."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from .config import output_path, json_write


def bootstrap_subjects(values, iterations=5000, seed=43):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_predictions(predictions):
    rows = []
    for keys, group in predictions.groupby(["model", "channels", "seed", "subject"], sort=True):
        model, channels, seed, subject = keys
        if group.trial_id.duplicated().any():
            raise ValueError("A trial was scored more than once within a model/channel/seed.")
        rows.append({"model": model, "channels": int(channels), "seed": int(seed), "subject": int(subject),
                     "n_trials": len(group),
                     "balanced_accuracy": float(balanced_accuracy_score(group.label, group.prediction)),
                     "macro_f1": float(f1_score(group.label, group.prediction, average="macro", zero_division=0))})
    per_seed = pd.DataFrame(rows)
    # Seeds are repeated measurements, not independent participants.
    per_subject = per_seed.groupby(["model", "channels", "subject"], as_index=False)[["balanced_accuracy", "macro_f1"]].mean()
    summary = []
    for (model, count), group in per_subject.groupby(["model", "channels"]):
        lo, hi = bootstrap_subjects(group.balanced_accuracy)
        summary.append({"model": model, "channels": int(count), "subjects": len(group),
                        "balanced_accuracy": float(group.balanced_accuracy.mean()), "ci_low": lo, "ci_high": hi,
                        "macro_f1": float(group.macro_f1.mean())})
    return per_seed, per_subject, pd.DataFrame(summary)


def report(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = output_path(cfg)
    p = pd.read_csv(out / "predictions.csv")
    per_seed, per_subject, summary = summarize_predictions(p)
    per_seed.to_csv(out / "metrics_by_subject_seed.csv", index=False)
    per_subject.to_csv(out / "metrics_by_subject.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired = []
    for model in per_subject.model.unique():
        pivot = per_subject[per_subject.model == model].pivot(index="subject", columns="channels", values="balanced_accuracy")
        if 64 not in pivot:
            continue
        for count in (16, 8):
            if count not in pivot:
                continue
            delta = (pivot[count] - pivot[64]).dropna()
            lo, hi = bootstrap_subjects(delta)
            paired.append({"model": model, "channels": count, "reference_channels": 64,
                           "subjects": len(delta), "mean_delta": float(delta.mean()), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(paired).to_csv(out / "paired_channel_differences.csv", index=False)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    colors = {"dummy": "#94a3b8", "csp_lda": "#0d9488", "eegnet": "#2563eb"}
    for model, group in summary.groupby("model"):
        group = group.sort_values("channels")
        mean = group.balanced_accuracy.to_numpy() * 100
        ax.errorbar(group.channels, mean,
                    yerr=np.stack([mean - group.ci_low.to_numpy() * 100, group.ci_high.to_numpy() * 100 - mean]),
                    fmt="o-", capsize=4, label=model, color=colors.get(model))
    ax.axhline(50, color="#94a3b8", linestyle="--", linewidth=1)
    ax.set(xlabel="Available EEG channels", ylabel="Subject-mean balanced accuracy (%)",
           title=f"NeuroEdge · {cfg['name']} · unseen-subject evaluation", xticks=[8, 16, 64])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.savefig(figures / "channel_accuracy.png", dpi=180)
    fig.savefig(figures / "channel_accuracy.svg")
    plt.close(fig)
    counts = sorted(p.channels.unique().tolist())
    fig, axes = plt.subplots(1, len(counts), figsize=(4 * len(counts), 3.8), layout="constrained", squeeze=False)
    for ax, count in zip(axes[0], counts):
        group = p[(p.model == "eegnet") & (p.channels == count)]
        matrices = [confusion_matrix(seed_group.label, seed_group.prediction, labels=[0, 1])
                    for _, seed_group in group.groupby("seed")]
        matrix = np.mean(matrices, axis=0)
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Left", "Right"], yticklabels=["Left", "Right"],
               xlabel="Predicted", ylabel="True", title=f"EEGNet · {count} channels")
    fig.suptitle("Outer-test confusion counts, averaged across seeds")
    fig.savefig(figures / "confusion_matrices.png", dpi=180)
    plt.close(fig)
    lines = [f"# NeuroEdge — {cfg['name']} research report", "", f"Scope: {cfg['scope']}", "",
             "## Question and design", "",
             "How do predefined channel reductions affect left/right motor-imagery decoding on unseen participants? "
             "A separate deployment study compares FP32 and training-calibrated INT8 QDQ inference. "
             "This is an independent engineering reproduction, with no claim of algorithmic novelty.", "",
             f"Cohort: {len(cfg['dataset']['subjects'])} fixed participants. Runs 4, 8, 12 only. "
             f"Outer folds: {cfg['evaluation']['folds']}; EEGNet seeds: {cfg['evaluation']['seeds']}. "
             "Inner validation participants are disjoint from both training and test. Classical and neural models use the same training participants.", "",
             "Methods: fourth-order zero-phase Butterworth 8–30 Hz, resampling to 128 Hz, "
             "one [1,4) second epoch per imagery event, configuration-specific average reference. "
             "CSP uses four components and fixed covariance shrinkage; LDA uses automatic shrinkage. "
             "EEGNet-8,2 uses a 32-sample temporal kernel, Adam, validation-loss early stopping, "
             "and per-channel scaling fitted on training trials only. See docs/METHODS.md.", "",
             "## Measured results", "",
             "| Model | Channels | Participants | Balanced accuracy | 95% participant-bootstrap interval | Macro-F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.model} | {r.channels} | {r.subjects} | {r.balanced_accuracy:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.macro_f1:.3f} |")
    best = summary.loc[summary.balanced_accuracy.idxmax()]
    neural = summary[summary.model == "eegnet"]
    lines += ["", "## Interpretation", "",
              f"The largest observed point estimate among the predefined configurations was {best.balanced_accuracy*100:.1f}% "
              f"for {best.model} with {int(best.channels)} channels. EEGNet point estimates ranged from "
              f"{neural.balanced_accuracy.min()*100:.1f}% to {neural.balanced_accuracy.max()*100:.1f}% across channel configurations. "
              "This describes the prespecified comparison; it is not an independently confirmed model-selection result. "
              "Binary balanced-accuracy chance is 50%. These results do not support a claim of a high-performing practical BCI. "
              "Investigating data sufficiency, optimization, or subject calibration would require a separately specified experiment using training/validation evidence."]
    lines += ["", "Intervals resample participants after averaging repeated seeds within each participant. "
              "They describe this fixed cohort, not all possible EEG users. Paired differences are in paired_channel_differences.csv. "
              "Overlapping intervals do not establish equivalence or non-inferiority.", "",
              "![Measured channel comparison](figures/channel_accuracy.png)", "",
              "## Deployment status", ""]
    deployment = out / "deployment" / "summary.json"
    if deployment.exists():
        d = json.loads(deployment.read_text())
        lines.append(f"Completed FP32 exports: {d['fp32_success']}; runnable INT8 QDQ models: {d['int8_success']}; "
                     f"planned fold/channel exports: {d['planned']}. See deployment/results.csv and deployment/README.md for measured costs and failures.")
    else:
        lines.append("NOT RUN at report generation. No quantization, latency, or device-efficiency claims are supported yet.")
    if deployment.exists() and (out / "deployment" / "results.csv").exists():
        costs = pd.read_csv(out / "deployment" / "results.csv")
        accuracies = pd.read_csv(out / "deployment" / "accuracy_summary.csv") if (out / "deployment" / "accuracy_summary.csv").exists() else pd.DataFrame()
        timing_file = out / "deployment" / "latency_samples.csv"
        times = pd.read_csv(timing_file) if timing_file.exists() else pd.DataFrame()
        lines += ["", "| ONNX variant | Channels | Test participants | Balanced accuracy | Mean artifact KiB | Pooled p50 ms | Pooled p95 ms |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for (variant, count), group in costs[costs.status == "success"].groupby(["variant", "channels"]):
            acc = accuracies[(accuracies.model == variant) & (accuracies.channels == count)] if len(accuracies) else pd.DataFrame()
            t = times[(times.variant == variant) & (times.channels == count)].milliseconds if len(times) else pd.Series(dtype=float)
            if len(acc) and len(t):
                r = acc.iloc[0]
                lines.append(f"| {variant} | {count} | {r.subjects} | {r.balanced_accuracy:.3f} | {group.bytes.mean()/1024:.2f} | {np.median(t):.3f} | {np.percentile(t,95):.3f} |")
        if len(accuracies) and len(times):
            fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
            for r in accuracies.itertuples():
                t = times[(times.variant == r.model) & (times.channels == r.channels)].milliseconds
                if not len(t):
                    continue
                x = float(np.median(t))
                y = r.balanced_accuracy * 100
                color = "#2563eb" if r.model == "fp32" else "#d97706"
                ax.errorbar(x, y, yerr=[[y-r.ci_low*100], [r.ci_high*100-y]],
                            fmt={8: "o", 16: "s", 64: "D"}[r.channels], color=color, capsize=3)
                ax.annotate(f"{r.channels} ch · {r.model}", (x, y), xytext=(7, 7),
                            textcoords="offset points", fontsize=8, color=color)
            ax.set_xscale("log")
            ax.margins(x=0.35, y=0.18)
            ax.set(xlabel="Model-only batch-one CPU latency (ms, median; log scale)",
                   ylabel="Subject-mean balanced accuracy (%)",
                   title="Accuracy and execution cost · first predefined training seed")
            ax.grid(alpha=0.15)
            fig.savefig(figures / "deployment_tradeoff.png", dpi=180)
            fig.savefig(figures / "deployment_tradeoff.svg")
            plt.close(fig)
            lines += ["", "![Measured accuracy and CPU latency](figures/deployment_tradeoff.png)"]
        lines += ["", "Deployment uses only the first predefined training seed; the earlier EEGNet table averages all training seeds. "
                  "Do not attribute that difference to export or quantization. Timing percentiles pool saved calls across the corresponding outer-fold models. "
                  "The main pipeline measures deployment only after its planned training completes; shared-host scheduling can still affect timing. "
                  "INT8 QDQ is mixed precision."]
    completions = []
    for filename in sorted((out / "experiments").rglob("complete.json")):
        record = json.loads(filename.read_text())
        if "epochs_run" in record:
            completions.append(record)
    if completions:
        lines += ["", "## Training budget actually used", "",
                  f"EEGNet fits: {len(completions)}; epochs executed per fit: "
                  f"{min(r['epochs_run'] for r in completions)}–{max(r['epochs_run'] for r in completions)}; "
                  f"selected checkpoint epochs: {min(r['best_epoch'] for r in completions)}–{max(r['best_epoch'] for r in completions)}. "
                  "Complete training/validation curves are retained in each experiment's history.csv. "
                  "Validation loss, not test accuracy, determined the saved checkpoint."]
    lines += ["", "## Limitations", "",
              "One public dataset and a small fixed cohort; no independent external dataset; no hyperparameter search. "
              "Small validation cohorts and limited epochs can disadvantage the neural model. "
              "Offline zero-phase filtering and cue-aligned epochs do not establish real-time or asynchronous BCI performance. "
              "This task does not establish a causal neurophysiological explanation for the classifier's decisions. "
              "Inference latency excludes electrode acquisition, the 3-second signal window, EDF reading, filtering, and resampling. "
              "Channel ablation on recorded 64-channel data does not measure physical acquisition power. "
              "CPU benchmarks do not demonstrate microcontroller deployment. No clinical or state-of-the-art claim.", "",
              "## Related work", "",
              "- Lawhern et al. (2018), EEGNet. DOI: 10.1088/1741-2552/aace8c.",
              "- Schalk (2009), EEG Motor Movement/Imagery Dataset. DOI: 10.13026/C28G6P.",
              "- Schalk et al. (2004), BCI2000. DOI: 10.1109/TBME.2004.827072.",
              "- Wang et al. (2020), EEGNet-based low-power edge computing. DOI: 10.1109/MeMeA49120.2020.9137134.",
              "- Wang et al. (2024), MI-BMInet. DOI: 10.1109/JSEN.2024.3353146.",
              "- Tuimebay et al. (2026), low-channel EEG configurations and temporal segmentation. DOI: 10.3390/a19070588.", "",
              "## Next experiment", "",
              "Freeze the present findings, then evaluate a separately specified cross-dataset or cross-session protocol. "
              "Review close prior work before proposing a novel method. Do not tune on the outer-test outcomes reported here.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    json_write(out / "report_status.json", {"source": "actual saved outer-test predictions", "rows": len(p),
                                            "participants": int(p.subject.nunique()), "synthetic_results": False})
    print(summary.to_string(index=False), flush=True)
    return summary

