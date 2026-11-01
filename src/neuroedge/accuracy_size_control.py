"""Prespecified same-holdout CSP control: 20 versus 70 source participants."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .accuracy_v2 import get_data, classical, subject_scores
from .config import json_write, sha256
from .data import select_reference
from .report import bootstrap_subjects


def freeze_control(plan):
    root=Path(plan["_root"])/plan["output_dir"]
    path=root/"source_size_control_plan.json"
    if path.exists():
        stored=json.loads(path.read_text())
        if stored["source_sha256"]!=sha256(__file__):
            raise RuntimeError("Frozen source-size control code changed.")
        return stored
    if (root/"confirmation_started.json").exists():
        raise RuntimeError("Cannot retrospectively prespecify a control after confirmation started.")
    if len(plan["development_subjects"])<=20:
        raise ValueError("This control is for the expanded cohort.")
    spec={"smaller_source_subjects":plan["development_subjects"][:20],
          "larger_source_subjects":plan["development_subjects"],
          "confirmation_subjects":plan["confirmation_subjects"],"channels":plan["channels"],
          "method":"same fixed CSP(4 components, shrinkage .1) + shrinkage LDA",
          "source_sha256":sha256(__file__),
          "selection_sha256":sha256(root/"selection.json"),
          "note":"One nested source subset; one holdout. This does not establish a universal data-size effect."}
    json_write(path,spec)
    return spec


def run_control(plan):
    root=Path(plan["_root"])/plan["output_dir"]
    spec=freeze_control(plan)
    out=root/"source_size_control"
    if (out/"complete.json").exists():
        return
    out.mkdir(exist_ok=True)
    X,meta,names=get_data(plan,plan["development_subjects"],"development_data")
    test_X,test_meta,test_names=get_data(plan,plan["confirmation_subjects"],"confirmation_data")
    source=meta.subject.isin(spec["smaller_source_subjects"]).to_numpy()
    train=select_reference(X[source],names,plan["channels"])
    test=select_reference(test_X,test_names,plan["channels"])
    with threadpool_limits(limits=plan["threads"]):
        estimator=classical("csp_lda",plan).fit(train.astype(np.float64),meta.loc[source,"label"].to_numpy(np.int64))
        p=estimator.predict_proba(test.astype(np.float64))
    joblib.dump(estimator,out/"model.joblib")
    predictions=test_meta[["trial_id","subject","run","onset","label"]].copy()
    predictions["prediction"],predictions["prob_right"]=p.argmax(1),p[:,1]
    predictions.to_csv(out/"predictions.csv",index=False)
    small=subject_scores(test_meta,p)
    small.to_csv(out/"metrics_by_subject.csv",index=False)
    full=pd.read_csv(root/"confirmation/metrics_by_subject.csv")
    full=full[full.model=="csp_lda"].set_index("subject")
    small=small.set_index("subject")
    delta=full.balanced_accuracy-small.balanced_accuracy
    lo,hi=bootstrap_subjects(delta)
    result={"smaller_sources":len(spec["smaller_source_subjects"]),"larger_sources":len(spec["larger_source_subjects"]),
            "confirmation_participants":len(delta),"small_balanced_accuracy":float(small.balanced_accuracy.mean()),
            "large_balanced_accuracy":float(full.balanced_accuracy.mean()),
            "paired_difference":float(delta.mean()),"difference_ci_low":lo,"difference_ci_high":hi,
            "method_unchanged":True,"same_confirmation_participants":True,
            "note":"Adding one fixed set of people changes both dataset size and source-population composition."}
    json_write(out/"summary.json",result)
    json_write(out/"complete.json",{"source_sha256":spec["source_sha256"],
               "model_sha256":sha256(out/"model.joblib"),"predictions_sha256":sha256(out/"predictions.csv")})
    path=root/"REPORT.md"
    base=path.read_text(encoding="utf-8").split("\n## Prespecified source-size comparison\n")[0]
    text=(f"\n## Prespecified source-size comparison\n\n"
          f"The same CSP+LDA pipeline trained on {result['smaller_sources']} source participants scored "
          f"{result['small_balanced_accuracy']*100:.2f}% balanced accuracy, versus "
          f"{result['large_balanced_accuracy']*100:.2f}% with {result['larger_sources']} source participants "
          f"on the same {len(delta)} confirmation participants. The paired difference was "
          f"{result['paired_difference']*100:+.2f} percentage points, descriptive 95% participant interval "
          f"[{lo*100:+.2f}, {hi*100:+.2f}]. This supplementary comparison was frozen before confirmation began. "
          "One fixed nested source cohort cannot establish a universal monotonic benefit of more data. "
          "See source_size_control for per-participant results and the original fitted control model.\n")
    path.write_text(base+text,encoding="utf-8")
    print("Source-size control",result,flush=True)


def verify_control(plan):
    root=Path(plan["_root"])/plan["output_dir"]
    spec=freeze_control(plan)
    out=root/"source_size_control"
    X,meta,names=get_data(plan,plan["confirmation_subjects"],"confirmation_data")
    test=select_reference(X,names,plan["channels"])
    estimator=joblib.load(out/"model.joblib")
    with threadpool_limits(limits=plan["threads"]):
        p=estimator.predict_proba(test.astype(np.float64))
    saved=pd.read_csv(out/"predictions.csv").set_index("trial_id").loc[meta.trial_id]
    np.testing.assert_allclose(saved.prob_right,p[:,1],atol=1e-6,rtol=1e-5)
    np.testing.assert_array_equal(saved.prediction,p.argmax(1))
    if set(spec["smaller_source_subjects"]) & set(spec["confirmation_subjects"]):
        raise AssertionError("Source-size control participant leakage.")
    result={"status":"PASS","saved_control_predictions_reproduced":True,
            "same_test_cohort_for_both_source_sizes":True,"source_code_unchanged":True}
    json_write(out/"verification.json",result)
    return result
