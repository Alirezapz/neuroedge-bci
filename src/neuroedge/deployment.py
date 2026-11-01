"""Verified FP32 export and train-only calibrated INT8 QDQ CPU measurements."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import output_path, json_write, environment, sha256
from .data import load_prepared, select_reference
from .models import load_checkpoint
from .experiment import probabilities


def calibration_indices(meta, train_subjects, n, seed):
    eligible = np.flatnonzero(meta.subject.isin(train_subjects).to_numpy())
    if not len(eligible):
        raise ValueError("No training trials available for quantization calibration.")
    return np.random.default_rng(seed).permutation(eligible)[:n]


def export_model(model, destination):
    dummy = torch.zeros(1, 1, model.spec["channels"], model.spec["samples"], dtype=torch.float32)
    torch.onnx.export(model.eval(), dummy, str(destination), dynamo=False, opset_version=17,
                      input_names=["eeg"], output_names=["logits"],
                      dynamic_axes={"eeg": {0: "batch"}, "logits": {0: "batch"}},
                      do_constant_folding=True)
    import onnx
    onnx.checker.check_model(str(destination))


def ort_session(path, threads=1):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def ort_probabilities(session, X, batch_size=64):
    chunks = []
    for start in range(0, len(X), batch_size):
        logits = session.run(None, {"eeg": np.ascontiguousarray(X[start:start + batch_size, None], dtype=np.float32)})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        chunks.append(exp / exp.sum(axis=1, keepdims=True))
    return np.concatenate(chunks)


def latency(session, X, warmup, iterations):
    # Includes Python session.run overhead; excludes acquisition and preprocessing.
    inputs = [np.ascontiguousarray(x[None, None], dtype=np.float32) for x in X[:min(32, len(X))]]
    for i in range(warmup):
        session.run(None, {"eeg": inputs[i % len(inputs)]})
    timings = []
    for i in range(iterations):
        start = time.perf_counter_ns()
        session.run(None, {"eeg": inputs[i % len(inputs)]})
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return timings


def quantize(fp32, destination, calibration_X):
    from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, X):
            self.items = iter(np.ascontiguousarray(X[:, None], dtype=np.float32))

        def get_next(self):
            try:
                return {"eeg": next(self.items)[None]}
            except StopIteration:
                return None

    preprocessed = Path(fp32).with_name("fp32_preprocessed.onnx")
    quant_pre_process(str(fp32), str(preprocessed), skip_symbolic_shape=False)
    quantize_static(str(preprocessed), str(destination), Reader(calibration_X),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=["Conv", "Gemm", "MatMul"],
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    import onnx
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    count = sum(n.op_type == "QuantizeLinear" for n in graph.graph.node)
    if count == 0:
        raise RuntimeError("Quantization produced no QuantizeLinear nodes; not an INT8 QDQ model.")
    return count


def deploy(cfg):
    out = output_path(cfg)
    progress = json.loads((out / "progress.json").read_text())
    if progress["status"] != "complete":
        raise RuntimeError("Complete all planned training before deployment evaluation.")
    X, meta, names = load_prepared(cfg)
    root = out / "deployment"
    root.mkdir(exist_ok=True)
    torch.set_num_threads(cfg["deployment"]["threads"])
    json_write(root / "environment.json", {**environment(), "settings": cfg["deployment"],
                                           "latency_scope": "batch=1 model-only incl. session.run overhead; preallocated standardized epochs"})
    seed = cfg["evaluation"]["seeds"][0]  # Predefined; never choose the best test seed.
    rows, predictions, timings = [], [], []
    for fold in range(cfg["evaluation"]["folds"]):
        for count in cfg["evaluation"]["channels"]:
            directory = root / f"fold_{fold}_channels_{count}"
            directory.mkdir(exist_ok=True)
            checkpoint = out / "experiments" / f"fold_{fold}" / f"channels_{count}" / f"eegnet_seed_{seed}" / "model.pt"
            model, scaler, cp = load_checkpoint(checkpoint)
            selected = select_reference(X, names, count)
            test_idx = np.flatnonzero(meta.subject.isin(cp["test_subjects"]).to_numpy())
            test_X = scaler.transform(selected[test_idx])
            cal_idx = calibration_indices(meta, cp["train_subjects"], cfg["deployment"]["calibration_trials"], seed)
            if set(meta.iloc[cal_idx].subject) & (set(cp["validation_subjects"]) | set(cp["test_subjects"])):
                raise RuntimeError("Quantization calibration leaked non-training participants.")
            json_write(directory / "calibration.json", {"training_subjects": cp["train_subjects"],
                       "trial_ids": meta.iloc[cal_idx].trial_id.tolist(), "checkpoint_sha256": sha256(checkpoint)})
            np.savez_compressed(directory / "preprocessing.npz", mean=scaler.mean_, scale=scaler.scale_,
                                channels=np.asarray(cp["channel_names"]))
            fp32 = directory / "fp32.onnx"
            extra = {}
            try:
                export_model(model, fp32)
                fp_session = ort_session(fp32, cfg["deployment"]["threads"])
                py_p = probabilities(model, test_X)
                fp_p = ort_probabilities(fp_session, test_X)
                max_error = float(np.abs(py_p - fp_p).max())
                agreement = float((py_p.argmax(1) == fp_p.argmax(1)).mean())
                if max_error > 1e-4 or agreement != 1.0:
                    raise RuntimeError(f"FP32 export agreement failed: max error {max_error}, agreement {agreement}.")
                extra = {"max_probability_error": max_error, "pytorch_class_agreement": agreement}
                variants = [("fp32", fp32, fp_session, fp_p, extra)]
            except Exception as error:
                rows.append({"fold": fold, "channels": count, "variant": "fp32", "status": "failed",
                             "error": f"{type(error).__name__}: {error}"})
                print(f"Export FAILED fold={fold} channels={count}: {error}", flush=True)
                continue
            int8 = directory / "int8_qdq.onnx"
            try:
                qnodes = quantize(fp32, int8, scaler.transform(selected[cal_idx]))
                q_session = ort_session(int8, cfg["deployment"]["threads"])
                q_p = ort_probabilities(q_session, test_X)
                variants.append(("int8_qdq", int8, q_session, q_p,
                                 {"quantize_nodes": qnodes, "fp32_class_agreement": float((q_p.argmax(1) == fp_p.argmax(1)).mean())}))
            except Exception as error:
                rows.append({"fold": fold, "channels": count, "variant": "int8_qdq", "status": "failed",
                             "error": f"{type(error).__name__}: {error}"})
                print(f"INT8 FAILED fold={fold} channels={count}: {error}", flush=True)
            # Alternate timing order across folds; both sessions use identical settings and inputs.
            if fold % 2:
                variants.reverse()
            for variant, artifact, session, probs, details in variants:
                ms = latency(session, test_X, cfg["deployment"]["warmup"], cfg["deployment"]["iterations"])
                rows.append({"fold": fold, "channels": count, "variant": variant, "status": "success",
                             "bytes": artifact.stat().st_size, "p50_ms": float(np.median(ms)),
                             "p95_ms": float(np.percentile(ms, 95)), "seed": seed, **details})
                frame = meta.iloc[test_idx][["trial_id", "subject", "run", "onset", "label"]].copy()
                frame["prediction"], frame["prob_right"] = probs.argmax(1), probs[:, 1]
                frame["model"], frame["seed"], frame["fold"], frame["channels"] = variant, seed, fold, count
                predictions.append(frame)
                timings.extend({"fold": fold, "channels": count, "variant": variant,
                                "iteration": i, "milliseconds": t} for i, t in enumerate(ms))
                print(f"Deployment fold={fold} channels={count} {variant}: {artifact.stat().st_size} bytes, p50={np.median(ms):.3f}ms", flush=True)
            pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
    pd.DataFrame(timings).to_csv(root / "latency_samples.csv", index=False)
    if predictions:
        all_predictions = pd.concat(predictions, ignore_index=True)
        all_predictions.to_csv(root / "predictions.csv", index=False)
        from .report import summarize_predictions
        per_seed, per_subject, summary = summarize_predictions(all_predictions)
        per_subject.to_csv(root / "metrics_by_subject.csv", index=False)
        summary.to_csv(root / "accuracy_summary.csv", index=False)
    status = {"planned": cfg["evaluation"]["folds"] * len(cfg["evaluation"]["channels"]),
              "fp32_success": sum(r["variant"] == "fp32" and r["status"] == "success" for r in rows),
              "int8_success": sum(r["variant"] == "int8_qdq" and r["status"] == "success" for r in rows),
              "seed": seed, "all_failures_reported": True}
    json_write(root / "summary.json", status)
    (root / "README.md").write_text(
        "# Deployment measurements\n\n"
        "Only the first predefined EEGNet seed is exported, for every outer fold and channel configuration. "
        "Calibration uses training participants only; calibration.json retains exact trial identifiers. "
        "FP32 numerical agreement is checked on all corresponding test trials before quantization.\n\n"
        "INT8 QDQ quantizes Conv/Gemm/MatMul operations; other operations can remain floating point. "
        "This is mixed-precision graph quantization, not a claim of an integer-only MCU implementation. "
        "Size is the actual ONNX file size, including graph overhead. A smaller parameter precision does not guarantee a smaller artifact or faster execution.\n\n"
        "latency_samples.csv contains every batch-one CPU timing after warm-up. Results include Python session.run overhead, "
        "exclude model loading, filtering/resampling, normalization, and signal acquisition, and use preallocated inputs. "
        "A 3-second input window still has to be acquired. Tests ran in a shared virtualized CPU environment. "
        "Treat timings as environment-specific, not physical device energy measurements. "
        "See environment.json for hardware, versions, threads, warm-up and iteration counts.\n\n"
        "results.csv reports failures without replacing them by estimates. accuracy_summary.csv averages performance across held-out participants.\n",
        encoding="utf-8")
    return status

