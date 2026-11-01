# Deployment measurements

Only the first predefined EEGNet seed is exported, for every outer fold and channel configuration. Calibration uses training participants only; calibration.json retains exact trial identifiers. FP32 numerical agreement is checked on all corresponding test trials before quantization.

INT8 QDQ quantizes Conv/Gemm/MatMul operations; other operations can remain floating point. This is mixed-precision graph quantization, not a claim of an integer-only MCU implementation. Size is the actual ONNX file size, including graph overhead. A smaller parameter precision does not guarantee a smaller artifact or faster execution.

latency_samples.csv contains every batch-one CPU timing after warm-up. Results include Python session.run overhead, exclude model loading, filtering/resampling, normalization, and signal acquisition, and use preallocated inputs. A 3-second input window still has to be acquired. Tests ran in a shared virtualized CPU environment. Treat timings as environment-specific, not physical device energy measurements. See environment.json for hardware, versions, threads, warm-up and iteration counts.

results.csv reports failures without replacing them by estimates. accuracy_summary.csv averages performance across held-out participants.
