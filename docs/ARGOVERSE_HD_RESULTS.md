# Adaptive Vision: Argoverse-HD five-sequence benchmark

Measured on 19 October 2026. This evaluates provided dataset annotations, using the repository evaluator. Development scores are excluded. No training, pseudo-labelling, parameter tuning, or detection/tracking/scheduling changes were made.

## Dataset and compatibility

All 3,416 images decoded successfully at 1920×1200. The subset contains 57,711 annotations: 55,412 regular targets and 2,299 crowds. No explicit ignore=true annotations and no entirely empty annotation frames were present. Empty/crowd-only frames are preserved by the adapter and covered by synthetic tests.

All eight classes map explicitly from stock COCO YOLO indices. Unrelated COCO predictions use the existing class-map filter. Provided pixel xywh boxes are converted to xyxy without clipping or rounding. Eight boxes extend one pixel past the right image edge; all are retained and listed in the audit.

The JSON retains 24 sequence metadata entries, but images and manifest agree on five downloaded sequences. Frame indices are contiguous and source nanosecond timestamps strictly increasing, with median intervals near 33,300,000 ns (30.030 FPS). Every sequence starts with a pipeline reset and FULL detection.

| Sequence | Original log ID | Frames | Regular targets | Crowds | FPS from timestamps |
|---|---|---:|---:|---:|---:|
| sequence_00 | 00c561b9-2057-358d-82c6-5b06d76cebcf | 901 | 9038 | 30 | 30.03001 |
| sequence_01 | 033669d3-3d6b-3d3d-bd93-7985d86653ea | 901 | 9508 | 1209 | 30.03002 |
| sequence_02 | 1d676737-4110-3f7e-bec0-0c90f74c248f | 469 | 15961 | 596 | 30.03004 |
| sequence_03 | 2d12da1d-5238-3870-bfbc-b281d5e8c1a1 | 475 | 8852 | 328 | 30.03001 |
| sequence_04 | 33737504-3373-3373-3373-633738571776 | 670 | 12053 | 136 | 30.03003 |

| Class | Dataset ID | YOLO COCO index | Annotations | Crowds | Regular targets |
|---|---:|---:|---:|---:|---:|
| person | 0 | 0 | 7177 | 84 | 7093 |
| bicycle | 1 | 1 | 687 | 0 | 687 |
| car | 2 | 2 | 30568 | 1589 | 28979 |
| motorcycle | 3 | 3 | 421 | 89 | 332 |
| bus | 4 | 5 | 4234 | 0 | 4234 |
| truck | 5 | 7 | 3668 | 1 | 3667 |
| traffic_light | 6 | 9 | 10632 | 535 | 10097 |
| stop_sign | 7 | 11 | 324 | 1 | 323 |

## Aggregate accuracy and efficiency

All accuracy entries are percentages. P/R and macro F1 are at each method’s best smoothed mean-F1 confidence. AP is pooled across all frames, then averaged over classes; it is not an average of per-sequence AP.

| Method | Precision | Recall | Macro F1 | mAP50 | mAP50–95 | FULL calls | SKIP frames | Detector-call savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 45.25 | 24.31 | 28.92 | 23.57 | 15.50 | 3416 | 0 | 0.00 |
| fixed2_093 | 50.77 | 21.10 | 26.97 | 20.55 | 13.77 | 1710 | 1706 | 49.94 |
| C_fullfallback | 50.86 | 21.02 | 26.94 | 20.43 | 13.74 | 1733 | 1683 | 49.27 |
| adaptive_observed | 50.57 | 21.03 | 26.91 | 20.43 | 13.92 | 1729 | 1687 | 49.39 |
| adaptive_quality | 50.17 | 21.12 | 26.92 | 20.42 | 13.96 | 1789 | 1627 | 47.63 |

No measured ROI passes/crops occurred. FULL and SKIP counts sum to 3,416 per method. Warmup is excluded from counts and timings: each method additionally uses three full passes and one batch of two crops in the inherited wrapper.

| Method | P at ~0.25 | R at ~0.25 | Macro F1 at ~0.25 | Best-F1 confidence | Ignored predictions at IoU50 |
|---|---:|---:|---:|---:|---:|
| baseline | 62.49 | 18.09 | 25.02 | 0.109109 | 8461 |
| fixed2_093 | 64.75 | 17.52 | 24.31 | 0.108108 | 4479 |
| C_fullfallback | 60.82 | 17.63 | 24.50 | 0.111111 | 4609 |
| adaptive_observed | 61.73 | 17.68 | 24.56 | 0.110110 | 4576 |
| adaptive_quality | 61.92 | 17.54 | 24.34 | 0.108108 | 4812 |

## Processing performance

FPS is pipeline processing throughput, despite the historical `fps_end_to_end` field name. It excludes JPEG decoding, annotation scoring, model loading, warmup, hashing and output I/O. All latency values below are milliseconds. These are single sequential runs; method order, temperature and background load can affect timings.

| Method | FPS | Mean | P50 | P95 | P99 | Max | Estimated detector GFLOPs/frame | Estimated GFLOPs savings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 45.21 | 22.12 | 21.98 | 25.38 | 33.95 | 78.18 | 5.66 | 0.00 |
| fixed2_093 | 70.63 | 14.16 | 21.51 | 25.41 | 29.95 | 92.72 | 2.83 | 49.94 |
| C_fullfallback | 66.62 | 15.01 | 23.51 | 27.81 | 28.92 | 61.44 | 2.87 | 49.27 |
| adaptive_observed | 66.24 | 15.10 | 22.83 | 27.76 | 28.47 | 81.51 | 2.86 | 49.39 |
| adaptive_quality | 67.25 | 14.87 | 23.49 | 25.32 | 26.82 | 72.33 | 2.96 | 47.63 |

GFLOPs use the inherited padded-input detector cost estimate (8.7 GFLOPs at 640×640 scaled by padded input area). This excludes tracker/scheduler arithmetic and is not measured hardware work, power or energy. The identical detector-call/GFLOPs savings here follow from no measured ROI work and uniform image dimensions.

## Per-sequence measurements

P/R/F1 confidence is selected independently for each sequence. FPS and latency retain the pipeline-only scope. Full precision machine-readable values are in each method’s `metrics.json` and `per_sequence.csv`.

| Sequence | Method | P | R | Macro F1 | AP50 | AP50–95 | FULL | SKIP | Savings % | FPS | Mean ms | P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sequence_00 | baseline | 21.81 | 12.92 | 14.33 | 11.71 | 7.26 | 901 | 0 | 0.00 | 47.20 | 21.19 | 24.76 |
| sequence_00 | fixed2_093 | 42.55 | 10.79 | 14.12 | 10.37 | 6.33 | 451 | 450 | 49.94 | 70.40 | 14.20 | 26.44 |
| sequence_00 | C_fullfallback | 23.96 | 11.41 | 13.77 | 10.27 | 6.33 | 466 | 435 | 48.28 | 67.74 | 14.76 | 25.34 |
| sequence_00 | adaptive_observed | 24.21 | 11.38 | 13.78 | 10.26 | 6.50 | 466 | 435 | 48.28 | 61.47 | 16.27 | 28.05 |
| sequence_00 | adaptive_quality | 24.15 | 11.40 | 13.76 | 10.29 | 6.60 | 486 | 415 | 46.06 | 66.14 | 15.12 | 25.51 |
| sequence_01 | baseline | 48.98 | 17.93 | 21.23 | 17.99 | 9.55 | 901 | 0 | 0.00 | 48.03 | 20.82 | 25.43 |
| sequence_01 | fixed2_093 | 54.55 | 15.57 | 20.29 | 15.30 | 7.96 | 451 | 450 | 49.94 | 73.76 | 13.56 | 23.60 |
| sequence_01 | C_fullfallback | 52.77 | 15.36 | 19.92 | 15.13 | 7.81 | 438 | 463 | 51.39 | 70.62 | 14.16 | 25.15 |
| sequence_01 | adaptive_observed | 52.60 | 15.27 | 19.79 | 15.01 | 8.11 | 435 | 466 | 51.72 | 65.01 | 15.38 | 28.02 |
| sequence_01 | adaptive_quality | 52.60 | 15.36 | 19.83 | 15.25 | 8.33 | 468 | 433 | 48.06 | 66.99 | 14.93 | 25.88 |
| sequence_02 | baseline | 44.69 | 20.08 | 25.82 | 21.29 | 13.82 | 469 | 0 | 0.00 | 42.28 | 23.65 | 27.03 |
| sequence_02 | fixed2_093 | 48.30 | 17.63 | 24.63 | 18.50 | 12.41 | 235 | 234 | 49.89 | 69.17 | 14.46 | 24.98 |
| sequence_02 | C_fullfallback | 48.04 | 17.63 | 24.61 | 18.56 | 12.41 | 235 | 234 | 49.89 | 68.47 | 14.60 | 25.26 |
| sequence_02 | adaptive_observed | 48.34 | 17.54 | 24.50 | 18.57 | 12.57 | 234 | 235 | 50.11 | 71.54 | 13.98 | 24.39 |
| sequence_02 | adaptive_quality | 48.64 | 17.63 | 24.63 | 18.60 | 12.59 | 237 | 232 | 49.47 | 69.30 | 14.43 | 24.72 |
| sequence_03 | baseline | 74.68 | 41.20 | 47.18 | 51.89 | 28.96 | 475 | 0 | 0.00 | 42.45 | 23.56 | 25.48 |
| sequence_03 | fixed2_093 | 75.16 | 39.95 | 46.33 | 44.82 | 23.39 | 238 | 237 | 49.89 | 68.84 | 14.53 | 26.21 |
| sequence_03 | C_fullfallback | 76.75 | 40.44 | 47.41 | 45.38 | 24.23 | 259 | 216 | 45.47 | 63.32 | 15.79 | 27.37 |
| sequence_03 | adaptive_observed | 76.70 | 40.45 | 47.41 | 45.45 | 25.44 | 259 | 216 | 45.47 | 66.90 | 14.95 | 24.43 |
| sequence_03 | adaptive_quality | 77.19 | 40.09 | 46.76 | 45.55 | 25.62 | 263 | 212 | 44.63 | 63.65 | 15.71 | 25.56 |
| sequence_04 | baseline | 55.19 | 42.69 | 35.34 | 34.75 | 24.60 | 670 | 0 | 0.00 | 43.44 | 23.02 | 25.08 |
| sequence_04 | fixed2_093 | 58.13 | 38.94 | 35.04 | 31.27 | 22.84 | 335 | 335 | 50.00 | 69.29 | 14.43 | 24.91 |
| sequence_04 | C_fullfallback | 58.13 | 38.94 | 35.04 | 31.27 | 22.84 | 335 | 335 | 50.00 | 61.66 | 16.22 | 28.68 |
| sequence_04 | adaptive_observed | 58.13 | 38.94 | 35.04 | 31.27 | 22.84 | 335 | 335 | 50.00 | 71.32 | 14.02 | 24.64 |
| sequence_04 | adaptive_quality | 58.13 | 38.94 | 35.04 | 31.27 | 22.84 | 335 | 335 | 50.00 | 70.60 | 14.16 | 24.45 |

## Per-class mAP50–95 (%)

| Class | baseline | fixed2_093 | C_fullfallback | adaptive_observed | adaptive_quality |
|---|---:|---:|---:|---:|---:|
| person | 16.00 | 13.59 | 13.80 | 14.20 | 14.33 |
| bicycle | 0.03 | 0.01 | 0.03 | 0.02 | 0.03 |
| car | 28.01 | 24.90 | 25.03 | 25.36 | 25.58 |
| motorcycle | 4.71 | 2.42 | 2.32 | 2.43 | 2.42 |
| bus | 54.95 | 52.62 | 52.57 | 52.57 | 52.57 |
| truck | 8.75 | 7.01 | 7.09 | 7.12 | 7.31 |
| traffic_light | 5.75 | 4.54 | 4.51 | 5.00 | 4.99 |
| stop_sign | 5.76 | 5.07 | 4.56 | 4.67 | 4.47 |

## Findings and limits

Quality-aware adaptive mAP50–95 is -1.53 percentage points relative to Dense, and +0.22 points relative to original adaptive. Its detector-call savings are 47.63% versus Dense. These are descriptive differences on this subset, without a statistical significance claim or matched-compute guarantee.

Crowds/explicit ignore regions are excluded from the target denominator. After regular IoU-greedy matching, unmatched same-class predictions are ignored per IoU threshold using crowd intersection/prediction-area (many-to-one) or explicit-ignore IoU (one-to-one). This follows COCO crowd overlap conventions while retaining the repository’s regular matching and AP implementation. These are not official COCO leaderboard metrics or streaming AP. No independent audit of annotation quality or upstream archive authenticity was performed.

The taxonomy, scenes, resolution and annotation source differ from the prior pseudo-label benchmark; do not directly compare their numerical accuracies. Five validation sequences are insufficient to claim performance on all Argoverse-HD or other driving conditions. No held-out testing, condition-stratified night analysis, real-time deadline test, timing repetitions or tuning was performed. A diagnostic midpoint frame from each sequence was visually inspected; sequence_01 includes a dark scene. No provided day/night taxonomy was assumed.

The stock Dense detector emitted one zero-height box at the top image border (sequence_02 frame 137, confidence 0.001948, traffic_light). It was retained and verified to count as a false positive at every IoU threshold; all other predictions have positive extents and all predictions are finite. This does not invalidate annotation coordinates or change the scoring protocol. No output filtering or rerun was performed.

Recommended next steps: independently verify the evaluator against an official COCO evaluation if official metric comparability is needed; repeat timing under controlled conditions (and an available, validated MPS environment if desired); then use separately authorized broader validation and freeze settings before held-out testing.

## Reproducibility and validation

All five configurations were recovered by equality checks against saved observation/quality-run settings. Only the eight-class map, explicit CPU device and absolute checkpoint path change. Full configs and reference SHA-256 hashes are in `execution_plan.json`. The snapshot contains the uncommitted pipeline and tests used for this run.

PyTorch reports MPS built=True, available=False in this execution environment. It was checked before full execution and reconfirmed; no GPU compatibility inference was possible. All methods actually resolved to CPU, as recorded in provenance and `device_check.json`. This does not establish that the hardware lacks a GPU. OpenCV flow/tracking remain on CPU.

Checkpoint SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`. Shared input/target manifest SHA-256: `81015fde270aa3c36a581132bc122e32fd2420dcbdf0553da0b61db6e3aaddd5`.

Runtime: Python 3.14.6; macOS-26.1-arm64-arm-64bit-Mach-O. Packages: `{"numpy": "2.5.3", "opencv-python": "5.0.0.93", "scipy": "1.18.1", "torch": "2.14.0", "ultralytics": "8.4.149"}`.

Validation: 55 synthetic tests passed (39 existing plus 16 integration checks); full image audit passed; the 30-frame development check passed for each method. Saved old Dense and quality-aware statistics reproduce their original metrics exactly with the extended evaluator. Final offline checks verified every FULL output against Dense exactly, prediction finiteness/classes/nonnegative extents, the single degenerate prediction’s FP treatment, per-sequence reset evidence, saved AP reproduction, input hash equality, and copied-source equality. The first audit attempt failed on tuple-versus-JSON-list comparison, before model loading or output creation; the serialization comparison was corrected and the audit passed. There were no inference failures or dropped frames in the completed benchmark.

Exact commands from the repository root:

```sh
PYTHONPATH=extracted .venv/bin/python -m unittest discover -s tests -q
PYTHONPATH=extracted .venv/bin/python -m adaptive_pipeline.experiments.run_argoverse --dataset data/argoverse_hd --validate-only --out runs/argoverse_hd_audit_v1
PYTHONPATH=extracted .venv/bin/python -m adaptive_pipeline.experiments.run_argoverse --dataset data/argoverse_hd --max-frames 6 --out runs/argoverse_hd_dev_v1
PYTHONPATH=extracted .venv/bin/python -m adaptive_pipeline.experiments.run_argoverse --dataset data/argoverse_hd --out runs/argoverse_hd_benchmark_v1
PYTHONPATH=extracted .venv/bin/python runs/argoverse_hd_benchmark_v1/verify_and_report.py
```

Reruns require fresh output directory names. Keep development results separate from final benchmark results.

Integration files: `evaluation/argoverse.py` (strict native loader/audit), `experiments/run_argoverse.py` (controlled runner/config checks), `adaptive_framing/sequence.py` (optional native target fields), `evaluation/evaluator.py` (ignore-aware scoring/statistics), `evaluation/provenance.py` (native target/ignore hashes), `run.py` (native target routing/per-sequence metrics), `tests/test_argoverse.py`, and `docs/ARGOVERSE_HD.md`. Existing uncommitted algorithm changes were preserved. No commits or pushes.

Artifacts: `dataset_audit.json`, `execution_plan.json`, `device_check.json`, `verification.json`, `summary.csv`, `per_sequence.csv`, `source_snapshot/`, `visual_audit.jpg` (provided annotations beside saved Dense predictions at confidence >=0.25 on one midpoint frame per sequence), and for every method `metrics.json`, `frames.csv`, `input_manifest.json`, `eval_stats.npz`, `preds.npz`, `prediction_details.npz`.
