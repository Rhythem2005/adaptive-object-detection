# Argoverse-HD evaluation

This adapter evaluates the provided annotations in the downloaded export;
it does not generate labels, train a model, or modify the dataset. The code
reads `annotations.json`, `manifest.json`, and their referenced images under
`data/argoverse_hd/`. Original source paths are provenance, not local inputs.

## Dataset and taxonomy

The inspected subset has five sequences, 3,416 images at 1920×1200, and
57,711 annotations. Of these, 55,412 are regular scoring targets and 2,299
are crowd regions. No annotations have `ignore=true`. No frames are entirely
unannotated; the adapter nevertheless preserves empty and crowd-only frames.

| Dataset category ID | Name | Stock YOLO COCO index |
|---|---|---|
| 0 | person | 0 |
| 1 | bicycle | 1 |
| 2 | car | 2 |
| 3 | motorcycle | 3 |
| 4 | bus | 5 |
| 5 | truck | 7 |
| 6 | traffic_light | 9 |
| 7 | stop_sign | 11 |

All eight provided classes are supported. Other COCO predictions are filtered
by the existing detector class-map mechanism. The old seven-class mapping is
unchanged for old workflows. This necessary taxonomy change also changes the
set of objects available to the adaptive scheduler.

Boxes are provided as pixel `x,y,width,height`, converted in memory to
`class,x1,y1,x2,y2`. Coordinates retain the provided values, without clipping
or rounding. Eight annotations extend one pixel beyond the right edge; the
audit records their IDs and boxes. These are retained, not silently repaired.

The export retains 24 entries in its sequence metadata arrays. Only five have
images/manifest entries; evaluation does not infer images from unused metadata.
The adapter cross-checks image IDs, paths, sequence IDs, source names, frame
indices, category metadata, annotation IDs, area, flags and finite positive
boxes. Missing, extra, corrupt, dimension-mismatched, duplicate, orphan, or
unknown-category records fail before loading a model. All images are decoded
during validation. Frames sort by manifest index within each sequence; indices
must be contiguous from zero and source timestamps strictly increasing, without
large gaps. The pipeline resets at each sequence boundary.

The source timestamp suffix is in nanoseconds, consistent with the
[Argoverse API](https://github.com/argoverse/argoverse-api/blob/master/argoverse/data_loading/synchronization_database.py).
Measured median deltas are approximately 33,300,000 ns (30.030 FPS). Exact
per-sequence deltas are saved in `dataset_audit.json`. Processing is offline,
one call per provided frame; no frame-rate conversion, encoding or pacing.

## Metric protocol

Use the repository's `DetectionEvaluator`, current AP protocol, IoU thresholds
0.50:0.05:0.95, and its existing regular-target IoU-greedy matching. AP is
101-point interpolated and averaged over classes with regular targets. P/R
and macro F1 use the maximum smoothed mean-F1 confidence; fixed-confidence
P/R/F1 at approximately 0.25 are also saved. The selected confidence may differ
by method/sequence, so fixed-confidence metrics should accompany comparisons.

Crowds and explicit ignore boxes are excluded from the recall denominator.
After regular-target matching, unmatched same-class predictions can be ignored
separately at each IoU threshold. Crowd overlap is intersection divided by
prediction area and permits multiple matches. Explicit non-crowd ignore boxes
use one-to-one IoU matching in descending prediction confidence. Matched regular
targets always take precedence. The overlap/multiplicity convention follows
the [COCO API](https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c);
the remaining matching and AP calculations retain the repository's protocol.
An `ignored` matrix is saved beside TP flags for offline rescoring/bootstrap.
False positives on empty frames remain in the statistics. Subsets with no
regular targets return null detection metrics, not fabricated zeros.

These are **repository detection metrics on provided Argoverse-HD annotations**,
not official COCO leaderboard metrics or streaming AP. Official COCO uses
different matching, AP accumulation, area ranges and max-detection protocols.
They are not directly comparable to the previous YOLO pseudo-label benchmark.
The export claims dataset-provided labels; no independent annotation-quality
audit or upstream archive authenticity verification was performed.

## Frozen method settings and execution

`experiments/run_argoverse.py` checks current presets against saved configs in
`runs/improvements_observation_v1/` and `runs/improvements_quality_v1/`, including
the checkpoint hash and AP protocol. It fails if these no longer agree. It
changes only the class map, device (`cpu`, matching prior resolved hardware),
and model path to the absolute local checkpoint. A missing checkpoint fails
before YOLO construction rather than triggering a download.

Methods, run sequentially in this order:

1. `baseline`: Dense YOLO on every frame.
2. `fixed2_093`: FULL every two frames, original flow, confidence decay 0.93.
3. `C_fullfallback`: original adaptive, `k_max=4`, novelty 0.10, failure fraction
   0.20, uncertain refresh interval 2, confidence decay 0.93, `max_local_rois=0`.
4. `adaptive_observed`: the same adaptive settings plus observation anchoring.
5. `adaptive_quality`: the same adaptive settings plus anchoring and quality-aware flow.

`use_local` retains its recovered value; the adaptive presets' zero ROI limit
forces full fallback whenever LOCAL is requested. No policy settings are tuned
on this subset. Optional risk/budget/camera/appearance features remain off.

Common detector settings: existing `yolov8n.pt`, CPU, FP32, size 640, confidence
0.001, NMS IoU 0.7, maximum 300 detections. Full config dictionaries and reference
file hashes are in each run's `execution_plan.json` and `metrics.json`.

MPS was checked before the full benchmark and reconfirmed during execution:
PyTorch 2.14.0 reports `torch.backends.mps.is_built() == True` and
`torch.backends.mps.is_available() == False` in this execution environment.
Therefore a GPU compatibility inference could not be attempted, and CPU was
used consistently for every method. This is a runtime availability result,
not a claim that the Mac hardware lacks a GPU. The benchmark records the
check in `device_check.json` and actual resolved devices in method provenance.
OpenCV flow and tracking remain on CPU in either case. Recheck availability
before any separately authorized future GPU run, perform a small compatibility
check first, and retain one device/settings combination across the suite.

From the repository root:

```sh
PYTHONPATH=extracted .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=extracted .venv/bin/python -m adaptive_pipeline.experiments.run_argoverse --dataset data/argoverse_hd --validate-only --out runs/argoverse_hd_audit_v1
PYTHONPATH=extracted .venv/bin/python -m adaptive_pipeline.experiments.run_argoverse --dataset data/argoverse_hd --max-frames 6 --out runs/argoverse_hd_dev_v1
PYTHONPATH=extracted .venv/bin/python -m adaptive_pipeline.experiments.run_argoverse --dataset data/argoverse_hd --out runs/argoverse_hd_benchmark_v1
```

All output directories must be fresh. `--validate-only` never loads a model.
`--max-frames` takes the first N frames **per sequence** and marks the output
as a development check. Benchmark uses every frame of all five sequences.
Reuse the same commands with new output directory names for later repetitions.

Each output includes the dataset audit, exact command/config plan, copied
pipeline/test sources and a working-tree diff. Each evaluated method includes
image SHA-256 hashes, native target/ignore hashes, frame telemetry, predictions,
temporal prediction details, matching statistics, aggregate detection and
efficiency metrics, decision/staleness breakdowns, and per-sequence metrics.
The runner verifies identical input/target hashes across all methods.

## Efficiency and timing

`full_detector_calls` counts evaluated frames receiving whole-frame detection.
`roi_crops_total` and `roi_pass_rate` capture additional LOCAL work, including
fallback frames. In this suite no ROI inference should occur during measured
processing. Warmup per method is excluded: three full detections plus one
batch of two crops, inherited from the existing detector wrapper.

Savings in detector calls are `1 - full_detector_calls / frames` when there
are no ROI passes. Report tracking-only fraction separately if LOCAL is used.
GFLOPs savings use the existing model's padded-input cost estimate; this is
an analytical estimate of detector arithmetic, not measured total compute,
energy or wall-time savings. Tracker and scheduler overhead are excluded from
that estimate and included in pipeline latency.

`fps_end_to_end` is historically named: it is `1000 / mean(t_total_ms)` for
pipeline processing only. It excludes JPEG decode, annotation loading/scoring,
model loading, warmup, input hashing and output I/O. Report mean, P50, P95,
P99, max latency as well as FPS. Decode time is separately recorded per frame.
The aggregate `cpu_ms_per_frame` includes non-pipeline work after warmup and
has a broader scope; no per-sequence CPU number is inferred from it. Runs are
sequential, one repetition per method, so thermal/background/order effects
limit performance conclusions. Offline throughput does not demonstrate a
live deadline guarantee.
