# Adaptive Vision

**Adaptive object detection for driving videos through frame scheduling, temporal tracking, and selective re-detection.**

Adaptive Vision is a computer vision research prototype that investigates when an object detector needs to run again. Instead of applying YOLO to every frame, it combines scene-change signals, track reliability, optical flow, and a bounded refresh interval to allocate detector work across a video. It produces predictions on processed frames even when inference is skipped.

The current implementation uses **stock YOLOv8n** and a modular evaluation harness. Its experiments demonstrate an efficiency–accuracy trade-off: adaptive scheduling can reduce detector computation, while propagated predictions lose accuracy between full detections. Superiority over fixed-interval scheduling has not been established.

## Contents

- [Project objectives](#project-objectives)
- [System architecture](#system-architecture)
- [Methods and implementation](#methods-and-implementation)
- [Dataset and preparation](#dataset-and-preparation)
- [Experimental results](#experimental-results)
- [Setup and usage](#setup-and-usage)
- [Configuration](#configuration)
- [Evaluation protocol and outputs](#evaluation-protocol-and-outputs)
- [Repository structure](#repository-structure)
- [Prototype status and next steps](#prototype-status-and-next-steps)
- [Experimental improvements awaiting evaluation](#experimental-improvements-awaiting-evaluation)

## Project objectives

Driving videos contain repeated visual information, but also camera motion, changing illumination, occlusion, and newly appearing objects. Processing every frame with a detector is expensive; skipping frames without updating predictions can miss those changes.

This project investigates four questions:

1. Can inexpensive visual signals identify frames that need fresh detection?
2. Can tracking and optical flow preserve useful predictions between detector calls?
3. Can local re-detection confirm uncertain tracks at lower cost than a full-frame pass?
4. How do adaptive policies compare with dense detection and fixed skipping under the same evaluation conditions?

The intended application is efficient video perception for driving and urban monitoring. The repository reports prototype and benchmark results, rather than performance from a deployed driving system.

## System architecture

The active implementation is in [`extracted/adaptive_pipeline/`](extracted/adaptive_pipeline/). Its central orchestrator is [`pipeline.py`](extracted/adaptive_pipeline/pipeline.py).

```text
Input frame
    |
    +--> Lightweight signals: luminance, background novelty, detection age
    |
    +--> Temporal propagation: Kalman prediction + sparse optical flow
    |
    v
Adaptive controller: scene signals + track health + uncertainty
    |
    +-- FULL  --> Whole-frame YOLO detection --> Reset keyframe and update tracks
    |
    +-- LOCAL --> Expand / merge ROIs --> Batched re-detection --> Confirm tracks
    |                                                       |
    |                           Failed confirmation / cost limit --> FULL
    |
    +-- SKIP  --> Output propagated tracks without a detector call
    |
    v
Frame-level predictions + decisions + stage timings + estimated detector cost
```

Two input modes serve different purposes:

- **Offline evaluation:** reads every image in sequence order, resets pipeline state at each video boundary, and evaluates predictions for every frame. It does not drop frames through the live capture buffer.
- **Paced video processing:** uses a capture thread and a thread-safe latest-frame buffer with capacity one. New arrivals replace unread frames when processing falls behind. These buffer replacements are reported separately from deliberate detector skips.

A one-slot buffer limits queued work; it does not by itself guarantee zero latency or prove a deployment-level latency improvement.

## Methods and implementation

### Adaptive frame scheduling

[`FrameController`](extracted/adaptive_pipeline/adaptive_framing/controller.py) chooses one of three actions before inference:

| Action | When it is selected | Processing |
|---|---|---|
| `FULL` | Initial frame, refresh deadline, illumination or scene change, widespread flow failure, or too many tracks requiring verification | Detect the complete frame and re-anchor track state |
| `LOCAL` | A limited number of tracks require verification and the ROI budget permits it | Re-detect selected regions and confirm those tracks |
| `SKIP` | No refresh condition fires | Return propagated track predictions without invoking YOLO |

The controller uses a 160-pixel-wide thumbnail for inexpensive scene signals. Background novelty measures changed pixels outside tracked boxes, helping identify changes that existing tracks do not explain. Illumination changes are measured against the last full-detection keyframe using mean LAB lightness. A hard `k_max` bound prevents indefinite propagation.

Track verification is requested for unreliable optical flow or stale tracks with low detector confidence. Decisions and trigger reasons are logged so their effects can be inspected and ablated.

### Detection and class mapping

[`YOLODetector`](extracted/adaptive_pipeline/detection/detector.py) wraps YOLOv8n and returns arrays containing pixel-space boxes, confidence, and dataset class IDs. The main experiments use a full-frame inference size of 640, confidence threshold `0.001`, NMS IoU `0.7`, and a maximum of 300 detections.

COCO classes are mapped into the driving dataset vocabulary. Stock YOLOv8n has no separate `rider` class; unsupported classes are filtered. On ordinary `FULL` frames with enhancement and refinement disabled, raw detector output is used directly rather than being filtered by track age.

### Tracking and optical flow

[`Tracker`](extracted/adaptive_pipeline/tracking/tracker.py) implements batched eight-dimensional constant-velocity Kalman filtering, with class-aware IoU matching through Hungarian assignment. Detector observations re-anchor tracks; optical-flow observations enter as noisier measurements.

[`FlowPropagator`](extracted/adaptive_pipeline/tracking/flow.py) uses pyramidal Lucas–Kanade optical flow on a configurable grid inside each box. The defaults use a 4 × 4 grid and a 640-pixel-wide working image. Forward–backward consistency filters unreliable points; median displacement and scale estimates propagate boxes.

Propagated confidence decays with time since the last detector confirmation. Tracks with repeated flow failure have a bounded coasting period. This allows temporal predictions to bridge detector gaps while exposing uncertainty to the controller.

### Selective re-detection and fusion

The [`selective_redetection`](extracted/adaptive_pipeline/selective_redetection/selective_redetection.py) module expands boxes with context, clamps crops to image boundaries, enforces a minimum crop size, and merges overlapping ROIs. Crops are processed in a batch, translated back into frame coordinates, and deduplicated with class-aware NMS.

By default, failed local confirmation escalates to a full detection on the same frame. Excessive ROI area also triggers `FULL`. This makes `LOCAL` a track-confirmation step, with a measurable cost when a crop pass fails to avoid a subsequent full pass.

An optional keyframe-refinement path implements additional selective re-detection on full detections. It is disabled in the main reported comparisons.

### Illumination handling

The [`IlluminationAdapter`](extracted/adaptive_pipeline/environmental_adaptation/illumination.py) offers LAB-channel CLAHE for low-light detector inputs, with hysteresis to avoid switching repeatedly around the threshold. Enhancement is applied to full frames or crops only when inference requires it.

**CLAHE is disabled in the main reported results.** Illumination-triggered refresh and image enhancement are separate features; the existing comparison does not demonstrate improved low-light detection from CLAHE.

## Dataset and preparation

### Prepared video dataset

The project uses a **BDD100K-derived driving-video subset/mirror**, rather than the complete BDD100K release. Locally available raw videos are organised under `data/videos_by_condition/`; prepared images and YOLO labels are stored under `Prep-Data/`.

The saved dataset audit records:

| Split | Source videos | Frames / label pairs |
|---|---:|---:|
| Training | 584 | 206,304 |
| Validation | 123 | 43,350 |
| Test | 122 | 43,772 |
| **Total** | **829** | **293,426** |

Splits are separated by source video. The saved audit reports passing checks for duplicates, missing pairs, invalid labels, video overlap, and condition splitting. These structural checks do not establish the accuracy or independence of the annotations.

Labels use normalized YOLO format:

```text
class_id center_x center_y width height
```

The declared classes are `car`, `bus`, `truck`, `pedestrian`, `rider`, `bicycle`, and `traffic light`. The prepared dataset contains no `rider` annotations. Evaluated AP therefore reflects the represented classes, not seven independently evaluated categories.

The prepared frames are predominantly daytime: **282,776 day**, **10,050 night**, and **600 evening/low-light** frames. Results on a daytime subset should not be interpreted as evidence of performance across all lighting conditions.

### Targeted frame-selection subset

A separate preparation experiment selected **11,680 training frames from 584 videos**, with **20 frames per video**. Each video was divided into temporal bins, then candidates were prioritised by rare-class presence, small objects, and crowding. A sampling manifest records the source video, frame index, temporal bin, and selection reason.

The saved validation summary reports deterministic selection, valid label structure, source-identical image/label pairs, disjoint video splits, and unchanged validation/test file hashes. At the same frame budget, the selected subset contains:

| Annotation coverage | Baseline sampled subset | Targeted subset |
|---|---:|---:|
| Bicycle instances | 199 | 438 |
| Bus instances | 1,483 | 3,065 |
| Pedestrian instances | 8,528 | 12,752 |
| Small-object instances | 25,651 | 34,289 |

Small objects were defined using the saved normalized-area threshold, approximately `0.00190547`. These numbers demonstrate improved **data coverage**, not improved detector accuracy.

Dataset audits and manifests live in the local `Prep-Data*/` directories, which are excluded from Git. The preparation selects existing data; it should not be described as collecting or manually annotating a new raw dataset.

### Fine-tuning assets

[`Model/Experiment-B-Targeted-Finetune.ipynb`](Model/Experiment-B-Targeted-Finetune.ipynb) contains preflight checks and a gated fine-tuning workflow for the targeted subset. Its current configuration specifies 30 epochs, batch size 64, image size 640, and MPS execution, with `RUN_TRAINING = False` by default.

The notebook has no saved execution outputs, and a completed targeted-training checkpoint is not present in the current checkout. The adaptive results below use stock weights. [`Experiment-B-Report.md`](Experiment-B-Report.md) records an earlier five-epoch proposal and should be read as historical documentation; the current notebook configuration differs.

## Experimental results

### Matched validation comparison

The following runs use the same **15 daytime validation videos, 4,680 frames, and 42,195 labelled instances**, with the `current` AP protocol. They compare dense detection, fixed scheduling with optical flow, and adaptive variants.

| Policy / preset | mAP@50 | mAP@50–95 | Pipeline FPS | Mean processing time (ms) | P95 processing time (ms) | Estimated detector compute |
|---|---:|---:|---:|---:|---:|---:|
| Dense detection (`baseline`) | 98.95% | 98.05% | 59.35 | 16.85 | 18.55 | 100.00% |
| Fixed interval 2 (`fixed2_093`) | 91.44% | 82.07% | 85.90 | 11.64 | 22.83 | 50.00% |
| Adaptive with LOCAL (`adaptive_C`) | 91.01% | 81.15% | 60.72 | 16.47 | 52.72 | 64.98% |
| Adaptive FULL/SKIP (`C_fullfallback`) | 91.78% | 82.62% | 78.25 | 12.78 | 24.66 | 50.34% |
| Adaptive FULL/SKIP, `k_max=3` (`C_k3_no_local`) | 91.96% | 82.85% | 79.11 | 12.64 | 23.69 | 51.13% |
| Adaptive FULL/SKIP, `k_max=5` (`C_k5_no_local`) | 91.76% | 82.57% | 80.43 | 12.43 | 23.55 | 49.94% |

The tracked [`results/metrics.csv`](results/metrics.csv) contains rounded comparison values. Detailed source records are local `runs/val_ablation_final/<preset>/metrics.json` and `frames.csv`; these run directories are not included in a fresh clone. The table uses aggregate pipeline timing from those records.

**What the comparison establishes:**

- `C_fullfallback` processed 2,356 frames with full detection and propagated predictions on 2,324 frames. Relative to dense detection, it achieved **31.8% higher pipeline throughput**, **24.2% lower mean processing time**, and **49.7% less estimated detector compute**.
- Its mAP@50–95 fell by **15.44 percentage points**. Mean processing time improved, while P95 processing time increased. This is an efficiency–accuracy trade-off, not accuracy parity or uniformly lower latency.
- Fixed interval 2 was faster at nearly the same estimated detector compute. The adaptive variant had slightly higher mAP, but these point estimates do not establish a statistically significant advantage or general superiority.
- The LOCAL variant produced 158 LOCAL outcomes and 1,994 ROI crops. It had 670 `local_escalate` triggers and an ROI confirmation rate of approximately 19.1%, explaining why crop work did not deliver the expected efficiency benefit at this operating point.

`C_fullfallback` sets `max_local_rois=0`: track-refresh requests use `FULL`, even though `use_local=True` remains in its saved configuration. It should be described as an adaptive FULL/SKIP policy.

### Full-test fixed-skip control

The locally saved full-test run is **`fixed2_093`**, consistent with [`results/final_config.json`](results/final_config.json). It is a fixed-scheduling control, not an adaptive test result.

| Measure | Saved result |
|---|---:|
| Source videos / frames | 122 / 43,772 |
| Labelled instances | 385,187 |
| Precision / recall | 91.08% / 89.88% |
| mAP@50 / mAP@50–95 | 89.96% / 80.00% |
| Pipeline FPS | 92.46 |
| Mean processing time | 10.82 ms |
| Estimated detector compute | 50.00% of the dense cost model |
| FULL / SKIP outcomes | 21,887 / 21,885 |

Source: local `runs/test_final/fixed2_093/metrics.json`. A matched full-test dense/adaptive comparison is still needed before making adaptive generalisation claims. The standalone `run_final_test.py` currently contains a different adaptive configuration and output directory; it does **not** reproduce this saved fixed-skip run.

### How to interpret the measurements

- **Accuracy:** the evaluator scores predictions against the prepared labels. Very high stock-model baseline AP warrants independent verification of label provenance. If labels were generated by the same detector, scores principally measure agreement with dense detector output, rather than independently established human-labelled accuracy.
- **Throughput:** `fps_end_to_end` is calculated from pipeline processing time, including signals, propagation, scheduling, inference, and track updates. It excludes image decoding, label loading, metric computation, and display. It is not complete application FPS or capture-to-display latency.
- **Compute:** the GFLOP fraction estimates detector work from full-frame and ROI calls. It excludes tracking/flow computation and is not measured energy, power, or total system compute.
- **Statistics:** run comparisons must use the same frames, labels, configuration scope, and AP protocol. Existing exported bootstrap summaries should be reconciled with the actual paired run records before their intervals are used to support a claim.

### Historical experiments

Earlier phases explored capture buffering, illumination enhancement, crop-based re-detection, and temporal tracking. Their telemetry remains under [`benchmarks/`](benchmarks/) and [`benchmarks/historical/`](benchmarks/historical/).

The historical 425-image ground-truth comparison reported mAP@50 of 21.57% for plain YOLOv8n and 18.46% for a Phase 5 crop-based variant; it highlighted weak small-object recall without establishing an overall improvement. See [`ground_truth_phase5_evaluation.json`](benchmarks/historical/ground_truth_phase5_evaluation.json).

Earlier reports of increased detection counts should not be interpreted as verified recovery of ground-truth objects. Historical image evaluations, streaming runs, and the current sequence harness have different datasets and timing scopes, so their numbers are not interchangeable.

## Setup and usage

### Environment

From a shell with Git and Python installed:

```bash
git clone https://github.com/Rhythem2005/adaptive-object-detection.git
cd adaptive-object-detection

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The inspected local environment uses Python **3.14.6**, PyTorch **2.14.0**, and Ultralytics **8.4.149**. [`requirements.txt`](requirements.txt) pins the main Python packages; it does not independently pin PyTorch or the entire transitive environment. Record the actual environment and selected device with each experiment.

### Run the test suite

Validate system functionality and correctness checks across all components:

```bash
PYTHONPATH=extracted python -m unittest discover tests
```

The commands below run from the repository root. `PYTHONPATH=extracted` makes the active package available without changing directories. Data must be provisioned locally; cloning the repository does not download the prepared dataset. Confirm that `yolov8n.pt` is available before starting inference.

### Prepare the input layout

```text
Prep-Data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The current flat filenames follow `<video>_f<frame>.jpg`, such as `day_1000_f000019.jpg`. Use the explicit sequence expression shown below: the generic default expression does not match this naming convention. Labels are inferred by replacing `/images/` with `/labels/`; `--labels` can specify an alternative mirrored directory.

### Check the entry points

```bash
PYTHONPATH=extracted python -m adaptive_pipeline.run --help
PYTHONPATH=extracted python -m adaptive_pipeline.experiments.run_ablations --help
```

### Run a small baseline check

```bash
PYTHONPATH=extracted python -m adaptive_pipeline.run \
  --preset baseline \
  --images Prep-Data/images/val \
  --seq-regex '^(?P<video>.+?)_f(?P<frame>\d+)$' \
  --max-videos 2 --max-frames 20 \
  --ap-protocol current \
  --out runs/readme_smoke/baseline
```

This limited run checks input discovery and execution. Its metrics are not comparable with the 4,680-frame table. The CLI currently prints a fixed test-set reference count even for validation input; the audited validation split contains 123 videos and 43,350 frames.

### Re-run the validation comparison

```bash
PYTHONPATH=extracted python -m adaptive_pipeline.experiments.run_ablations \
  --presets baseline fixed2_093 adaptive_C C_fullfallback C_k3_no_local C_k5_no_local \
  --images Prep-Data/images/val \
  --seq-regex '^(?P<video>.+?)_f(?P<frame>\d+)$' \
  --max-videos 15 \
  --ap-protocol current \
  --out runs/validation_reproduction
```

The saved comparison used the first 15 sequence names in sorted discovery order. With the audited dataset, this selects `day_1`, `day_1004`, `day_1016`, `day_1038`, `day_1058`, `day_1060`, `day_1115`, `day_1149`, `day_1158`, `day_1162`, `day_117`, `day_1171`, `day_1175`, `day_1216`, and `day_124`. Confirm the selected frame total is 4,680. Changes to input files, dependencies, device, or AP protocol can change results; throughput is hardware-dependent.

For broader method studies, replace `--presets ...` with a suite:

| Suite | Purpose |
|---|---|
| `sanity` | Dense baseline check |
| `propagation` | Hold, Kalman, and optical-flow propagation |
| `main` | Dense, default adaptive, and fixed-interval policies |
| `pareto` | Novelty/refresh-interval sweep against fixed scheduling |
| `components` | Controller signals, track health, LOCAL, and uncertainty ablations |
| `illumination` | CLAHE comparisons |
| `refine` | Optional keyframe re-detection comparisons |

Use validation data to tune policies. Evaluate the frozen choice on test data separately. Generic suite defaults differ from the tuned presets in the result table.

### Run the saved full-test control configuration

```bash
PYTHONPATH=extracted python -m adaptive_pipeline.run \
  --preset fixed2_093 \
  --images Prep-Data/images/test \
  --seq-regex '^(?P<video>.+?)_f(?P<frame>\d+)$' \
  --ap-protocol current \
  --out runs/test_reproduction/fixed2_093
```

### Process a paced video

Replace the sample path with a local video:

```bash
PYTHONPATH=extracted python -m adaptive_pipeline.run \
  --preset C_fullfallback \
  --video data/videos_by_condition/day/50.mp4 \
  --realtime \
  --out runs/live_demo
```

This mode writes `realtime.json` with processing telemetry and buffer counts. It does not provide labelled AP evaluation or an annotated display. Without `--realtime`, video frames are extracted to a temporary directory for sequential efficiency evaluation; account for the temporary image storage required.

### Compare finished runs statistically

```bash
PYTHONPATH=extracted python -m adaptive_pipeline.experiments.compare \
  runs/validation_reproduction/fixed2_093 \
  runs/validation_reproduction/C_fullfallback \
  --n-boot 200 --ap-protocol current
```

The tool performs paired per-video bootstrap comparison and reports the second run minus the first. Both runs must contain `eval_stats.npz` from identical evaluation sequences.

## Configuration

Configuration is defined by [`PipelineConfig`](extracted/adaptive_pipeline/shared/config.py), with named overrides in [`presets.py`](extracted/adaptive_pipeline/experiments/presets.py). Every run saves its resolved configuration.

| Parameter | Base default | Purpose |
|---|---:|---|
| `policy` | `adaptive` | Adaptive, fixed, or always-detect scheduling |
| `imgsz` / `roi_imgsz` | `640` / `320` | Full-frame and ROI inference sizes |
| `k_max` | `8` | Maximum age before a full refresh |
| `novelty_thr` | `0.20` | Changed-pixel fraction outside tracks |
| `fail_frac_thr` | `0.30` | Flow-failure fraction triggering FULL |
| `max_local_rois` | `3` | Maximum number of tracks eligible for LOCAL |
| `roi_margin` / `roi_min_size` | `0.5` / `96` | Per-side context margin and minimum crop size |
| `roi_max_area_frac` | `0.35` | ROI-area budget before FULL fallback |
| `local_miss_policy` | `escalate` | Response to failed local confirmation |
| `k_uncertain` | `3` | Age before low-confidence track verification |
| `conf_decay` | `0.97` | Per-frame propagated-confidence decay |
| `use_clahe` / `keyframe_refine` | `False` / `False` | Optional enhancement and refinement |

Base defaults are not the tuned operating point. `adaptive_C` and `C_fullfallback` use `k_max=4`, `novelty_thr=0.10`, `conf_decay=0.93`, `k_uncertain=2`, and `fail_frac_thr=0.20`. Their LOCAL budgets are five and zero respectively.

Use `--set key=value` to override fields; values are parsed as JSON when possible. For example, add `--set device=mps` or `--set use_clahe=true` to a run command. With dataset-specific weights that output dataset class IDs directly, use `--set class_map=null` after verifying the checkpoint class order.

## Evaluation protocol and outputs

The evaluator uses class-aware greedy box matching at IoU thresholds from 0.50 to 0.95 and 101-point interpolated AP. It provides `current` and `legacy` interpolation protocols. Comparisons must use the same protocol. Precision and recall are reported at the selected mean-F1 operating point, with additional fixed-confidence measurements at `0.25`.

Sequence runs warm up the detector, reset state at every video boundary, and evaluate every frame. Reports include overall metrics and subsets grouped by decision and detection age.

| Output | Contents |
|---|---|
| `metrics.json` | Resolved configuration, AP protocol, accuracy, decision/staleness breakdowns, and efficiency |
| `frames.csv` | Per-frame actions, trigger reasons, stage timings, ROI activity, luminance, and cost estimates |
| `eval_stats.npz` | Matching statistics and tags for paired comparisons; generated for labelled evaluation |
| `preds.npz` | Predictions when `--save-preds` is enabled |
| `summary.csv` / `summary.md` | Comparison tables from an ablation suite |
| `realtime.json` | Paced-video telemetry and capture-buffer accounting |

`detector_skip_rate` means the fraction of frames without a **full-frame** pass; LOCAL frames may still incur detector work. `tracking_only_rate` counts frames with no detector work. `roi_confirm_rate` measures ROI attempts that resolved a frame without escalation, rather than the fraction of individual boxes correctly detected.

The label loader can treat missing or malformed labels as empty. Validate data integrity before evaluation rather than relying on successful execution as a data-quality check. Memory telemetry also needs platform-aware interpretation because the RSS conversion currently assumes Linux units.

## Repository structure

```text
adaptive-object-detection/
├── README.md
├── requirements.txt
├── yolov8n.pt                         # Stock detector checkpoint
├── extracted/adaptive_pipeline/       # Active modular implementation
│   ├── pipeline.py                    # Per-frame orchestration
│   ├── run.py                         # Sequence and paced-video entry points
│   ├── adaptive_framing/              # Controller, capture, buffer, sequence discovery
│   ├── detection/                     # YOLO wrapper and detector cost model
│   ├── tracking/                      # Kalman association and optical flow
│   ├── selective_redetection/         # ROI construction, re-detection, fusion
│   ├── environmental_adaptation/      # Luminance and optional CLAHE
│   ├── evaluation/                    # Detection metrics and paired bootstrap
│   ├── telemetry/                     # Frame records and aggregate efficiency
│   ├── experiments/                   # Presets, suites, and comparison CLI
│   └── shared/                        # Configuration and box operations
├── results/                          # Tracked result exports and chosen configuration
├── benchmarks/                       # Earlier phase telemetry and historical diagnostics
├── Model/Experiment-B-Targeted-Finetune.ipynb
├── Experiment-B-Report.md             # Historical preparation/fine-tuning report
├── run_10video_experiment.py           # Earlier selected-video experiment
├── run_validation_tuning.py           # Earlier validation sweep script
├── run_final_test.py                  # Adaptive wrapper; differs from saved fixed test run
├── main.py                           # Legacy entry point
├── Prep-Data/                         # Local prepared data and audits; Git-ignored
├── Prep-Data-Sampled-Targeted/         # Local targeted subset and manifest; Git-ignored
├── data/videos_by_condition/          # Local source videos; media excluded from Git
└── runs/                             # Detailed local experiment outputs; Git-ignored
```

Use the modular package commands above for current work. The root `main.py` depends on older `config` and `src` modules absent from the current checkout. Historical diagnostics and wrapper scripts may retain older paths, subsets, or assumptions; they are retained for experiment history rather than serving as the primary setup guide.

## Prototype status and next steps

| Area | Current state |
|---|---|
| Adaptive FULL/LOCAL/SKIP controller | Implemented and exercised in sequence experiments |
| Kalman and optical-flow propagation | Implemented; predictions emitted between detector calls |
| ROI confirmation and FULL fallback | Implemented; measured overhead remains a limitation |
| Latest-frame capture buffer | Implemented; controlled deployment-latency validation remains pending |
| Prepared splits and targeted sampling | Available locally with saved audits and manifests |
| Matched validation comparisons | Saved for dense, fixed, and adaptive policies |
| Full-test control | Saved for `fixed2_093` |
| Fine-tuning workflow | Notebook prepared; completed training benefit not established |
| Low-light accuracy improvement | Not established by the main daytime comparison |

The next research priorities are to validate annotation provenance with independent human labels, improve propagated-box accuracy, reduce failed LOCAL work, and compare frozen adaptive and fixed policies on the same held-out test set. Deployment work should measure total wall-clock latency, frame age, buffer replacements, and actual resource consumption on the intended device.

This repository is a working research prototype. Its strongest current result is a measured reduction in detector work, accompanied by a quantified loss in detection accuracy and a competitive fixed-scheduling baseline.

## Experimental improvements awaiting evaluation

An opt-in implementation now adds observation-anchored prediction, flow-quality
feedback, risk-aware refresh decisions, a soft detector budget, and cost-gated
LOCAL confirmation. Optional variants cover confidence decay, low-score
association, background-aligned novelty, appearance verification, and bounded
LOCAL births. Original presets remain comparison controls.

**These changes have not been evaluated on the dataset. The results above belong
to the existing experiments and do not demonstrate improvement from the new code.**
Synthetic tests check correctness using generated inputs and a fake detector.
Dataset and real-video execution require explicit user approval.

See [the implementation and evaluation guide](docs/ADAPTIVE_IMPROVEMENTS.md) for
the new presets, ablation suites, unvalidated settings, test command and proposed
first experiment. New runs protect existing outputs, validate labels, record
input/weight provenance and include per-class Macro F1 summaries. Optional saved
prediction diagnostics support later geometry analysis.

## Argoverse-HD provided-annotation benchmark

The local five-sequence Argoverse-HD export is supported through a strict,
read-only native annotation adapter. See [the integration and execution guide](docs/ARGOVERSE_HD.md)
for class mapping, crowd handling, validation, frozen configurations and commands.
The [measured benchmark report](docs/ARGOVERSE_HD_RESULTS.md) uses the provided
dataset annotations; its scores are separate from earlier YOLO pseudo-label results.
