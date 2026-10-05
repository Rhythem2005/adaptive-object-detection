# Adaptive Vision — detector scheduling for driving video

Research question: **how much full-frame YOLOv8n computation can be replaced by
temporal propagation, and at what cost in detection quality — and does an
adaptive schedule beat a fixed one at the same compute?**

## 1. Architecture

```
frame t ──► signals (160-px thumbnail: LAB-L luminance, blurred gray; 640-px gray for flow)
        ──► propagate tracks t-1 → t  (constant-velocity Kalman predict + LK median-flow update)
        ──► FrameController.decide()  →  FULL | LOCAL | SKIP   (+ logged reasons)
              FULL : YOLOv8n on whole frame  [CLAHE if low-light & enabled] [keyframe refine if enabled]
                     output = raw detections (identical to the baseline on that frame)
                     tracker re-synchronised (Hungarian, class-aware IoU); keyframe signals reset
              LOCAL: ONE batched YOLOv8n pass on ROI crops around flagged tracks (imgsz 320)
                     confirmed tracks re-anchored; any unconfirmed → escalate to FULL same frame
              SKIP : no detector; output = propagated tracks, conf = conf_det · 0.97^age
        ──► telemetry record (per-stage ms, GFLOPs, decision, reasons)
        ──► evaluator (per-image stats, tagged by decision / staleness / video)
```

Decision rules (adaptive policy), all computed before any detector call:

| trigger | signal | action |
|---|---|---|
| init | first frame of a video | FULL |
| staleness | frames since last FULL ≥ `k_max` | FULL |
| illumination | \|L − L_keyframe\| > `lum_jump` | FULL |
| novelty | fraction of thumbnail pixels changed vs. last keyframe, **outside track boxes** > `novelty_thr` | FULL |
| track_failure | fraction of visible tracks failing flow FB-check > `fail_frac_thr` | FULL |
| flow_fail / uncertain | specific tracks failed flow, or last conf ∈ [τ_low, τ_high) and unconfirmed ≥ `k_uncertain` frames | LOCAL (≤ `max_local_rois`, ROI area ≤ `roi_max_area_frac`), else FULL |
| — | nothing fired | SKIP |

## 2. Directory structure

```
adaptive_pipeline/
├── pipeline.py                     NEW  per-frame orchestration (FULL/LOCAL/SKIP)
├── run.py                          NEW  run one preset on a split / video / live
├── adaptive_framing/
│   ├── controller.py               NEW  scheduler: signals → decision + reasons
│   ├── sequence.py                 NEW  deterministic in-order frame/label loading
│   ├── capture.py                  CHG  live demo only; metadata race + stop flag fixed
│   └── frame_manager.py            —    unchanged (live demo only)
├── environmental_adaptation/illumination.py   CHG  thumbnail luminance, hysteresis, cached CLAHE, detector inputs only
├── detection/detector.py           CHG  (N,6) arrays in dataset ids, batched ROI pass, val-matched settings, GFLOPs model
├── selective_redetection/selective_redetection.py  CHG  ROI tier + corrected keyframe refiner
├── tracking/
│   ├── tracker.py                  CHG  rewritten: outputs on non-detector frames, PSD Kalman, batched
│   └── flow.py                     NEW  vectorised LK median-flow propagation + per-track reliability
├── telemetry/metrics.py            CHG  per-frame records, end-to-end throughput, resources
├── evaluation/evaluator.py         NEW  mAP/P/R exactly matching ultralytics; subsets; paired bootstrap
├── experiments/
│   ├── presets.py                  NEW  every ablation as named config overrides
│   ├── run_ablations.py            NEW  run a suite, write summary.csv/.md
│   └── compare.py                  NEW  paired per-video bootstrap between two runs
└── shared/
    ├── config.py                   CHG  single PipelineConfig dataclass
    ├── boxes.py                    NEW  vectorised IoU / class-aware NMS
    └── __init__.py                 CHG  (old one imported from nonexistent `src.`)
```

## 3. Usage

```bash
# 0. sanity gate: must reproduce the frozen baseline (mAP50 .9843, mAP50-95 .9749, P .9779, R .9752)
python -m adaptive_pipeline.run --preset baseline --images DATA/test/images --out runs/test/baseline
#    if it does not: fix class_map (shared/config.py), --ap-protocol (current|legacy),
#    --set half=true (ultralytics val uses fp16 on CUDA), imgsz, --seq-regex. Do not proceed until it matches.

# 1. tune on VAL
python -m adaptive_pipeline.experiments.run_ablations --suite pareto     --images DATA/val/images --out runs/val
python -m adaptive_pipeline.experiments.run_ablations --suite components --images DATA/val/images --out runs/val

# 2. freeze chosen thresholds (edit PRESETS / defaults), then TEST once
python -m adaptive_pipeline.experiments.run_ablations --suite main --images DATA/test/images --out runs/test
python -m adaptive_pipeline.experiments.compare runs/test/fixed4_flow runs/test/adaptive_full

# single video, efficiency only / live paced demo
python -m adaptive_pipeline.run --preset adaptive_full --video clip.mp4 --out runs/clip
python -m adaptive_pipeline.run --preset adaptive_full --video clip.mp4 --realtime --out runs/live
```
Overrides: `--set k_max=12 novelty_thr=0.3 device=\"0\"`. Labels: YOLO txt via
`/images/`→`/labels/`, or `--labels`. Sequences: one subdirectory per video, or a
flat directory grouped by `--seq-regex` (default `<video>_<frame>`); the run prints
the discovered video/frame counts — check them against 122 / 43,772.

## 4. Outputs per run
`metrics.json` (config, detection overall / by decision / by staleness, efficiency),
`frames.csv` (one row per frame: decision, reasons, per-stage ms, GFLOPs, novelty,
fail_frac, luminance), `eval_stats.npz` (for bootstrap).

Efficiency fields: `detector_skip_rate` (frames without a full pass),
`local_redetect_rate`, `roi_pass_rate`, `roi_confirm_rate`, `tracking_only_rate`,
`gflops_fraction_of_baseline`, `latency_{mean,p50,p95,p99,max}_ms`, `fps_end_to_end`,
`full/local/skip_frame_ms_mean`, `overhead_ms_mean`, `cpu_ms_per_frame`,
`peak_rss_mb`, `peak_gpu_mem_mb`, `trigger_counts`.

## 5. Evaluation protocol
1. **Same conditions as baseline.** Same images, labels, class map, imgsz, conf=0.001,
   iou=0.7, max_det=300, precision, and AP protocol; baseline re-run in this harness
   (`baseline`) is the reference for every efficiency number. Never compare
   end-to-end ms to the frozen 7.97 ms (inference-only, different scope/hardware).
2. **Primary claim = Pareto comparison at matched compute.** Plot mAP50-95 vs.
   `gflops_fraction_of_baseline` for `fixed{2,3,4,6,8}_flow` and the `adaptive_nov*_k*`
   sweep. Adaptivity is justified only if the adaptive curve lies above the fixed
   curve. Confirm the gap at the chosen operating point with `compare.py`
   (CI must exclude 0).
3. **Tune on val, report test once.** All `[TUNE-ON-VAL]` fields.
4. **Report the breakdowns:** quality on SKIP frames and by staleness bin
   (where error comes from), trigger shares (which signal spends the compute),
   `roi_confirm_rate` (whether the LOCAL tier pays: it saves compute only if
   confirm rate > ROI-cost / full-cost, roughly 0.4–0.6).
5. **Two efficiency views:** hardware-independent (detector calls, GFLOPs) and
   wall-clock on the target device. Report CPU/edge and GPU separately.

## 6. Ablation suites (`experiments/presets.py`)
| suite | question |
|---|---|
| `sanity` | does the harness reproduce the frozen baseline? |
| `propagation` | hold vs. Kalman vs. flow at K = 2, 4, 8: is propagation worth its cost? |
| `main` | baseline, adaptive, fixed-K flow curve |
| `pareto` | adaptive threshold sweep vs. fixed-K at matched GFLOPs |
| `components` | scheduler-only → +track health → +LOCAL → +uncertainty; −novelty; Kalman-only |
| `illumination` | CLAHE on baseline and adaptive (needs human labels + night subset) |
| `refine` | original selective re-detection on keyframes (needs human labels) |

## 7. Known limitations
* **Labels.** Stock YOLOv8n scoring 0.975 mAP50-95 on BDD-style data is only
  plausible if labels are close to YOLOv8n's own outputs (in this repo's test,
  baseline vs. its own conf≥0.25 outputs scores 0.995). Then metrics measure
  *agreement with dense YOLOv8n*, which is valid for the skip question if named
  as such, but CLAHE and keyframe refinement can only lose on these labels even
  when they find real objects. Evaluate those on human labels (e.g. BDD100K's
  human-annotated tracking-set boxes, or a hand-labelled night subset).
* **Frame rate.** Propagation difficulty depends on the frame interval. Confirm the
  native fps of the derived sequences (~359 frames/video); at 5 fps far fewer
  frames can be skipped than at 30 fps.
* **New objects** can only be found by FULL passes; worst-case miss latency is
  `k_max` frames. Novelty is a proxy, not a detector.
* **GPU wall-clock.** Skip-frame overhead (signals + flow, ~5 ms on a weak CPU
  core) is comparable to an 8 ms GPU detector call; GFLOPs savings may not turn
  into GPU latency savings. The target where skipping pays most is CPU/edge.
* Propagated boxes are kept for at most `coast_max` frames after flow fails;
  thresholds are heuristic and must be tuned on val.
