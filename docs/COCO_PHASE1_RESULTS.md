# Phase 1: verified COCO evaluation of Adaptive Vision

Measured 19 October 2026. Scope: score existing saved predictions only. No YOLO inference, adaptive replay, training, new pseudo-labels or algorithm changes. Historical results remain intact.

The standard COCO evaluation changes AP by only +0.237 to +0.251 percentage points. It does not explain the low absolute accuracy or make original adaptive better than Fixed-2. Use these new COCO results as the primary evaluation reference for this exact subset; keep historical custom metrics labelled as such.

## Evaluation population and protocol

All five methods use the same 3,416 images in five sequences (901, 901, 469, 475, 670 frames), 1920×1200. There are 57,711 annotations: 55,412 regular objects and 2,299 crowds. No ignore-only annotations or entirely empty annotation frames exist in this export. No frames or classes were dropped. Synthetic tests cover empty frames and absent classes.

Scoring uses pinned `pycocotools==2.0.11` bbox COCOeval with standard defaults: category-aware evaluation, IoU 0.50 through 0.95 in increments of 0.05, 101 recall samples, maxDets [1,10,100], all image IDs and all eight provided categories. Primary AP uses the cap of 100 detections per image **per category**. AP is pooled over images and then averaged over supported classes and IoUs; it is not the mean of sequence APs. Undefined class/area cells are omitted by COCO, not assigned zero. Zero-valued supported-class results are retained.

The [official evaluator source](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) defines matching, accumulation, crowd handling and metric defaults. The installed [maintained pycocotools distribution](https://pypi.org/project/pycocotools/2.0.11/) is pinned; runtime Python, NumPy, Python-source and native-extension hashes are recorded. These are COCO-protocol bbox metrics on an Argoverse-HD subset, not an official Argoverse challenge submission, tracking metric or streaming AP.

Ground-truth area and prediction coordinates remain in original-image pixels. The original COCO area ranges are used: [0,32²], [32²,96²], [96²,10¹⁰], with the source implementation's inclusive boundaries (boundary areas can belong to both adjacent ranges). These are not sizes after resizing to 640.

Saved predictions are Nx6 original-pixel xyxy, confidence, mapped class. Conversion only changes xyxy to xywh and maps the saved class name back to the annotation category ID. No confidence cutoff, coordinate clipping, rounding, re-NMS or output repair is applied. Eight GT boxes extending one pixel beyond the right edge remain unchanged. The Dense zero-height prediction remains present and is verified to receive no TP or ignore credit whenever considered.

Stock COCO model indices [0,1,2,3,5,7,9,11] correspond to person, bicycle, car, motorcycle, bus, truck, traffic_light, stop_sign; dataset IDs are [0,1,2,3,4,5,6,7]. The saved configurations, named categories, frame/target/ignore hashes and per-row prediction classes were checked independently across all methods. No mapping inconsistency was found.

Source inference settings were identical: stock YOLOv8n checkpoint, CPU FP32, imgsz 640, detector confidence 0.001, NMS IoU 0.7 and original global detector max_det 300. They are historical settings, not executions in Phase 1. Standard evaluation's per-category cap100 is distinct from that original inference cap300. Saved outputs cannot recover detections already removed by inference, class filtering or tracking.

## Verified aggregate accuracy

All entries below are percentages except the delta, which is percentage points. AP means COCO AP50–95. AR100 averages recall over the ten IoU thresholds and categories; it is not recall at a selected confidence threshold.

| Method | Old custom AP | COCO AP | Δ pp | COCO AP50 | COCO AP75 | COCO AR100 |
| --- | --- | --- | --- | --- | --- | --- |
| Dense YOLOv8n | 15.50 | 15.73 | +0.237 | 23.87 | 15.65 | 23.19 |
| Fixed-2 | 13.77 | 14.02 | +0.248 | 20.81 | 14.00 | 18.09 |
| Original adaptive | 13.74 | 13.98 | +0.243 | 20.70 | 14.06 | 18.13 |
| Adaptive + anchoring | 13.92 | 14.16 | +0.241 | 20.69 | 14.48 | 18.26 |
| Adaptive + quality-aware flow | 13.96 | 14.21 | +0.251 | 20.69 | 14.45 | 18.38 |

The AP ordering remains Dense > quality-aware > anchoring > Fixed-2 > original adaptive. At AP50, Fixed-2 remains best among sparse methods. Tiny differences must not be interpreted as evidence of statistical significance.

## Object size and class results

| Method | AP small | AP medium | AP large | AR small | AR medium | AR large |
| --- | --- | --- | --- | --- | --- | --- |
| Dense YOLOv8n | 1.01 | 15.40 | 53.46 | 3.35 | 27.13 | 67.79 |
| Fixed-2 | 0.73 | 12.58 | 47.89 | 1.88 | 18.67 | 57.20 |
| Original adaptive | 0.67 | 12.73 | 47.68 | 1.78 | 19.16 | 57.13 |
| Adaptive + anchoring | 0.67 | 13.23 | 49.41 | 1.76 | 19.48 | 58.79 |
| Adaptive + quality-aware flow | 0.72 | 13.24 | 49.68 | 1.88 | 19.18 | 58.97 |

Dense AP is 1.01% for small objects versus 53.46% for large objects under this protocol. This confirms a pronounced size-associated weakness in the saved baseline. These scores alone do not establish whether resolution, domain shift, confidence filtering or another detector property causes it.

| Class | Dense YOLOv8n | Fixed-2 | Original adaptive | Adaptive + anchoring | Adaptive + quality-aware flow |
| --- | --- | --- | --- | --- | --- |
| person | 16.41 | 13.99 | 14.20 | 14.59 | 14.72 |
| bicycle | 0.04 | 0.02 | 0.04 | 0.03 | 0.04 |
| car | 28.25 | 25.17 | 25.29 | 25.63 | 25.84 |
| motorcycle | 4.90 | 2.56 | 2.45 | 2.57 | 2.56 |
| bus | 54.90 | 52.60 | 52.55 | 52.55 | 52.55 |
| truck | 9.18 | 7.44 | 7.52 | 7.56 | 7.75 |
| traffic_light | 6.11 | 4.91 | 4.92 | 5.40 | 5.39 |
| stop_sign | 6.08 | 5.47 | 4.88 | 4.99 | 4.87 |

Class-macro AP gives every supported class equal weight; cars do not dominate it merely because they have the most annotations. Full AP50 and AR per class are in each new metrics.json.

## Per-sequence COCO results

Each sequence was evaluated independently with COCOeval, rather than slicing accumulated aggregate results. Categories without regular GT in a sequence are undefined and excluded by the official evaluator, so sequence APs have different class support.

| Sequence | Frames | Dense YOLOv8n | Fixed-2 | Original adaptive | Adaptive + anchoring | Adaptive + quality-aware flow |
| --- | --- | --- | --- | --- | --- | --- |
| sequence_00 | 901 | 7.39 | 6.47 | 6.48 | 6.64 | 6.74 |
| sequence_01 | 901 | 9.73 | 8.13 | 7.99 | 8.28 | 8.50 |
| sequence_02 | 469 | 14.02 | 12.62 | 12.62 | 12.78 | 12.80 |
| sequence_03 | 475 | 29.16 | 23.64 | 24.44 | 25.65 | 25.84 |
| sequence_04 | 670 | 24.66 | 22.88 | 22.88 | 22.88 | 22.88 |

Above: AP50–95 (%). Full per-sequence AP50/AP75, area AP, AR and class metrics are retained in machine-readable metrics.json.

## What differed from the historical evaluator

1. **AP integration is different.** [Historical _compute_ap](../extracted/adaptive_pipeline/evaluation/evaluator.py:96) interpolates precision onto a recall grid and integrates with a trapezoidal rule. COCO samples the precision envelope at the first achieved recall at or above each recall threshold, then takes its arithmetic mean. A synthetic perfect detection gives 0.995 historically and 1.0 officially. This test illustrates non-equivalence; it is not a universal correction factor.
2. **Regular matching is different.** [Historical match_predictions](../extracted/adaptive_pipeline/evaluation/evaluator.py:36) sorts candidate pairs by IoU and deduplicates prediction then GT indices. COCO processes predictions by descending score and searches remaining eligible GT, with ignored GT ordered last. The old matcher can discard an alternative valid assignment. A two-GT synthetic case verifies one historical TP versus two official TPs. Real-data flags change, especially at IoU50; neither matcher is assumed globally optimal.
3. **Tie ordering and caps differ.** [Historical ap_per_class](../extracted/adaptive_pipeline/evaluation/evaluator.py:109) uses NumPy's default unstable argsort and lacks COCO's image/category cap100. COCO uses stable score ordering. Original row order is retained in exports, and image ordering is explicitly checked. Auxiliary cap300 scoring below isolates the cap's practical size using the same official rules.
4. **Crowd handling is similar in overlap convention but not an identical complete matcher.** [Historical match_ignored](../extracted/adaptive_pipeline/evaluation/evaluator.py:50) uses crowd intersection divided by prediction area, permits multiple ignored predictions, and applies ignore after historical regular matching. COCO handles regular and crowd matching together, prioritizing nonignored eligible GT. The changes in ignore flags are small here and do not support blaming a wholesale crowd-policy error.
5. **Historical P/R/F1 are different estimands.** [Historical result computation](../extracted/adaptive_pipeline/evaluation/evaluator.py:180) reports class-macro IoU50 P/R at a method-specific confidence selected from smoothed mean F1 curves. Its 24.31% Dense recall is not COCO AR100 (23.19%). Comparing those as a before/after recall correction would be invalid. The old approximate confidence-0.25 values also interpolate on a 1000-point grid (selected grid value 0.250250...), unlike the exact threshold counts below.
6. **Annotation ID 0 is a COCO import hazard, not an explanation for old custom AP.** The export has a GT annotation with ID0. Official dtMatches uses 0 for unmatched, and a direct unnormalized perfect synthetic ID0 case scores AP0. The [new GT converter](../extracted/adaptive_pipeline/evaluation/coco.py:30) assigns positive IDs in a deep copy, retains source_annotation_id and records the complete mapping. Original annotation IDs and files remain unchanged. The historical array-based matcher does not use those IDs.
7. **Ignore-only and fully empty inputs need explicit handling.** COCO bbox preparation overwrites generic ignore with iscrowd; this dataset has no ignore-only objects. The new scorer rejects such future inputs rather than silently applying an unsupported policy. COCO loadRes dereferences the first detection; the wrapper handles a valid fully empty prediction set explicitly. Crowds are excluded from regular GT denominators, and crowd-only/empty-frame predictions still pass through official matching.

The taxonomy and numeric conversion checks found no evidence of a saved-prediction class-index or xyxy/xywh mismatch. This is not a new inference-level audit of letterboxing or NMS: saved-output scoring cannot independently reconstruct preprocessing or recover discarded predictions.

### Measured discrepancy decomposition

The following is a **sequential, order-dependent diagnostic**, not alternative official scores: A = historical AP; B = same historical TP/ignore flags and original quicksort ordering with COCO-style recall accumulation; C = same flags with stable ordering plus cap100; D = full official matching and accumulation. B/C hold old assignments fixed even after a cap, so they must not be used as benchmark metrics. D−C includes official rematching, ignore and numerical IoU semantics; it is not a pure crowd or matching ablation.

| Method | A old AP % | B accumulation % | C stable/cap % | D official % | B−A pp | C−B pp | D−C pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense YOLOv8n | 15.495414 | 15.718730 | 15.715829 | 15.732713 | +0.223316 | -0.002901 | +0.016884 |
| Fixed-2 | 13.771941 | 14.011509 | 14.011357 | 14.020016 | +0.239567 | -0.000152 | +0.008660 |
| Original adaptive | 13.738511 | 13.972195 | 13.970857 | 13.981381 | +0.233684 | -0.001338 | +0.010523 |
| Adaptive + anchoring | 13.923108 | 14.155645 | 14.154266 | 14.164180 | +0.232537 | -0.001379 | +0.009914 |
| Adaptive + quality-aware flow | 13.961943 | 14.204334 | 14.200924 | 14.212644 | +0.242391 | -0.003410 | +0.011720 |

Most of the AP increase in this diagnostic path comes from accumulation, not extra credited objects. For Dense, official matching changes 382 considered TP flags at IoU50, with a net gain of 210 TPs (30,039→30,249), and changes five ignore flags. At IoU75 through IoU95, considered TP/ignore flags agree exactly. Full per-IoU counts for every method are saved. TP counts alone cannot reproduce AP, which also depends on score ordering.

### Cap sensitivity: auxiliary, nonstandard primary AP

| Method | Saved predictions | Rows beyond cap100 | AP100 % | AP300 % | AP300−AP100 pp |
| --- | --- | --- | --- | --- | --- |
| Dense YOLOv8n | 601389 | 21612 | 15.732713 | 15.736849 | +0.004137 |
| Fixed-2 | 311655 | 10913 | 14.020016 | 14.020143 | +0.000127 |
| Original adaptive | 314644 | 10951 | 13.981381 | 13.982668 | +0.001287 |
| Adaptive + anchoring | 314094 | 10971 | 14.164180 | 14.165531 | +0.001351 |
| Adaptive + quality-aware flow | 323789 | 11531 | 14.212644 | 14.217597 | +0.004953 |

Using official matching with a cap of 300 raises AP by less than 0.005 percentage points for every method. The standard cap100 does not explain the low AP in these outputs. AP300 is explicitly auxiliary; COCOeval.summarize assumes cap100 for its primary AP slot, so the auxiliary result is read directly from the official tensors instead of mislabelling summarize output.

## Supplementary fixed-confidence precision, recall and F1

These use official all-area matching, cap100, IoU50 and exact score ≥0.25. They are supplementary operating-point statistics, not official COCO summary slots. Macro is the mean of each class statistic; macro F1 is not the harmonic mean of macro P and R. Micro pools TP/FP/GT counts. Regular targets =55,412 for all methods.

| Method | Macro P % | Macro R % | Macro F1 % | Micro P % | Micro R % | Micro F1 % | TP | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense YOLOv8n | 62.46 | 18.08 | 25.00 | 76.80 | 28.31 | 41.37 | 15688 | 4740 |
| Fixed-2 | 64.99 | 17.54 | 24.35 | 77.80 | 27.65 | 40.80 | 15320 | 4371 |
| Original adaptive | 60.82 | 17.64 | 24.51 | 77.99 | 27.72 | 40.91 | 15362 | 4336 |
| Adaptive + anchoring | 61.70 | 17.68 | 24.56 | 77.80 | 27.74 | 40.89 | 15369 | 4386 |
| Adaptive + quality-aware flow | 61.88 | 17.54 | 24.34 | 77.85 | 27.69 | 40.86 | 15346 | 4366 |

No confidence was optimized to obtain these values. The same 0.25 threshold is used for every method. Historical best-F1 P/R remain in previous_custom, clearly separated from the new metrics.

## Efficiency and implications for the next phase

The following are original benchmark measurements copied unchanged; Phase 1 only measures accuracy by rescoring saved outputs. FPS is historical pipeline processing throughput, excluding image decoding, scoring, loading, warmup, hashing and output I/O. Latencies retain that same scope. They are single-run timings, not a new controlled speed comparison.

| Method | Detector calls | Call savings % | Historical FPS | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Dense YOLOv8n | 3416 | 0.00 | 45.21 | 22.12 | 25.38 |
| Fixed-2 | 1710 | 49.94 | 70.63 | 14.16 | 25.41 |
| Original adaptive | 1733 | 49.27 | 66.62 | 15.01 | 27.81 |
| Adaptive + anchoring | 1729 | 49.39 | 66.24 | 15.10 | 27.76 |
| Adaptive + quality-aware flow | 1789 | 47.63 | 67.25 | 14.87 | 25.32 |

- **Confirmed:** evaluation conventions caused a small numerical discrepancy, not the baseline's large accuracy deficit. Dense remains 15.73% AP. It already performs poorly on small objects and certain classes before any SKIP propagation.
- **Confirmed on this subset:** original adaptive is 0.0386 AP percentage points below Fixed-2 while using 23 more detector calls. These aggregate AP/call metrics do not demonstrate a better scheduler trade-off.
- **Confirmed descriptively:** quality-aware adaptive is +0.2313 AP points over original adaptive, +0.1926 over Fixed-2 and −1.5201 versus Dense. It uses 56 more detector calls than original adaptive and 79 more than Fixed-2. Quality-aware versus anchoring adds only +0.0485 AP points with 60 more calls. These are different compute budgets, not isolated equal-budget causal improvements.
- **Supported priority:** retain stock YOLOv8n and freeze this scoring protocol before further work. Next-phase diagnosis should separate Dense misses from SKIP propagation losses on paired frames, stratified by object size and time since FULL. Official matched flags and lossless COCO exports are saved for that work. No new scheduling, tracking or detection improvement was implemented or tested here.
- **Unresolved:** these pooled scores do not allocate the adaptive gap among confidence deletion, missing births, classification, localisation or scheduling decisions. Five sequences and small AP differences do not establish statistical significance or generalization. Original inference's global top300 and prior filtering cannot be undone by scoring. A separately approved diagnostic or experiment is needed before claiming any algorithmic remedy.

## Validation and reproducibility

New implementation: [coco.py](../extracted/adaptive_pipeline/evaluation/coco.py), [score_saved_coco.py](../extracted/adaptive_pipeline/evaluation/score_saved_coco.py), [pinned evaluation dependency](../requirements-evaluation.txt), [13 synthetic tests](../tests/test_coco_scoring.py) and [independent serialized-export verifier](../analysis/argoverse_hd_coco_phase1/verify_exports.py). This report is new. Existing pipeline/evaluator files and old reports were not edited by Phase 1; pre-existing dirty work was preserved.

Validation outcomes:

- 68 tests passed, including 13 new synthetic protocol/conversion tests: positive GT IDs, taxonomy conversion, crowd IoA/many-to-one ignores, empty frames and results, degenerate/outside boxes, invalid records, per-category caps and stable ties, alternate-GT matching, perfect-detection AP and independent sequence/area scoring.
- Preflight validated all five frame key sets, order, source settings/checkpoint hashes, per-frame target/ignore hashes and prediction-row alignment. Historical AP50/AP/P/R reproduced to absolute tolerance 1e-14 for every method. Dataset annotations/manifest hashes agree with the original benchmark audit.
- Every standard summary was checked against COCOeval's 12 official stats; independent 101-recall accumulation using official flags reproduced AP/AP50 to tolerance 1e-14. Independent cap masks exactly matched considered official predictions. Per-sequence metrics were scored with separate COCOeval instances.
- All source annotation/manifest and saved benchmark evidence hashes were unchanged after scoring. No torch or ultralytics import occurred during the scorer. All 35 Python files in the original benchmark snapshot remain byte-identical to their current counterparts, including the historical evaluator and detection/tracking/scheduling code.
- The separate export verifier passed: every serialized annotation agrees with its source except documented ID normalization, all 1,865,571 prediction rows across five methods roundtrip losslessly, and reloading Dense's gzip JSON directly through COCO (bypassing new adapter/summary helpers) reproduces the complete precision/recall/score tensors exactly. Completion evidence is stored separately.

Commands executed from the repository root:

```sh
.venv/bin/python -m pip install --no-cache-dir --no-deps --only-binary=:all: pycocotools==2.0.11
PYTHONPATH=extracted .venv/bin/python -m unittest discover -s tests -q
PYTHONPATH=extracted .venv/bin/python -m adaptive_pipeline.evaluation.score_saved_coco --dataset data/argoverse_hd --benchmark runs/argoverse_hd_benchmark_v1 --validate-only --out runs/argoverse_hd_coco_phase1_preflight_v1
PYTHONPATH=extracted .venv/bin/python -m adaptive_pipeline.evaluation.score_saved_coco --dataset data/argoverse_hd --benchmark runs/argoverse_hd_benchmark_v1 --out runs/argoverse_hd_coco_phase1_v1
.venv/bin/python analysis/argoverse_hd_coco_phase1/verify_exports.py --scoring runs/argoverse_hd_coco_phase1_v1 --dataset data/argoverse_hd --benchmark runs/argoverse_hd_benchmark_v1
```

Reproduction must choose a fresh output path: the scorer refuses an existing directory or a directory inside the original benchmark/dataset. Install `requirements-evaluation.txt` into the existing project environment (which already supplies NumPy); no model dependency is imported by scoring. Full recovered source configurations are in evidence.json; the exact scoring invocation is in execution_plan.json. runtime.json captures the executed package environment. Original checkpoint SHA256 and input-manifest SHA256 are also retained.

Artifacts: [new summary CSV](../runs/argoverse_hd_coco_phase1_v1/summary.csv), [completion record](../runs/argoverse_hd_coco_phase1_v1/COMPLETE.json), [input/protocol evidence](../runs/argoverse_hd_coco_phase1_v1/evidence.json), [independent export verification](../runs/argoverse_hd_coco_phase1_v1/independent_export_verification.json). Each method folder contains normalized COCO predictions, full metrics/decomposition, per-sequence official logs, precision/recall/score tensors and row-aligned official TP/ignore/cap masks. The normalized annotation copy and annotation-ID map are retained at the scoring root. scorer_sources contains the executed scoring, historical evaluator, package and test source snapshots.

Historical reports [ARGOVERSE_HD_RESULTS.md](ARGOVERSE_HD_RESULTS.md) and [ARGOVERSE_HD_DIAGNOSIS.md](ARGOVERSE_HD_DIAGNOSIS.md) remain historical evidence. For this subset, use this Phase 1 report's standard COCO metrics when assessing future work; do not compare them directly to previous pseudo-label scores or streaming AP. Phase 2 has not started.
