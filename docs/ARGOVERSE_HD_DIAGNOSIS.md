# Adaptive Vision: diagnosis of the five-sequence Argoverse-HD benchmark

Research date: 19 October 2026. **No new YOLO inference, tracker/pipeline replay, AP evaluation, benchmark, training, or algorithm changes were performed.** This investigation reads the existing annotations, predictions, saved TP/ignore flags and telemetry. New diagnostic scripts/results are isolated in [analysis/argoverse_hd_root_cause_v1](../analysis/argoverse_hd_root_cause_v1). The benchmark and original dataset remain unchanged.

The main finding is that **weak baseline detection and a mismatch between detection and track-admission confidence are more important here than wholesale optical-flow failure**. Anchoring helps localisation of objects that actually become tracks. Quality-aware flow changes accepted boxes very little; its extra accuracy mostly accompanies a different detection schedule. The legacy scheduler does not demonstrate an advantage over Fixed-2 at matched cost.

## Evidence and interpretation

Three read-only scripts produced [saved-output summaries](../analysis/argoverse_hd_root_cause_v1/saved_output_diagnostics.json), [focused history/geometry summaries](../analysis/argoverse_hd_root_cause_v1/focused_diagnostics.json), and [scheduler/FP summaries](../analysis/argoverse_hd_root_cause_v1/scheduler_and_fp_diagnostics.json). Their ordered frame/target/prediction counts were checked against the existing archives. Current pipeline Python sources match the benchmark's source snapshot.

Current annotation and manifest SHA-256 hashes also match the benchmark audit. An [evidence manifest](../analysis/argoverse_hd_root_cause_v1/evidence_manifest.json) records hashes of the archived metrics, predictions, matching flags and telemetry used here.

Two kinds of numerical evidence are kept distinct:

- **Saved-match counts:** sums/slices of the original `eval_stats.npz` TP and ignored flags. No boxes are rematched and no new AP is calculated. Micro recall below means these TP counts divided by regular target count, at a stated IoU and confidence.
- **Box coverage/history diagnostics:** a target has at least one same-class saved box at a specified IoU/confidence. These allow many-to-one overlaps and can overcount crowded targets; they are **not AP or official one-to-one recall**. For example, Dense has 31,088 targets covered at IoU≥0.50, versus 30,166 saved one-to-one TPs. Candidate loss attribution is not a reconstruction of internal tracker identities.

The annotations provide track IDs, but the [authors' format documentation](https://github.com/mtli/sAP/blob/master/doc/data_setup.md) warns that tracks can fragment across annotation segments and linking can be noisy. History-based findings therefore refer to provided annotation identities, not independently verified physical identities. No class-changing annotation track IDs were found in this subset. Regular labels contain only six bicycle track IDs and two stop-sign track IDs, despite hundreds of frame instances; frames are not independent samples.

## Ranked causes and scientific priorities

| Priority | Finding and evidence level | Mechanism / expected impact | Smallest unresolved diagnostic |
|---|---|---|---|
| Measurement prerequisite | **Confirmed:** repository AP is not official COCO AP. No class-map or scale-conversion failure was found. | May change absolute scores and close rankings; changing the metric itself does not improve detections. | Official COCO rescoring of the five existing prediction archives, with approval. |
| 1 | **Confirmed size/confidence association; strong support for baseline limitation:** native small-target coverage at IoU50 is 24.42%, large-target coverage 97.12%. | Affects every frame and all methods. Low resolution, tiny objects, classification and confidence ranking limit the stock detector. Exact causal contributions are unmeasured. | Same-stock-model resolution diagnostic; retain raw class/cap information. |
| 2 | **Confirmed mechanism; strongly supported contribution:** 6,239/7,266 Dense-only SKIP coverage cases had only a correct box below 0.25 at the last FULL. | Track admission/confirmation discards useful low-score evidence before flow can help. Recovering some may improve adaptive recall but increases FP and tracking cost. | Controlled cached-detection replay separating existing-track recovery from lower-threshold births. |
| 3 | **Confirmed:** confidence/age refreshes dominate; about 95% of SKIPs have age one. Original adaptive uses 23 more calls than Fixed-2 and has 0.033 pp less custom AP. | Heuristic refreshes mostly change the phase of a near-Fixed-2 schedule. Better allocation could improve savings/accuracy, but is not established. | Compare on an identical schedule first; later compare schedulers at matched calls. |
| 4 | **Confirmed secondary limitation:** propagated geometry loses high-IoU matches; anchoring improves matches on shared SKIP frames. | Improves localisation of already-trackable objects. It cannot recover objects never admitted to the tracker or correct their class. | Instrument reliable-seed box/point trajectories; isolate anchoring and quality gates under fixed scheduling. |
| Lower priority | **Hypotheses:** camera compensation, appearance verification, long-occlusion recovery, new-object discovery. | May matter in selected scenes/longer gaps; current evidence does not identify them as dominant. | Targeted diagnostics after the higher-priority causes, rather than enabling all optional features. |

## 1. Evaluation and integration audit

### Taxonomy, inputs and coordinates

The actual map is correct: YOLO COCO indices `0,1,2,3,5,7,9,11` map respectively to provided category IDs `0,1,2,3,4,5,6,7`: person, bicycle, car, motorcycle, bus, truck, traffic_light, stop_sign. The old seven-class mapping is overridden in the Argoverse runner. Names are checked against checkpoint names before the benchmark; the detector's LUT maps once, class-aware tracker association/NMS retain those IDs, and the evaluator receives the dataset's eight names. References: [dataset mapping](../extracted/adaptive_pipeline/evaluation/argoverse.py:16), [configuration recovery](../extracted/adaptive_pipeline/experiments/run_argoverse.py:28), [checkpoint taxonomy assertion](../extracted/adaptive_pipeline/experiments/run_argoverse.py:87), [prediction LUT](../extracted/adaptive_pipeline/detection/detector.py:45), [association class check](../extracted/adaptive_pipeline/tracking/tracker.py:155).

Annotations are native pixel xywh, converted once to class+xyxy, not normalized YOLO text labels. Dimensions and area agree with 1920×1200 images. Eight one-pixel right-edge overflows are retained explicitly; one zero-height Dense prediction is retained as an FP. These edge cases do not explain the broad deficit. [Native conversion](../extracted/adaptive_pipeline/evaluation/argoverse.py:73), [target routing](../extracted/adaptive_pipeline/run.py:85), [continuous-coordinate IoU](../extracted/adaptive_pipeline/shared/boxes.py:8).

Installed Ultralytics preprocessing letterboxes the BGR image, converts to RGB/float and divides by 255. Detection results use `scale_boxes` to remove padding, divide by resize gain and clip to original dimensions; the wrapper reads original-coordinate `b.xyxy` without scaling it again. With this image shape and `imgsz=640`, image content is 640×400 inside the stride-padded 640×416 input, with eight vertical padding pixels on each side. **No double scaling, wrong color order or missing padding restoration was found.** Sources: [preprocess](../.venv/lib/python3.14/site-packages/ultralytics/engine/predictor.py:162), [letterbox](../.venv/lib/python3.14/site-packages/ultralytics/engine/predictor.py:203), [restore coordinates](../.venv/lib/python3.14/site-packages/ultralytics/models/yolo/detect/predict.py:109), [gain/padding implementation](../.venv/lib/python3.14/site-packages/ultralytics/utils/ops.py:122). This is a code/metadata/qualitative audit, not a new inference round-trip test.

Manifest image IDs, frame indices and sequence metadata are cross-checked; indices are contiguous and timestamps monotonic. `pipe.reset()` runs at every sequence boundary. All 3,416 frames were scored. All eight categories have regular targets globally; there are no unsupported annotation classes or entirely empty annotation frames in this subset. Empty/crowd-only frames are nevertheless represented by the loader. [Ordering validation](../extracted/adaptive_pipeline/evaluation/argoverse.py:118), [reset](../extracted/adaptive_pipeline/run.py:67), [saved benchmark verification](../runs/argoverse_hd_benchmark_v1/verification.json).

### A real limitation in the filtering audit

The detector uses confidence 0.001, class-aware NMS IoU 0.7, maximum 300 detections and default best-class prediction. It filters to the eight classes **after** Ultralytics' NMS and global max-detection truncation. Thus unsupported COCO classes can occupy slots before the LUT discards them. Supported-box counts average 176/frame and never equal 300, but this **does not prove that the original 80-class output avoided the 300-box cap**. Raw outputs before mapping and alternative class scores were not saved. Whether useful low-confidence detections were removed by the cap, NMS or best-class selection remains unknown. This is a plausible additional baseline limitation, not a demonstrated cause. [Detector arguments](../extracted/adaptive_pipeline/detection/detector.py:37), [post-NMS LUT](../extracted/adaptive_pipeline/detection/detector.py:53), [best-class filtering/cap](../.venv/lib/python3.14/site-packages/ultralytics/utils/nms.py:116).

### Repository AP versus official COCO AP

| Component | Repository | Official COCO |
|---|---|---|
| Regular matching | IoU-priority candidate deduplication, inherited from Ultralytics | Score-ordered detections choose available GT; regular GT preferred to crowds |
| AP accumulation | Trapezoidal integration of linearly interpolated precision envelope | Mean precision sampled at 101 recall thresholds |
| Detection limit | Model caps 300 globally over 80 classes before mapping; evaluator keeps saved outputs | Default main AP uses up to 100 detections per image/category |
| Confidence ties | Default NumPy argsort | Stable score sorting |
| Size metrics | No official area-range AP implementation | Area-range ignore rules for AP_S/M/L |

These distinctions are explicit in [local matching/AP](../extracted/adaptive_pipeline/evaluation/evaluator.py:36) and the [official COCO implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py). The exact score/rank effect on these saved predictions is **not measured**; official rescoring requires approval. The small adaptive deltas should not be treated as metric-independent findings. A synthetic limiting example illustrates non-equivalence: this trapezoidal formulation gives 0.995 for an otherwise perfect single-class precision–recall curve, while standard COCO gives 1.0.

Crowd intersection/prediction-area and multiple-match semantics agree with the [COCO overlap convention](https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c). The repository first fixes regular matches and then ignores unmatched same-class predictions separately at every IoU threshold. That is an extension to its existing matcher, not the complete COCO matching protocol. There are 2,299 crowd annotations and no explicit `ignore=true` labels here; Dense has 8,461 ignored predictions at IoU50. No evidence of a missing-crowd or wrong-denominator failure was found. [Ignore implementation](../extracted/adaptive_pipeline/evaluation/evaluator.py:50).

The saved **24.31% recall is class-macro recall at the chosen smoothed best-F1 confidence (~0.109)**. It is not maximum recall, micro recall or COCO AR. Saved Dense TP flags yield:

| Confidence cut | Saved TP at IoU50 | Micro recall | Macro recall |
|---|---:|---:|---:|
| 0.001 | 30,166 | 54.44% | 37.72% |
| 0.10 | 20,894 | 37.71% | 24.89% |
| 0.25 | 15,664 | 28.27% | 18.06% |

The tiny difference between the 0.25 row and the report's interpolated fixed-confidence recall reflects its 1,000-point confidence grid, not missing labels. [P/R aggregation](../extracted/adaptive_pipeline/evaluation/evaluator.py:109).

## 2. Why Dense accuracy is low

### Small objects and localisation are demonstrable problems

Of 55,412 regular target instances, 23,790 (42.93%) have native area below 32² pixels, 20,159 are medium and 11,463 large. After the actual one-third linear resize, 38.79% have a minimum side below eight pixels; 79.31% have area below 32² input pixels. Median input width/height are 11.67/13.67 pixels. These are descriptive size statistics, not official COCO AP_S assignments at resized coordinates.

| Native size group | Instances | Same-class coverage IoU50, conf≥0.001 | Coverage IoU75, conf≥0.001 | Coverage IoU50, conf≥0.25 |
|---|---:|---:|---:|---:|
| Small | 23,790 | 24.42% | 2.93% | 2.47% |
| Medium | 20,159 | 70.17% | 30.61% | 34.59% |
| Large | 11,463 | 97.12% | 82.44% | 77.12% |

This is strong evidence that the weak Dense result is not merely an adaptive tracking issue. It supports testing resolution, but does not prove that resolution alone causes the failures: size covaries with class, distance, blur, visibility and lighting. Stock YOLO's smallest detection head has stride eight; this is context, not a hard lower object-size limit. The [original Argoverse-HD paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470460.pdf), Fig. 5, also reports a large small-object deficit relative to large objects, using a different detector. Its numerical AP is not a reference score for this model/subset/protocol.

Saved Dense matches fall from 30,166 at IoU50 to 16,302 at IoU75 and 3,354 at IoU95. Among overlap diagnostics, 14,679 targets have a same-class candidate with IoU in [0.10,0.50), and 4,910 lack any saved class box at IoU≥0.10. These diagnostic sets are not additive AP error categories. They indicate both missing useful detections and imprecise boxes; simply adjusting the scheduler cannot fix either on FULL frames.

### Confidence and class failures matter independently of geometry

The 601,389 Dense outputs have median confidence 0.00440. At confidence≥0.001 the saved flags contain 30,166 TP50, 562,762 FP50 and 8,461 ignored predictions. At ≥0.25 they contain 15,664 TP50 and 4,763 FP50. Raising the threshold suppresses false positives but also removes almost half the retained correct low-threshold matches. AP depends on their ranking, so merely increasing the evaluation threshold is not an accuracy fix.

| Class | Targets | Saved TP50 at ≥0.001 | Saved TP50 at ≥0.25 | Reported custom AP50–95 |
|---|---:|---:|---:|---:|
| Person | 7,093 | 3,618 | 1,504 | 16.00% |
| Bicycle | 687 | 25 | 0 | 0.03% |
| Car | 28,979 | 18,767 | 10,459 | 28.01% |
| Motorcycle | 332 | 79 | 10 | 4.71% |
| Bus | 4,234 | 3,037 | 2,347 | 54.95% |
| Truck | 3,667 | 1,424 | 579 | 8.75% |
| Traffic light | 10,097 | 3,162 | 747 | 5.75% |
| Stop sign | 323 | 54 | 18 | 5.76% |

All eight APs have equal macro weight; excellent bus detection does not compensate for bicycle/traffic-light failures. **No correctly overlapping bicycle detection reaches the tracker admission threshold.** Better flow therefore has almost no correct bicycle seeds to work with.

There are 3,433 targets without a same-class IoU50 box but with a different-class IoU50 box. Examples include 1,177 truck→car, 782 bus→car, 169 bicycle→person and 145 motorcycle→car nearest-overlap candidates. These suggest classification/semantic errors, but overlap with another real object can mimic confusion; they are not a formal confusion matrix. The [dataset authors](https://www.cs.cmu.edu/~mengtial/proj/streaming/) specifically document pickup-truck versus car ambiguity. The local export lacks their extra pickup-truck attribute. A small manual review is justified; arbitrary class merging/remapping would change the benchmark and is not recommended.

A dark representative frame exists in sequence_01, but sequence_00 has even lower Dense AP. No day/night labels or controlled enhancement study exist here. Low light or CLAHE should not be declared the main explanation. The earlier pseudo-label benchmark measures teacher agreement and cannot establish the detector's real annotation-based accuracy; the old label-builder threshold remains unrecovered.

## 3. Separate baseline misses from SKIP losses

On the exact 1,627 frames skipped by quality-aware adaptive, there are 26,635 regular targets. Using same-class IoU50 box coverage:

| Saved-box outcome | Target instances | Share |
|---|---:|---:|
| Dense and adaptive both cover | 7,518 | 28.23% |
| Neither covers | 11,753 | 44.13% |
| Dense covers, adaptive misses | 7,266 | 27.28% |
| Adaptive covers, Dense misses | 98 | 0.37% |

Thus a substantial problem already exists in Dense output, and another substantial loss is introduced on SKIP. These categories do not partition AP, which depends nonlinearly on scores, classes and one-to-one matching. Dense is a useful reference, **not a mathematical ceiling**: propagation sometimes covers targets that Dense misses.

The 7,266 Dense-only cases have the following candidate attribution, based on the corresponding annotation identity at the previous FULL:

| Previous FULL evidence | Cases | Interpretation |
|---|---:|---|
| Correct IoU50 box exists, but its maximum confidence is below 0.25 | 6,239 (85.87%) | Strong evidence for admission/confirmation bottleneck |
| No correct IoU50 box | 959 | Previous detector localisation/classification/visibility or temporal variation; not specifically a flow error |
| Correct IoU50 box at confidence≥0.25 | 41 | Seed existed; association, propagation or visibility needs individual tracing |
| Annotation identity absent at the last FULL | 27 | Possible newly visible object or annotation fragmentation |

The tracker first filters FULL detections at `track_min_conf=0.25`. With low-score association disabled, an existing track is not confirmed by a correct lower-score detection and becomes invisible immediately (`missed!=0`), even though it can remain internally for ID continuity. This is confirmed code behaviour: [update_full](../extracted/adaptive_pipeline/tracking/tracker.py:224), [visibility/output gate](../extracted/adaptive_pipeline/tracking/tracker.py:264). The table is not proof that all 6,239 boxes could be tracked successfully: GT-to-tracker associations, false-positive suppression and low-confidence geometry were not traced internally.

Among those 6,239 cases, 3,924 never had a correct ≥0.25 box at any earlier FULL under this schedule; only 492 had such evidence at the FULL immediately preceding the most recent FULL. Also, 3,782 of the 6,239 previous correct boxes score below 0.10. Consequently **enabling the existing 0.10 low-score association option alone cannot recover most of this candidate pool**: it needs an existing track and does not create low-score births. Lowering birth confidence is a separate accuracy/FP/cost hypothesis, not an automatic solution.

The published SKIP gap depends strongly on confidence:

| Saved matching on quality-aware SKIP frames | Dense TP | Adaptive TP |
|---|---:|---:|
| IoU50, all retained scores | 14,358 | 7,240 |
| IoU50, confidence≥0.25 | 7,340 | 7,005 |
| IoU75, confidence≥0.25 | 5,571 | 5,307 |
| IoU95, confidence≥0.25 | 1,584 | 1,304 |

The much smaller high-confidence gap supports the admission diagnosis; the remaining stringent-IoU gap supports a real localisation limitation. It does not establish exact causal percentages of AP loss.

**Confidence decay is not deleting tracks.** Outputs multiply stored confidence by `0.93**age` but do not delete on that score. The minimum saved SKIP score is ~0.2325, well above each run's ~0.11 best-F1 threshold. Deletion/hiding instead follows detector misses and flow-coasting rules. Decay can affect ranking and the fixed-0.25 summary, but is not the main missing-object mechanism here. [Decay and output](../extracted/adaptive_pipeline/tracking/tracker.py:263).

New-object discovery is structurally limited without a detector call, but only 27 Dense-only cases lack the previous-FULL annotation identity. That is 0.37% of this candidate loss set, not evidence that new objects are the dominant current issue. This result should not be extrapolated to longer SKIP gaps or trusted as a physical birth count given annotation-ID noise.

### Localisation and false positives that remain

In cases with both a previous FULL box and current Dense box correct at IoU75 and confidence≥0.25, the original adaptive subset has 366/5,366 propagated boxes below IoU75 (6.82%); anchoring has 174/5,381 (3.23%); quality-aware has 145/5,117 (2.83%). These are different selected SKIP populations, so the percentages are descriptive. On **1,610 shared SKIP frames**, anchoring has 5,375 saved TP75 versus the original's 5,181, which more directly supports improved geometry, while allowing different preceding histories.

Quality-aware SKIP predictions have median nearest-same-class IoU 0.837, median centre error 3.47% of GT diagonal, median absolute width error 5.93%, and median absolute height error 4.94%, conditional on finding a same-class IoU≥0.10 counterpart. This selection excludes total misses and is optimistic about identity; these are not unbiased tracking error estimates.

The quality-aware run has 2,207 non-ignored SKIP FPs at IoU50. Overlap-based candidate labels divide them into 627 coarse-localisation cases, 626 wrong-class overlaps, 215 duplicates/matching conflicts, and 739 background/stale/very-poor-localisation cases. Exact stale ghosts cannot be distinguished from detector-seeded FPs with the saved exports. FULL rows deliberately store track ID `-1`, so they lack the prediction→internal-track links needed for exact identity tracing. [Diagnostic export](../extracted/adaptive_pipeline/pipeline.py:216).

## 4. Why anchoring helps and quality-aware flow adds little

The original tracker smooths detector observations into Kalman state and later emits that state, although FULL output itself is raw YOLO. Anchoring separates emitted geometry, restores confirmed detector boxes exactly, derives velocity from detector observations, and emits accepted flow measurements instead of a lagging filtered box. This mechanism fits the high-IoU improvement. It is observation-inspired engineering, not an implementation of the full OC-SORT algorithm. [Box selection](../extracted/adaptive_pipeline/tracking/tracker.py:67), [flow update](../extracted/adaptive_pipeline/tracking/tracker.py:125), [detector confirmation](../extracted/adaptive_pipeline/tracking/tracker.py:169).

Sparse flow uses 16 interior grid points, forward/backward validation, median translation and **one isotropic scale factor shared by width and height**. Scale is fixed to one when the minimum side is below eight flow pixels; accepted scale is clipped to [0.8,1.25]. This cannot independently model aspect-ratio/visible-extent changes. Background points can move consistently while object boundaries change. However, only 289/9,646 emitted quality-aware SKIP boxes have a flow-resolution minimum side below eight pixels: the frozen-scale rule is not the main explanation for the current emitted-box population. Most tiny targets never became tracks. [Flow estimation and scale](../extracted/adaptive_pipeline/tracking/flow.py:91).

Quality-aware flow adds point-fraction/spread/dispersion gates and a composite confidence; it does **not replace the accepted translation/scale estimator**. It also changes Kalman measurement variance, but anchoring overwrites emitted geometry with the accepted measurement, limiting its direct effect. The legacy controller uses boolean flow failures rather than the continuous risk/covariance policy. Most importantly:

- Anchoring and quality-aware flow both SKIP 1,427 frames; **1,417/1,427 (99.30%) prediction arrays are bitwise identical**. Their saved TP75 totals on these common frames are exactly equal (4,555 each).
- Quality-aware flow changes 260 anchoring-SKIP frames to FULL and 200 anchoring-FULL frames to SKIP: a net **60 additional detector calls**.
- Anchoring→quality custom AP50–95 changes only **+0.0388 percentage points**, from 13.9231% to 13.9619%. Original→anchoring improves +0.1846 points with four fewer calls.

This strongly supports changed refresh decisions as the main mechanism of quality-aware flow's incremental result. It does not demonstrate better accepted box estimates. [Quality gates](../extracted/adaptive_pipeline/tracking/flow.py:110), [legacy policy](../extracted/adaptive_pipeline/adaptive_framing/controller.py:93).

Flow quality is not calibrated to box correctness. For emitted quality≥0.95 boxes, only 3,530/7,138 have any same-class IoU75 target. This includes false detector seeds and existing localisation errors; it does **not** prove that point displacement itself is wrong. Raw point trajectories, FB residuals and per-track box histories were not saved, so camera-motion, occlusion-boundary contamination and scale-model contributions cannot be separated now.

## 5. Is adaptive scheduling better than Fixed-2?

**Not demonstrated by this benchmark.** Original adaptive uses 1,733 FULL calls versus Fixed-2's 1,710 and scores 13.7385% versus 13.7719% custom AP50–95. The difference is only −0.0334 points, without uncertainty estimates. Its total saved TP50 count is actually one higher (22,860 versus 22,859), illustrating why an AP ranking cannot be explained by object-count totals alone.

The legacy “uncertainty” rule refreshes tracks whose stored detector confidence is below 0.65 after two frames. It does not use KF covariance. With `max_local_rois=0` and `use_local=True`, any such need forces FULL. The original run logs 1,450 `too_many_refresh` reasons, versus only 21 `track_failure`, 34 `staleness`, and 227 `novelty` reasons; reasons can overlap. Of all FULL calls, 1,369 have the sole reason `too_many_refresh` with zero failed-flow fraction. **Confidence-age refreshes dominate the computation.** [Decision rule](../extracted/adaptive_pipeline/adaptive_framing/controller.py:100).

Original adaptive has 1,608/1,683 SKIPs at age one; quality-aware has 1,552/1,627. The remaining quality-aware SKIPs have ages two (42 frames) and three (33). All risk/budget/covariance scheduling options remain disabled (`controller_mode=legacy`); the recorded detector-budget fraction is not an enforced budget in this run. Also, setting `use_local=False` is **not equivalent** to the recovered zero-ROI FULL fallback: it would remove this legacy refresh branch. These settings must not be casually changed during a “fair” control.

Original adaptive and Fixed-2 disagree on many frames despite similar call totals:

| Fixed-2 / original-adaptive decision | Frames | Fixed TP50 | Adaptive TP50 |
|---|---:|---:|---:|
| FULL / FULL | 995 | 9,362 | 9,362 |
| FULL / SKIP | 715 | 5,765 | 2,677 |
| SKIP / FULL | 738 | 2,848 | 5,950 |
| SKIP / SKIP | 968 | 4,884 | 4,871 |

Detection upgrades gain 3,102 TP50 while downgrades lose 3,088, and both-SKIP differences lose another 13: almost complete cancellation. Different histories and scores matter, so this is call-allocation accounting, not a causal intervention or an AP decomposition. It explains why additional calls need not establish a better trade-off.

The scheduler is not wholly insensitive to motion. At age one, original-adaptive FULL frames have median annotated motion 0.047 GT diagonals versus 0.028 on SKIPs. However, within quality-aware age-one SKIPs, Spearman correlation of the missing-Dense-TP fraction with median GT motion is only 0.11, novelty 0.13, and mean flow quality +0.06. These observational correlations are restricted to selected SKIPs, affected by size/confidence and temporal dependence, and have no causal/significance interpretation. They provide little evidence that current scalar signals target the dominant confidence-gate loss.

Quality-aware beats Fixed-2 by +0.1900 custom AP points but uses **79 additional calls**. It is not a matched-compute victory. On sequence_04, all methods have identical reported AP while adaptive overhead/timing varies; that variation must not be interpreted as a new accuracy mechanism. Timing is one sequential CPU repetition, excludes decode/scoring and is not streaming AP.

## 6. Research implications and compatibility

| Primary work | Relevant lesson | Application here, cost and limits |
|---|---|---|
| [Li et al., Towards Streaming Perception, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470460.pdf) | Streaming accuracy evaluates the latest available output against the world at query time, including latency. | Our frame-aligned offline AP/FPS does not implement this. A future timestamped output-buffer/deadline experiment is separate. It needs timing/capture instrumentation, not a different YOLO model. Do not import their streaming AP gains as expected detection-AP gains. |
| [Luo et al., Detect or Track, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4906) | Scheduling should estimate tracking quality; its scheduler labels depend on tracking IoU and emerging/disappearing objects. | This supports calibrating refresh signals to actual localisation/retention risk. Its learned Siamese/CNN scheduler requires feature computation and training, a substantial addition outside the current scope. Copying it is not the minimal next step; a fixed control and diagnostic calibration come first. |
| [Cao et al., OC-SORT, CVPR 2023](https://arxiv.org/html/2203.14360v3) | Observation-centric re-update repairs accumulated filter error after reacquisition; momentum/association also matter. | Supports preserving clean detector observations. Existing anchoring is simpler and already helps. A full replacement adds trajectory-history/association logic and cannot recover never-detected classes or retrospectively repair an already-emitted online result. Tracker-only speed claims omit detector cost. Its tracking results are not detection AP. |
| [Zhang et al., ByteTrack, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) | Low-score boxes can confirm existing tracks instead of being discarded; new low-score births need care. | Most directly relevant to the admission/confirmation mismatch. The existing second-stage association option is cheap relative to another detector pass but cannot address never-seeded targets. Lowering birth confidence increases track count, flow and association cost and propagates FPs. Paper MOT metrics/detector/training differ; no AP gain is promised. |

These are mechanism-level connections, not endorsements of whole algorithm replacements. Optional motion compensation currently adjusts novelty/scene validity rather than warping box geometry directly ([call site](../extracted/adaptive_pipeline/pipeline.py:119)); raw optical flow already measures apparent camera-plus-object motion. Appearance verification only helps trigger refresh, not classify unseen objects. Quality-based score decay, low-score association, risk/budget scheduling, camera/appearance verification, LOCAL births and CLAHE were all disabled. None has been demonstrated effective on this subset.

## 7. Minimal strategy and approval-gated experiments

Keep stock YOLOv8n, the existing raw FULL outputs, the recovered class mapping, and anchoring as the current candidate. **First establish official scoring and isolate confidence handling; do not add a new tracker or scheduler network.** Treat Fixed-2 with the same propagation as the required control. Only after these checks should box-risk scheduling or more complex geometry be designed.

A fixed schedule remains a valid final candidate if the matched-propagation, matched-cost comparison favours it. Adaptive scheduling should earn its additional complexity through measured benefit; the project should not force an adaptive winner or trade away a strong control to preserve that narrative.

The following are **proposed and not executed**. No command is presented as runnable for a driver that does not yet exist; approval would authorize preparing the small scoring/replay adapters and then running exactly this scope. Each directory must be fresh.

### E0 — official scoring of existing predictions (first recommendation)

- **Inputs:** all five methods' existing `preds.npz`, the existing manifest, and provided `annotations.json`; all 3,416 frames and eight categories. No image decoding or model inference is necessary.
- **Configuration:** official pycocotools COCO bbox protocol, IoU 0.50:0.05:0.95, 101 recall points, standard area ranges and `maxDets=[1,10,100]`; retain crowd semantics and provided pixel boxes. Export saved xyxy to xywh, keep supplied category IDs. In a copied/in-memory annotation object, renumber annotation IDs from one: this export has ID zero, while the official matcher uses zero as an unmatched sentinel. Do not alter the original JSON.
- **Isolation:** optionally report the same official scorer with a 300 per-class limit as an explicitly nonstandard auxiliary, and official-style accumulation of existing TP flags to distinguish accumulation effects from rematching. Report each protocol independently; no tuning to obtain favourable scores.
- **Output:** `runs/argoverse_hd_official_scoring_v1/`. Record pycocotools version, conversions, hashes, aggregate/per-sequence AP and AP_S/M/L. A scoring dependency/driver may need installation/preparation; neither was performed during this research.
- **Question resolved:** actual magnitude/direction of protocol differences and whether close method rankings survive. This improves measurement reliability, not detector accuracy.

### E1 — cached-detection replay to isolate tracking admission and geometry

- **Inputs:** all 3,416 existing images in the same five sequences plus the saved Dense detections. A detector adapter returns the cached Dense output only when the unchanged pipeline requests FULL. No new YOLO inference; no LOCAL passes. Reset state at every original boundary.
- **Common settings:** recovered CPU/FP32/640 configuration and eight-class map; fixed interval two; original `conf_decay=0.93`, association IoU 0.3 and output NMS 0.7. Use the same provided labels and a fixed scoring protocol established in E0.
- **Six arms:** (A) `fixed2_093`; (B) `fixed2_observed`; (C) `fixed2_quality`; (D) B with only `low_score_association=True, association_low_conf=0.10`; (E) B with only `track_min_conf=0.10` (low-score association stays off); (F) B with only `conf_decay=1.0`.
- **Output:** `runs/argoverse_hd_cached_tracking_diagnostic_v1/{A,B,C,D,E,F}/`. Record TP/FP/FN by size/confidence/age, class AP, births, confirmations, hiding/deletion reasons, linked FULL/SKIP IDs, raw flow points/spread/residuals, track count and requested calls. Cached-replay FPS is **not** a deployed pipeline speed measurement.
- **Question resolved:** whether useful low-score evidence can be recovered without unacceptable FPs/cost; anchoring versus quality gates at an exactly fixed schedule; confidence ranking versus geometric effects. E is a diagnostic threshold control, not a recommended production setting. At confidence 0.10 Dense has 21,638 FP50 versus 4,763 at 0.25, so indiscriminate lower-threshold births are risky.

### E2 — controlled Dense resolution/raw-output diagnostic

- **Inputs:** exactly 64 frames per sequence, 320 total, with indices `round(linspace(0, N-1, 64))` for each verified sequence length N; write and hash the selected manifest before any inference. This spans the sequences rather than selecting favourable examples.
- **Three arms:** stock checkpoint at `imgsz=640,960,1280`, each Dense only, same explicit CPU device, FP32, confidence 0.001, NMS IoU 0.7, maximum 300, default best-class behaviour and identical eight-class mapping. Same warmup policy per arm. No CLAHE, crops, augmentation, training or model replacement.
- **Extra diagnostics:** retain the 80-class post-NMS output before LUT filtering and whether the global 300 limit is reached; retain candidate/class information before truncation if technically available without changing prediction semantics. These records were missing in the original run.
- **Output:** `runs/argoverse_hd_resolution_diagnostic_v1/{imgsz640,imgsz960,imgsz1280}/`. Compare official size/class AP, confidence/FP distributions and processing latency. Original model cost formula estimates ~2.19× and ~3.85× detector arithmetic at 960 and 1280 relative to 640 for this aspect ratio; these are not measured time estimates.
- **Question resolved:** causal benefit of input resolution and evidence for cap/class-selection losses. No assumption that the highest resolution wins the accuracy–cost trade-off. If global truncation is confirmed, a separately specified pre-cap class-filter experiment is needed; it should not be silently combined with resolution changes.

Do not run a broad risk/LOCAL/camera/appearance sweep now. After E0/E1, a scheduler experiment should compare the **same** propagation/confidence settings at matched detector calls and assess refresh benefit with counterfactual propagated outputs logged on FULL frames. Existing FULL outputs omit that counterfactual, so present “wasted call” claims cannot be proved. Freeze choices before held-out testing, keep these five sequences as development/diagnostic data, and request separate authorization for broader data or timing repetitions.

## Reproduction and remaining limits

Only these archive-inspection commands were executed for new numerical analysis:

```sh
.venv/bin/python analysis/argoverse_hd_root_cause_v1/inspect_saved_outputs.py
.venv/bin/python analysis/argoverse_hd_root_cause_v1/focused_diagnostics.py
.venv/bin/python analysis/argoverse_hd_root_cause_v1/scheduler_and_fp_diagnostics.py
```

The scripts intentionally fail rather than overwrite their JSON outputs. No evaluation/runner modules are imported. Exploratory shell snippets only inspected source or summarized those existing files. The earlier benchmark's [metrics and per-sequence table](ARGOVERSE_HD_RESULTS.md) remain the source of all AP numbers in this report.

Unresolved items are exact official COCO scores, pre-LUT/global-cap losses, pre-NMS alternative classes, true tracker-ID/GT links, point displacement versus box-boundary error, physical births versus annotation fragmentation, causal resolution gains, and matched-budget scheduler superiority. There is no basis to promise metric gains, declare a new detection algorithm necessary, or attribute all failures to optical flow. No existing results, dataset files or algorithm sources were modified; no commits or pushes were made.
