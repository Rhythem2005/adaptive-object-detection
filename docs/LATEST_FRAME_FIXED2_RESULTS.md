# Latest-frame + Practical Fixed-2: controlled integration results

**Recommendation: retain the combined processor as an optional integrated mode; do not replace latest-frame Dense640 as the reference.** Original latest-frame behavior was preserved and the combined system improved overload freshness, precision and timing-aware accuracy. It did not pass all predeclared adoption criteria: overload CPU savings were 4.873%, below the required 5%, and the second 15 FPS repetition exceeded the paired P95 latency allowance. No criterion was relaxed and no candidate was tuned after results.

## Scope and controls

- One candidate, 12 fresh-process passes: A=FIFO Dense640, B=latest-frame Dense640, C=latest-frame Practical Fixed-2. Two repetitions at 15 FPS and 90 FPS. Orders: R1 15 ABC; R1 90 BCA; R2 90 ACB; R2 15 CBA. Ten-second pauses separate the four blocks. This counterbalances order; it does not fully balance every predecessor effect.
- Exactly 300 unique Argoverse-HD sequence_00 frames (IDs 0–299), reused in each pass. Source timestamps are original integer nanoseconds, time-scaled for 15/90 FPS replay. 1,512 regular annotations and 30 crowds remain unchanged. This prefix supports only car/truck/bus regular targets and is neither held-out validation nor five-sequence streaming evaluation.
- Stock YOLOv8n, CPU FP32, imgsz640 (observed tensor 1×3×416×640), confidence .001, NMS IoU .7, global max_det300, unchanged eight-class mapping. Torch intra/inter-op8/1, OpenCV10 threads/OpenCL off. Three full-frame warmups precede each source epoch; no ROI inference, GPU execution, retraining or resolution experiment occurred.
- Configuration C is copied exactly from Experiment E config B: observation anchoring, existing-track low-confidence recovery, 3×3 LK flow, decay .93, and fixed_signal_fastpath. Safety overrides are new to this isolated stream adapter; the frozen offline algorithms/defaults/configurations are unchanged.
- Source JPEG reads/decode are independent of processing. The same unchanged capacity-one buffer selects the newest published packet before processing decisions. Retaining the last processed grayscale image is temporal state, not another pending queue. Inputs were hashed before execution, so disk-cache conditions are warm rather than forced cold.

## Primary measured comparison

AP in this table is the custom replay-time-aligned held-output diagnostic, scored on the same 300 source-arrival queries. It is **not official streaming AP or conventional per-frame detector AP**. P/R are micro at confidence .25 and IoU .50; TP/FP are frame-observation counts.

| FPS | Rep | Arm | Consumed / dropped | YOLO calls | Flow attempted / accepted | Loop CPU s | P95 latency ms | Active output FPS | Aligned AP % | TP / FP | P / R % |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 1 | A | 300 / 0 | 300 | 0 / 0 | 19.198 | 37.57 | 15.00 | 18.3806 | 876 / 396 | 68.87 / 57.94 |
| 15 | 1 | B | 300 / 0 | 300 | 0 / 0 | 18.632 | 40.33 | 15.00 | 18.3731 | 876 / 397 | 68.81 / 57.94 |
| 15 | 1 | C | 300 / 0 | 161 | 161 / 139 | 13.985 | 45.71 | 15.00 | 17.7408 | 862 / 370 | 69.97 / 57.01 |
| 15 | 2 | A | 300 / 0 | 300 | 0 / 0 | 19.015 | 37.55 | 15.00 | 18.3806 | 876 / 396 | 68.87 / 57.94 |
| 15 | 2 | B | 300 / 0 | 300 | 0 / 0 | 19.079 | 37.63 | 15.00 | 18.3806 | 876 / 396 | 68.87 / 57.94 |
| 15 | 2 | C | 300 / 0 | 161 | 161 / 139 | 14.412 | 46.57 | 15.00 | 17.7408 | 862 / 370 | 69.97 / 57.01 |
| 90 | 1 | A | 300 / 0 | 300 | 0 / 0 | 16.328 | 2384.64 | 50.27 | 0.7585 | 254 / 957 | 20.97 / 16.80 |
| 90 | 1 | B | 176 / 124 | 176 | 0 / 0 | 9.838 | 36.25 | 52.07 | 6.1179 | 685 / 593 | 53.60 / 45.30 |
| 90 | 1 | C | 248 / 52 | 138 | 138 / 110 | 9.052 | 33.75 | 74.05 | 8.3480 | 748 / 444 | 62.75 / 49.47 |
| 90 | 2 | A | 300 / 0 | 300 | 0 / 0 | 17.037 | 2896.59 | 41.84 | 0.8117 | 255 / 913 | 21.83 / 16.87 |
| 90 | 2 | B | 171 / 129 | 171 | 0 / 0 | 9.655 | 37.50 | 50.57 | 5.7628 | 654 / 621 | 51.29 / 43.25 |
| 90 | 2 | C | 279 / 21 | 150 | 149 / 129 | 9.491 | 32.17 | 83.08 | 8.5768 | 768 / 441 | 63.52 / 50.79 |

**At 90 FPS, C versus B:** mean YOLO calls fell 173.5→144 per 300 arrivals (**17.0029%**). Mean CPU fell 9.746627→9.271662 s (**4.8731%**). Paired CPU savings were about 7.98% and 1.70%; the mean misses the 5% gate by 0.1269 percentage points. Do not round this to a pass.

Mean aligned AP rose **5.94036→8.46240% (+2.52205 points)**. Mean FP25 fell **607→442.5 (−27.10%)**, while mean TP25 rose 669.5→758. P95 capture-deadline-to-result latency improved in both repetitions. C processed substantially more source frames while retaining one pending frame. These are useful overload quality/freshness gains despite modest total CPU savings.

**At 15 FPS:** all arms processed 300 frames without drops. C used 161 detections and 139 accepted propagations in each repetition, saving **46.3333%** of detector calls and **24.6971%** of mean CPU versus B. Mean aligned AP fell **18.37687→17.74077% (−0.63610 points)**, inside the declared accuracy tolerance. Precision improved and recall fell slightly. The second repetition C P95 was **46.5723 ms**, above its permitted **43.2769 ms**; that latency gate failed.

## Latency, freshness, occupancy and resources

Capture-to-result starts at the mapped replay arrival deadline **before JPEG decode** and ends at pipeline output readiness. It includes decode, pending wait and selected-frame processing; excludes model startup/warmups, scoring, serialization and physical display. Historical CSV column inference_start_ns means selected-frame **processing start** for C, including FULL or flow; it is not always a YOLO invocation.

| FPS / rep / arm | P50 / P95 / max latency ms | Processing-start age P95 ms | Query-visible source age P95 ms | Peak pending | Peak RSS MiB | Post-EOF drain ms | Output FPS through drain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 / 1 / A | 36.73 / 37.57 / 38.46 | 16.76 | 66.67 | 1 | 494.66 | 20.01 | 15.02 |
| 15 / 1 / B | 36.55 / 40.33 / 71.21 | 17.49 | 66.67 | 1 | 503.17 | 19.74 | 15.02 |
| 15 / 1 / C | 37.64 / 45.71 / 59.20 | 19.85 | 66.67 | 1 | 511.44 | 2.27 | 15.03 |
| 15 / 2 / A | 36.69 / 37.55 / 38.67 | 16.75 | 66.67 | 1 | 494.52 | 19.64 | 15.02 |
| 15 / 2 / B | 36.67 / 37.63 / 46.59 | 16.89 | 66.67 | 1 | 499.00 | 19.60 | 15.02 |
| 15 / 2 / C | 35.48 / 46.57 / 49.74 | 20.64 | 66.67 | 1 | 508.11 | 2.29 | 15.03 |
| 90 / 1 / A | 1315.88 / 2384.64 / 2539.12 | 2361.87 | 1402.23 | 131 | 1810.92 | 2531.90 | 51.18 |
| 90 / 1 / B | 31.40 / 36.25 / 38.18 | 17.36 | 44.44 | 1 | 537.38 | 31.29 | 52.37 |
| 90 / 1 / C | 25.96 / 33.75 / 51.34 | 17.49 | 47.22 | 1 | 534.94 | 9.99 | 74.27 |
| 90 / 2 / A | 1917.82 / 2896.59 / 3003.38 | 2878.53 | 1711.12 | 160 | 2029.48 | 2995.96 | 47.43 |
| 90 / 2 / B | 32.17 / 37.50 / 42.91 | 17.48 | 55.56 | 1 | 528.30 | 35.88 | 50.82 |
| 90 / 2 / C | 25.55 / 32.17 / 35.42 | 15.13 | 44.44 | 1 | 532.77 | 18.47 | 83.34 |

Active output FPS counts results delivered inside the common first-to-last source-arrival window. Drain-inclusive FPS includes post-EOF processing. Neither is an unpaced capacity estimate or YOLO-only throughput. The old offline 62.101 FPS result has not been reproduced as an integrated streaming capacity claim.

Query-visible age is the source age of the last completed output, sampled on the same source-time queries. Empty output before first completion is retained in AP; undefined initial ages are excluded from age summaries, with counts retained separately. Nonempty-result query counts and COCO recall are both provided; nonempty output alone is not object detection coverage.

CPU seconds use process time across capture, processing and backend threads, including source waits/drain in the wall scope. CPU per source and per consumed frame differ:

| FPS / rep / arm | CPU ms / source arrival | CPU ms / consumed frame | YOLO % of consumed |
|---|---:|---:|---:|
| 15 / 1 / A | 63.994 | 63.994 | 100.00 |
| 15 / 1 / B | 62.108 | 62.108 | 100.00 |
| 15 / 1 / C | 46.617 | 46.617 | 53.67 |
| 15 / 2 / A | 63.385 | 63.385 | 100.00 |
| 15 / 2 / B | 63.595 | 63.595 | 100.00 |
| 15 / 2 / C | 48.041 | 48.041 | 53.67 |
| 90 / 1 / A | 54.427 | 54.427 | 100.00 |
| 90 / 1 / B | 32.793 | 55.897 | 100.00 |
| 90 / 1 / C | 30.174 | 36.501 | 55.65 |
| 90 / 2 / A | 56.789 | 56.789 | 100.00 |
| 90 / 2 / B | 32.185 | 56.464 | 100.00 |
| 90 / 2 / C | 31.637 | 34.018 | 53.76 |

Pure detector/tracker CPU attribution is unavailable while decoding runs concurrently. Internal component wall scopes and total process CPU are reported; overlapping process-CPU intervals are not added as independent component costs. Peak RSS is the macOS process high-water mark including startup, not incremental tracker allocation.

## Scheduling, gap safeguards and original latest-frame behavior

- Every latest-mode dequeue was independently checked against publication timestamps: it selected the newest available source frame. Capture continued independently, pending occupancy never exceeded one, and all pending work drained at EOF. Frames were never intentionally selected or withheld to obtain a Fixed-2 source-ID pattern.
- Scheduling advances once per consumed packet. Tracker prediction/age retain verified original source-frame units, cross-checked against original timestamp elapsed time; the scaled replay arrival clock does not define object-motion time. Actual source cadence bounds were 33,299,640–33,300,352 ns.
- FULL restarts the consumed-step cycle. A rejected nominal propagation can therefore shift later FULL decisions. Forced-FULL counts are not identical to calls above ceil(consumed/2).
- Unsafe visible flow is rejected before tracker correction, then the normal FULL detector/association path refreshes the same frame. No rejected-flow retry/double tick occurs. Empty/invalid state, incompatible cadence and existing k_max=8 staleness safeguards are available; they are not calibrated LK reliability guarantees.

| FPS / rep | Forced FULL | Gap-associated forced FULL | Fallback reason | State resets | Observed source gaps |
|---|---:|---:|---|---:|---|
| 15 / 1 | 22 | 0 | {'visible_flow_failure': 22} | 0 | {'1': 300} |
| 15 / 2 | 22 | 0 | {'visible_flow_failure': 22} | 0 | {'1': 300} |
| 90 / 1 | 28 | 26 | {'visible_flow_failure': 28} | 0 | {'1': 199, '2': 47, '4': 1, '3': 1} |
| 90 / 2 | 20 | 14 | {'visible_flow_failure': 20} | 0 | {'1': 258, '2': 21} |

All 92 observed overrides were visible-flow failures; no scale-limit, cadence, nonfinite-state or staleness reset fired. Of these overrides, 40 occurred across source gaps greater than one. Consecutive frames can also fail flow. Accepted real-data propagation covered gaps of one and two; larger observed gaps of three/four received FULL. Thus the experiment does not validate arbitrary-gap optical flow, long outages, or the untriggered safety branches; those branches have synthetic coverage only.

Across all passes: **3,600 source presentations of 300 unique images; 3,274 consumed frames; 326 overwrites; 2,757 measured YOLO forwards; 609 flow attempts; 517 accepted propagations; 92 rejected-flow fallbacks.** Normal EOF accounting was exact in every pass: captured=completed+overwritten, consumed=completed, remaining pending=0. There were no capture or detector failures.

## Official detection AP on comparable populations

Official pycocotools bbox AP uses the same image IDs within each row below. At 90 FPS these are the intersection consumed by all three arms; they are conditional populations, not unbiased full-stream AP estimates. All A/B outputs and every C FULL output matched archived Dense predictions/details byte for byte. C propagated outputs were evaluated as emitted, not replaced by Dense cache rows.

| Rate / repetition | Common images | A / B official AP % | C official AP % |
|---|---:|---:|---:|
| 15 / 1 | 300 | 23.600778 | 22.729762 |
| 15 / 2 | 300 | 23.600778 | 22.729762 |
| 90 / 1 | 144 | 23.370999 | 22.777661 |
| 90 / 2 | 159 | 23.381881 | 22.739494 |

The common-frame AP loss shows that propagation is intrinsically less accurate than a fresh detector on the same images. The better overload timing-aware AP arises from fresher, more frequent completed outputs, not superior per-frame detector accuracy. These prefix AP values must not be compared directly with the full five-sequence 15.732713% Dense benchmark.

## Error evidence and remaining bottlenecks

- At 15 FPS, 139 accepted flow frames lost 19 Dense TP25 events and recovered 7 (net −12 TP); FP25 fell 176→148 on those same flow frames. Among the 19 losses, 11 have a same-class partial-overlap candidate (IoU .10–.50), 7 retain a geometrically valid candidate but lose confidence/matching, and 1 has no same-class candidate at IoU .10. These are overlap diagnostics, not proven physical object identities or a pure optical-flow causal decomposition.
- In overload, accepted flow frames lost 19/16 Dense TP25 events and recovered 8/8 respectively. Both source-gap-one and source-gap-two losses occur. The small number of accepted gap-two observations—23 and 7—limits claims about larger-gap robustness.
- Ground-truth track metadata passed uniqueness and category-consistency checks. At query times, newly annotated targets since the last delivered FULL were missed in 17/39/41 cases (normal/R1 overload/R2 overload). **All these cases were also missed by instantaneous cached Dense at confidence .25.** These counts therefore do not establish missed new arrivals as a propagation-specific bottleneck on this sample. They describe annotation appearances within the prefix, not physical arrival times.
- The normal-rate latency regression is associated with longer measured FULL service: C means 24.80/24.74 ms versus B 20.53/20.41 ms. C detector-call wall means were 23.43/23.36 ms versus B 20.50/20.38 ms; C residual non-detector FULL work was about 1.37/1.38 ms. FULL source populations and workload cadence differ, so the detector-duration difference cannot be causally assigned to tracking. Thermal state, CPU frequency and duty-cycle effects were not measured.
- Fewer calls do not translate proportionally to CPU savings. In overload C processes 248/279 frames rather than 176/171, pays flow/preparation/association costs and uses 28/20 safety fallbacks. The same capture/decode work covers all 300 arrivals. These measured costs leave only 4.873% average CPU saving despite 17.003% fewer calls.

## Predeclared gates

Criteria were frozen before inference: at 15 FPS ≥30% fewer calls and ≥20% less CPU; at 90 FPS ≥10% fewer calls and ≥5% less CPU; neither paired CPU increase >5%; C P95≤max(B+5 ms,1.15×B) in each pair; max latency≤150 ms. Mean aligned AP loss≤min(1 point,10% of B), no paired loss>max(1 point,25% of B); mean FP25≤110% and recall25≥90% of B. At normal input C must consume all300 without drops.

| Rate | Calls reduction | CPU reduction | Accuracy / FP / recall gates | Latency gate | Overall |
|---|---:|---:|---|---|---|
| 15 | 46.3333% | 24.6971% | Passed | Failed R2 P95 | Not accepted |
| 90 | 17.0029% | 4.8731% | Passed | Passed | Not accepted |

At 90 FPS the failure is the CPU minimum, not latency or accuracy. These margins are descriptive engineering criteria; two repeats of one scene do not provide statistical significance or independent generalization. No threshold was changed after results.

## Validation, preservation and execution costs

- Initial development run:21/22 new tests passed; one assertion incorrectly counted an empty/no-op Kalman-update call as an actual correction. The assertion was corrected to inspect the empty index and detector correction separately. Final result: **157 tests passed, zero failures, zero skips.** This includes 27 new integration/evaluation tests and the previous130. Synthetic coverage includes capture during both detection and flow, capacity-one replacement/metadata, consumed-step schedules, confidence aging, rejected correction, repeated fallbacks, staleness/state resets, covariance checks, and scoring actual propagated predictions.
- The separate cached offline regression reproduced all300 archived Practical predictions, details and tracker-state hashes exactly using150 cache lookups and zero new YOLO forwards (1.94 wall s / 2.31 CPU s). No five-sequence replay/evaluation occurred.
- Actual detector forwards **2,793 = 2,757 measured + 36 warmups**, below3,636. Timed worker phase including setup/validation and interblock pauses finished in **196.58 s**. Final acceptance JSON was recovered **327.52 s** after execution start, below the600 s limit. Total measured replay-loop process CPU was 175.724 s; it excludes startup/warmup and scoring CPU.
- All inference passes and32 official scoring jobs completed once. Final JSON assembly then failed on a NumPy boolean. A separate report assembler rebuilt JSON from saved outputs using the frozen criteria; **no inference, replay, scoring, algorithm or configuration was rerun/changed**. FAILED.json, the partial summary, original traceback and recovery record remain preserved. The standalone scoring timer was lost in the exception and is reported unavailable rather than estimated.
- Preservation audit checked **2,710 prior archived files with zero mismatches**, plus107 protected current files. Only the approved run_stream.py integration was changed among preexisting source files. Base pipeline/tracker/controller/flow, capture/buffer, global defaults, frozen configs, historical reports/predictions and unrelated uncommitted work are unchanged.
- All workers stayed below4 GiB RSS. Outputs remain far below250 MB; no image/video copies were created. Actual per-worker runtime/device/thread metadata and power/thermal command outputs are retained. Sensor temperature/CPU frequency telemetry is unavailable; no thermal explanation is asserted.
- Camera capture, hardware exposure timestamps, upstream camera/decoder queues, physical display latency, other scenes and long-run memory stability remain unverified. This is timestamp-paced JPEG replay, not a live-camera result. General variable-frame-rate timing is not certified by this bounded source-specific cadence check.

## Implementation and reproduction

- [Streaming adapter](../extracted/adaptive_pipeline/streaming_fixed2.py): instance-local controller/flow wrappers delegate existing algorithms. No second pending queue or waiting-for-frame-ID mechanism exists.
- [Stream entry point](../extracted/adaptive_pipeline/run_stream.py): default Dense remains selectable; opt in with --processor practical-fixed2 and --policy latest. Packet timestamps reach the adapter and telemetry.
- [One-shot harness](../analysis/latest_frame_fixed2_validation.py), [report-only recovery](../analysis/latest_frame_fixed2_report.py), and [tests](../tests/test_streaming_fixed2.py). Full configs, exact12 subprocess commands, raw clocks, actual predictions, COCO tensors/matches, assignments, diagnostics and source hashes are under the new run root.

Executed commands (output locations refuse reuse):

```sh
PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=extracted:. .venv/bin/python -m analysis.latest_frame_fixed2_validation regression
PYTHONPATH=extracted:. .venv/bin/python -m analysis.latest_frame_fixed2_validation prepare
PYTHONPATH=extracted:. .venv/bin/python -m analysis.latest_frame_fixed2_validation run
# Final JSON-only repair after the preserved serialization exception:
PYTHONPATH=extracted:. .venv/bin/python -m analysis.latest_frame_fixed2_report
```

The integration is correct on this bounded replay and delivers a useful conditional trade-off: stronger overload delivered-output quality, lower false positives and lower P95; meaningful low-rate CPU savings with a latency-tail cost. **Keep it optional, preserve latest-frame Dense as the reference, and freeze these results for review.** It has not met the predefined bar for a universal replacement. No further optimization, paused960+Fixed-3 work, training, commit/push or new research phase was started.
