# Phase 3: accuracy, precision and computation trade-offs

Completed 20 October 2026. All approved initial experiments and the applicable conditional checks are complete: **12 scored full-sequence replays and 6 cache-only timing replays**, or 61,488 frame steps. Zero new YOLO inference, training, D/E experiments or Phase 4 work occurred. No algorithm defaults were changed and no commit/push was performed by this agent.

The best practical candidate from this batch is **the existing anchored Fixed-2 recovery configuration with a 3x3 flow grid**, subject to the limitations below. Its gains are small and descriptive, not a generalization or statistical-significance claim. No tested configuration improves every possible metric: low-score FP count, precision ratios, AP and total tracking work must be reported together.

## Validity and controls

Before new variants, the anchoring, recovery and shared-threshold controls reproduced Phase 2 predictions, prediction details, per-frame tracker-state hashes and complete official COCO precision/recall/score tensors bitwise. Aggregate, per-class, per-sequence and exact-confidence metrics agree exactly. Production source changes are confined to opt-in tracker settings and association/birth handling; detector weights, inference settings, flow estimator, scheduling algorithms and evaluator are unchanged.

All five sequences and 3,416 frames are retained, including all provided classes/crowds. The same 1920x1200 images are hashed before each replay; each sequence resets state. FULL frames return only the original, image-bound Dense cache; ROI/warmup/model calls are prohibited. Stock YOLOv8n's historical settings remain CPU FP32, imgsz 640, confidence 0.001, NMS 0.7 and global max_det 300. These settings describe the cache source; no YOLO device executed in this phase. Replay uses NumPy/SciPy/OpenCV CPU; scoring uses pinned pycocotools 2.0.11, normalized positive annotation IDs, original pixel coordinates, all eight categories and official maxDets=[1,10,100].

COCO AP means AP50–95. AP/AR tables are percentages; differences in AP are percentage points. Precision/recall/F1 at an exact confidence and IoU0.50 are supplementary operating-point statistics, not COCO summary slots. AP is not averaged across sequences. All results use the same official matching/crowd/cap semantics. No score was optimized for best F1, and no outputs were promoted to higher confidence.

## Initial eight controlled replays

All eight replay arms use identical 1,710 FULL requests and 1,706 SKIPs: 49.94% fewer requests than Dense. The previously scored Dense reference uses all 3,416 detector frames. Birth/association variants change tracking only; FULL outputs are identical to Dense on each selected image.

| Method | AP50 | AP50–95 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| Dense existing reference | 23.8650 | 15.7327 | 15.6544 | 23.1914 |
| Anchor control | 20.8355 | 14.2426 | 14.4945 | 18.2611 |
| Recovery control | 21.8745 | 14.6718 | 14.7636 | 19.0015 |
| Shared threshold 0.10 | 22.7152 | 15.0911 | 15.1171 | 20.0109 |
| V1 separate births 0.10 | 22.7152 | 15.0911 | 15.1171 | 20.0109 |
| V2 active recovery | 21.7815 | 14.6230 | 14.7334 | 18.9081 |
| V3 low-stage IoU 0.50 | 21.8473 | 14.6623 | 14.7634 | 18.9758 |
| V4 active + IoU 0.50 | 21.7486 | 14.6171 | 14.7343 | 18.8879 |
| V5 gated assignment | 21.8749 | 14.6716 | 14.7636 | 19.0013 |

All-frame exact-confidence 0.25, class-macro operating points:

| Method | Precision % | Recall % | F1 % |
| --- | --- | --- | --- |
| Anchor control | 66.2466 | 17.5720 | 24.4101 |
| Recovery control | 66.2466 | 17.5720 | 24.4101 |
| Shared threshold 0.10 | 66.2396 | 17.5702 | 24.4073 |
| V1 separate births 0.10 | 66.2396 | 17.5702 | 24.4073 |
| V2 active recovery | 66.2466 | 17.5720 | 24.4101 |
| V3 low-stage IoU 0.50 | 66.2466 | 17.5720 | 24.4101 |
| V4 active + IoU 0.50 | 66.2466 | 17.5720 | 24.4101 |
| V5 gated assignment | 66.2466 | 17.5720 | 24.4101 |

On the same 1,706 SKIP frames, official IoU0.50 matches show the low-score trade-offs:

| Method | TP≥0.10 | FP≥0.10 | Precision≥0.10 % | TP≥0.25 | FP≥0.25 | Mean tracks |
| --- | --- | --- | --- | --- | --- | --- |
| Anchor control | 7755 | 2392 | 76.4265 | 7492 | 2024 | 6.9824 |
| Recovery control | 9017 | 4434 | 67.0359 | 7492 | 2024 | 8.8448 |
| Shared threshold 0.10 | 10258 | 9878 | 50.9436 | 7491 | 2025 | 15.6724 |
| V1 separate births 0.10 | 10258 | 9878 | 50.9436 | 7491 | 2025 | 15.6894 |
| V2 active recovery | 8793 | 4042 | 68.5080 | 7492 | 2024 | 8.3249 |
| V3 low-stage IoU 0.50 | 8885 | 3969 | 69.1225 | 7492 | 2024 | 8.4578 |
| V4 active + IoU 0.50 | 8717 | 3688 | 70.2701 | 7492 | 2024 | 8.0849 |
| V5 gated assignment | 9019 | 4437 | 67.0259 | 7492 | 2024 | 8.8399 |

**V1 resolves the Phase 2 confound on this subset.** It keeps high-score association at 0.25, performs ordinary existing-track low-score recovery, then independently admits unmatched births down to 0.10. Relative to recovery-only it adds 0.419247 AP points, but adds 5,444 SKIP FPs for 1,241 extra TPs at 0.10. It reproduces the shared-threshold control's aggregate COCO/operating metrics exactly, although 4 frame prediction arrays differ. This supports low-score births as the conditional source of additional AP here; it does not establish universal equivalence of one/two-stage association or make births reliable.

**V2–V4 reduce low-score FP, but also remove true detections.** V3 is the most conservative precision-oriented option: AP changes by -0.009479 points; SKIP FPs at 0.10 fall 10.49% and TPs fall by 132. Mean tracks fall 4.38%. Its confidence 0.25 outputs have identical TP/FP/macro P/R/F1 counts to recovery-only. Therefore it is a precision/low-score-recall trade-off, not an established accuracy-preserving improvement. No noninferiority margin or deployment miss/false-alarm cost was supplied. V3 did not receive repeated timing or a grid-combination experiment.

The pre-specified conservative screen required no degradation in AP, AR100, macro P25 or TP25, and no increase in FP25, FP10 or mean tracks. **None of the five new variants passed.** The screen was not relaxed after seeing V3. Thus the optional second confidence engine was not promoted: only four efficiency controls on the existing recovery baseline were run. All V2–V5 results at 0.25 remain identical to recovery-only; their effects are primarily in lower-score retained evidence and subsequent history. V3 reduces low-score confirmations from 3,647 to 2,967: source TP50 confirmation events fall 1,232→1,088 and source FP50 events fall 2,258→1,744. This supports removal of both valid and invalid continuation evidence, not a new detection capability or a flow improvement.

### What did threshold-aware assignment actually change?

The old implementation maximizes summed IoU before rejecting invalid pairs. A synthetic matrix [[0.30,0.29],[0.29,0]] at threshold 0.30 selects two invalid pairs even though one valid match exists. V5 prioritizes valid-match cardinality, then summed IoU, within the same class and confidence stages. This alternative can also replace one very strong pair with two weaker valid pairs. It is an explicit objective change, not a universal correctness guarantee.

| History | Matrices | Different pair sets | Matrices with more valid pairs | Extra valid pairs |
| --- | --- | --- | --- | --- |
| Recovery control | 3253 | 40 | 32 | 32 |
| V5 gated assignment | 3253 | 38 | 30 | 30 |

V5's AP changes by -0.000206 points, with two more all-frame TPs and three more FPs at 0.10. The theoretical issue is real, but it is not a material bottleneck demonstrated by this benchmark. Do not adopt a more complicated association objective on this evidence. Same-matrix alternatives are diagnostic counterfactuals, not whole-run causal AP decompositions.

### Where the extra low-score seeds come from

| Method | Low birth TP50 events | Low birth FP50 events | Low-confirm TP50 events | Low-confirm FP50 events |
| --- | --- | --- | --- | --- |
| Anchor control | 0 | 0 | 0 | 0 |
| Recovery control | 0 | 0 | 1232 | 2258 |
| Shared threshold 0.10 | 427 | 2448 | 0 | 0 |
| V1 separate births 0.10 | 437 | 2548 | 2261 | 5896 |
| V2 active recovery | 0 | 0 | 990 | 1827 |
| V3 low-stage IoU 0.50 | 0 | 0 | 1088 | 1744 |
| V4 active + IoU 0.50 | 0 | 0 | 909 | 1444 |
| V5 gated assignment | 0 | 0 | 1236 | 2261 |

These are observation/update events, not unique physical objects. Crowd-ignored, capped and ambiguous duplicated source rows are retained separately in diagnostic JSON. Source TP50 establishes the current detector box/class match; it does not prove physical identity continuity. The paired target and linked-geometry diagnostics retain gained/lost GT identities, hidden states and all denominator definitions. Provided track IDs can fragment.

## Approved efficiency and phase controls

| Method | AP50–95 | AP50 | AR100 | FULL requests | Call savings % | Flow points |
| --- | --- | --- | --- | --- | --- | --- |
| Recovery control | 14.6718 | 21.8745 | 19.0015 | 1710 | 49.94 | 241312 |
| Fixed-2 phase U1 | 14.6084 | 21.8016 | 19.1702 | 1710 | 49.94 | 241568 |
| Fixed-3 phase U0 | 14.1113 | 20.8327 | 17.5541 | 1142 | 66.57 | 322016 |
| Fixed-3 phase U1 | 13.9549 | 20.9041 | 17.5560 | 1142 | 66.57 | 321376 |
| Recovery + 3x3 flow | 14.6922 | 21.8974 | 19.0122 | 1710 | 49.94 | 135855 |

The 3x3 grid changes only existing flow_grid from 4 to 3; flow width 640, minimum 4 valid points, LK window/pyramid/FB settings, anchoring, association and score decay remain fixed. It uses 43.70% fewer initial flow points (241,312→135,855). This changes spatial sampling and the valid-point fraction required by the fixed four-point rule; it is not a numerical-equivalence optimization. Both forward and backward flow still run. Pyramid preparation and other fixed work remain, so point count is not a latency or energy estimate.

Grid3 AP changes by +0.020428 points, AP75 by +0.023817, and AR100 by +0.010710. The accuracy changes are tiny. It does not solve the stock detector's small-object weakness.

All-frame IoU0.50 counts at the same FULL/SKIP frames:

| Confidence | Recovery TP | Grid3 TP | Recovery FP | Grid3 FP | Recovery micro P % | Grid3 micro P % |
| --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 24249 | 24261 | 275857 | 275862 | 8.0801 | 8.0837 |
| 0.1 | 19549 | 19561 | 15236 | 15239 | 56.1995 | 56.2098 |
| 0.25 | 15326 | 15335 | 4382 | 4373 | 77.7654 | 77.8110 |

At 0.25, grid3 adds 9 TPs and removes 9 FPs. At 0.10, it adds 12 TPs and 3 FPs. It slightly improves micro precision at all three reported thresholds, but **does not reduce the absolute FP count at every threshold**. Mean tracks increase by 0.007611. Consequently it also fails the original strict all-criteria screen (FP10/track-count proxies); it was timed as the pre-specified cost challenger, not secretly passed by relaxing that screen. The practical recommendation below weighs its measured point/time reduction and explicit precision ratios rather than calling it a strict all-metric Pareto win.

The second Fixed-2 phase scores below the original recovery phase. Fixed-3 uses 1,142 requests (66.57% savings) but loses accuracy under both phases. All count-matched phases share the anchored fixed early-FULL path; propagation is skipped before scheduled FULL. U0 is every k-th frame from 0. U1 retains frame 0 and shifts the remaining budget to 1+j*k. Budgets are equal per sequence, and end-of-sequence withholding can lengthen the final gap. **U1 is an offline phase/budget control, not a newly validated online scheduler.** The engine's large native fixed_k disables extra implicit refreshes; the complete prescribed schedule is the source of truth. No GT/future detection content enters decisions. U0 corresponds to ordinary periodic scheduling. These tests do not rehabilitate the original adaptive scheduler or establish a better closed-loop policy.

![Phase 3 measured trade-offs](../runs/argoverse_hd_phase3_v1/figures/tradeoffs.png)

## Repeated cache-only processing costs

Six runs used AB,BA,AB order, three per arm, with all 3,416 frames including initialization. Every output, detail array and tracker-state hash matches its corresponding scored replay bitwise. No deep observation/event wrapper executes in the timed stage; only a common lightweight flow-point counter and bound cache are present. Timing excludes image decoding, input hashing, output validation, serialization and model execution. It includes Python orchestration and construction of prediction details. CPU time includes library worker threads and can exceed wall time.

| Arm | Samples | Mean wall ms | Median wall ms | P95 wall ms | P99 wall ms | Mean CPU ms |
| --- | --- | --- | --- | --- | --- | --- |
| Recovery control | 10248 | 2.8372 | 2.8569 | 3.8477 | 4.3649 | 4.6563 |
| Recovery + 3x3 flow | 10248 | 2.7241 | 2.8129 | 3.5145 | 4.0072 | 4.4759 |

| Round | Recovery mean wall ms | Grid3 mean wall ms | Grid3 change % |
| --- | --- | --- | --- |
| 1 | 2.7538 | 2.6408 | -4.10 |
| 2 | 2.8584 | 2.8722 | +0.48 |
| 3 | 2.8992 | 2.6593 | -8.27 |

Pooled mean cache-step wall time changes -3.98%; CPU time changes -3.88%. These are measured cache-processing differences on this machine/runtime, not deployed YOLO+tracking speedups, real-time FPS or energy savings. Only three interleaved repetitions were run; system/thermal noise and order effects remain. No GPU/MPS speed claim is made. Original historical CPU-inference FPS is not combined with these timings to manufacture a new end-to-end number. D/E remain deferred.

Recorded stage wall-time sums were 378.9s for 12 scored replays (including decode/hash/diagnostics/I/O), 297.0s for official scoring, and 173.6s for six timing stages including decode/validation/I/O. These are execution-accounting sums, not elapsed task time or deployable throughput; setup, analysis, reporting, audit and the interruption are excluded.

## Per-sequence and class/size accuracy

All sequence cells below are independently evaluated COCO AP50–95 (%). Categories without regular GT in a sequence are excluded by COCO, so sequence class support differs.

| Method | sequence_00 | sequence_01 | sequence_02 | sequence_03 | sequence_04 |
| --- | --- | --- | --- | --- | --- |
| Anchor control | 6.6149 | 8.4806 | 12.7971 | 25.5091 | 22.8807 |
| Recovery control | 6.7533 | 8.8201 | 12.9992 | 26.1303 | 24.0009 |
| Shared threshold 0.10 | 6.8942 | 9.0975 | 13.3211 | 27.2428 | 24.3095 |
| V1 separate births 0.10 | 6.8942 | 9.0975 | 13.3211 | 27.2428 | 24.3095 |
| V2 active recovery | 6.7311 | 8.7841 | 12.9509 | 26.0934 | 23.8775 |
| V3 low-stage IoU 0.50 | 6.7375 | 8.8038 | 12.9617 | 26.1465 | 24.0189 |
| V4 active + IoU 0.50 | 6.7241 | 8.7705 | 12.9444 | 26.0694 | 23.8912 |
| V5 gated assignment | 6.7531 | 8.8160 | 12.9992 | 26.1326 | 24.0009 |
| Fixed-2 phase U1 | 6.7474 | 8.9056 | 13.0170 | 26.3582 | 23.9835 |
| Fixed-3 phase U0 | 6.3095 | 8.2452 | 12.3888 | 24.6221 | 23.5692 |
| Fixed-3 phase U1 | 6.3224 | 8.4214 | 12.3876 | 24.7970 | 24.0283 |
| Recovery + 3x3 flow | 6.7939 | 8.8583 | 13.0188 | 26.1725 | 23.9962 |

Grid3 improves AP in four sequences and decreases it slightly in sequence_04. The small pooled gain is not uniform across every class/size/sequence. Selected per-class AP50–95 (%); complete class metrics for every arm remain in scoring/metrics.json:

| Class | Dense | Recovery | Grid3 | V3 |
| --- | --- | --- | --- | --- |
| person | 16.4111 | 14.8743 | 14.8753 | 14.8579 |
| bicycle | 0.0406 | 0.0221 | 0.0221 | 0.0221 |
| car | 28.2476 | 26.3831 | 26.4573 | 26.3148 |
| motorcycle | 4.8986 | 3.5245 | 3.5345 | 3.5245 |
| bus | 54.9009 | 53.3860 | 53.3852 | 53.3861 |
| truck | 9.1756 | 7.7970 | 7.8749 | 7.8023 |
| traffic_light | 6.1056 | 5.5976 | 5.5987 | 5.6010 |
| stop_sign | 6.0817 | 5.7900 | 5.7900 | 5.7900 |

| Method | AP small | AP medium | AP large |
| --- | --- | --- | --- |
| Dense | 1.0088 | 15.3995 | 53.4577 |
| Recovery control | 0.7807 | 13.5934 | 51.3527 |
| Recovery + 3x3 flow | 0.7805 | 13.5939 | 51.3986 |
| V3 low-stage IoU 0.50 | 0.7702 | 13.5840 | 51.3354 |

For the common strong-seed cohort, previous FULL and current Dense must both match the provided GT identity at TP75 and confidence≥0.25. This cohort is identical across the original Fixed-2-frame arms; it does not represent all objects.

| Arm | Eligible GT instances | TP50 | TP75 |
| --- | --- | --- | --- |
| Recovery control | 5498 | 5489 | 5308 |
| Recovery + 3x3 flow | 5498 | 5493 | 5326 |
| V3 low-stage IoU 0.50 | 5498 | 5489 | 5308 |

Interpretation: confidence/birth changes add or remove low-score evidence; source TP/FP event counts expose their cost. Stricter recovery also loses valid low-score confirmations, not just background. The sparse-grid change modifies box sampling; paired identities and the strong-seed cohort show whether localisation/retention changes rather than a score-only effect. Flow rejection and centre/width/height/IoU summaries are retained for every arm in diagnostics. These conditional geometry statistics are not an unbiased dataset-wide error decomposition, and AP contributions are not additive.

Grid3 rejects 286 of 15,095 flow track updates versus 212 of 15,082 with 4x4. It also changes later association/retention (3,665 versus 3,647 low-score confirmations). Thus this is not a pure reduction in arithmetic with unchanged decisions: spatial sampling, the relative valid-point requirement and downstream history all change. The common strong-seed TP75 gain supports a small geometry/retention benefit, but does not isolate which sampling/acceptance mechanism caused it. No extra minimum-point sweep was run to force an explanation.

## Recommended final candidate and limits

**Practical development candidate:** anchored Fixed-2 + existing-track recovery, flow_grid=3, flow_min_points=4. Keep track_min_conf=0.25, association_low_conf=0.10, low_score_association=True, assoc_iou=0.30 and conf_decay=0.93. Keep stock YOLOv8n and all historical inference settings. New track_birth_conf remains None; low_association_visible_only=False, low_association_iou=None and threshold_aware_assignment=False. Quality-aware flow, score boosting, camera/appearance additions, ROI inference and new scheduling are not enabled. The complete candidate_config.json is saved alongside this report; no existing preset/default is silently switched.

This candidate has the best measured practical combination of approximately retained/slightly improved accuracy, better confidence 0.25 TP/FP, unchanged detector requests and much less flow-point work. Repeated cache timings above price the observed processing benefit and its variability. It is **not** a strict all-metric winner because of the small low-threshold FP-count increase, nor a proven end-to-end acceleration. The absolute detector accuracy remains weak. Do not deploy or claim generalization based only on these five already-analysed sequences.

**Precision-oriented alternative:** V3 low-stage IoU0.50 reduces low-score FPs and internal tracking work, with a measured AP/AR and low-score-recall loss. If the application explicitly accepts that trade-off it deserves consideration. Its actual repeated runtime and its interaction with grid3 remain unverified. A V3+grid3 combination would be a new conditional experiment requiring approval; it was not run and no additive gain is predicted.

Do not adopt unrestricted low-score births, active-only recovery, or cardinality-first assignment merely because they are plausible research transfers. Do not introduce a learned scheduler, detector replacement, retraining, score promotion or extra geometry machinery now. No such complexity earned a measured unqualified benefit here.

Research context: [ByteTrack's paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) and [reference implementation](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py) motivate separating high-score births from lower-score continuation and restricting its second-stage association to active tracks. Our negative active-only result shows that this restriction does not automatically transfer from MOT to frame-aligned YOLOv8n detection AP. [OC-SORT](https://arxiv.org/abs/2203.14360) supports preserving detector observations; the existing simpler anchoring was retained. [Detect or Track](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/02/AAAI-LuoH.pdf) motivates judging refresh value, but does not justify a trained scheduler here. [Towards Streaming Perception](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470460.pdf) explicitly includes output latency; our COCO AP/cache costs are not streaming AP. These papers motivate mechanisms, not expected numerical gains in this experiment.

Remaining uncertainty includes statistical stability, new-sequence generalization, annotation identity fragmentation, original detector truncation/confidence/class errors, input-resolution causality, actual model+tracking contention, deployed latency/throughput and energy. D/E and Phase 4 remain unexecuted. No new data was downloaded.

## Implementation, provenance and reproduction

Only [PipelineConfig opt-ins](../extracted/adaptive_pipeline/shared/config.py) and [Tracker association/birth logic](../extracted/adaptive_pipeline/tracking/tracker.py) changed relative to the pre-Phase 3 production snapshot. Historical defaults and controls reproduce. New [synthetic tests](../tests/test_phase3_tracking.py) cover independent admission, no duplicate births, hidden-track miss bookkeeping, low/high-stage gating, class/empty/rectangular assignment, objective counterexamples and native/forced fixed fast-path parity. All 84 tests pass. Experimental opt-ins remain disabled by default; the recommended configuration uses the original association algorithm.

The [execution driver](../analysis/argoverse_hd_phase3/run.py), observational adapter, cache-only timing, paired analysis/report and final-verification helpers are isolated under analysis/argoverse_hd_phase3. The executed source snapshot, original production copies, full pre-existing working-tree diff, all config/schedule specifications and file hashes are preserved in ../runs/argoverse_hd_phase3_v1. Production algorithm source stayed fixed across all scored and timed runs. A stage cannot silently overwrite completed work.

An interruption exposed an external repository update: three earlier Markdown reports changed only absolute/relative links and repository-root wording. The timing guard stopped before any replay. Exact inverse normalization reconstructed each original document hash; all executed code and 635 archived Phase 2 artifact hashes remained unchanged. Both document versions and the explicit manifest revision are preserved in [document_portability_revision.json](../runs/argoverse_hd_phase3_v1/document_portability_revision.json). The changed documentation and external Git commit were preserved. No completed experiment was repeated, and no commit/push was made by this agent. A denied read-only process-list query was unnecessary; completion markers resolved execution state.

Commands executed from the repository root:

```sh
PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -q
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage prepare
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage controls
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage variants
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage select
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage efficiency
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.run --stage timing
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.analyze
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.report
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase3.verify
```

The first timing invocation was stopped by the documented report-hash guard; the successful invocation performed six new cache-only repetitions. No scoring/replay stage was restarted. Scored archive resumability was available but unnecessary. The run completed 18 of the approved maximum 22 replays because no new confidence arm passed the pre-specified screen. Timings/outputs are fresh; original dataset and experimental results remain untouched.

Artifacts: [summary.csv](../runs/argoverse_hd_phase3_v1/diagnostics/summary.csv), [full paired/event diagnostics](../runs/argoverse_hd_phase3_v1/diagnostics/summary.json), [selection gates](../runs/argoverse_hd_phase3_v1/selection.json), [full candidate config](../runs/argoverse_hd_phase3_v1/candidate_config.json), [timing_summary.json](../runs/argoverse_hd_phase3_v1/timing_summary.json), [final audit](../runs/argoverse_hd_phase3_v1/final_verification.json) and [artifact hashes](../runs/argoverse_hd_phase3_v1/artifact_sha256.json). Per-arm scoring/metrics.json contains aggregate/class/size/sequence scores; official match identities and tensors, predictions, details, tracker states, source links, events and per-frame timing are retained. Full source and test snapshots accompany the audit. Reproduction requires a newly specified fresh destination/plan; completed stages refuse overwrites.
