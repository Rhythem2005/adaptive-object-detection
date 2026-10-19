# Phase 2: measured performance bottlenecks

Completed on 19 October 2026 using the approved A/R/B/C scope. No new YOLO inference, model loading, training, D/E run, acceleration port or Phase 3 implementation occurred. The original detector, tracker, flow and scheduling source files remain unchanged.

The results support **confidence lifecycle handling on top of an anchored fixed schedule** as the next research direction. They do not support a more complicated scheduler or treating flow quality as box correctness. Lower-score observations recover genuine AP, but also propagate more false positives; the preferred deployment threshold and accuracy/cost tolerance remain unspecified.

## Validity and scope

Experiment R passed before dependent experiments: every prediction array, decision and prediction-detail array reproduces bitwise for all four original temporal runs on all 3,416 frames. Track/output counts and ages also agree. Official COCO aggregate metrics reproduce Phase 1 to absolute tolerance 1e-14. Frozen-schedule C controls reproduce the corresponding natural anchoring and quality-aware runs as well.

There were 21 complete cached replays (71,736 frame steps), plus saved-output scoring/analysis and visual review. Every FULL request returned the original stock-YOLOv8n Dense prediction for that validated image. These are simulated detector requests, not new detector executions. Torch/Ultralytics imports are blocked in the execution driver, crop/warmup calls raise errors, and source/image/cache hashes are checked. State resets occur at each of the five original sequence boundaries. No new data or pseudo-labels were produced.

The scorer reuses pinned pycocotools 2.0.11 and Phase 1's positive annotation-ID conversion, original-image coordinates, eight-class taxonomy, crowd policy and standard maxDets=[1,10,100]. AP means COCO AP50–95; AR100 is averaged over classes and IoUs. Per-sequence scores are evaluated independently. All score entries below are percentages; deltas are percentage points. Aggregate AP is not an average of sequence APs.

All replay components execute on the original CPU NumPy/SciPy/OpenCV path. Detector inference device is **not applicable** to cached execution. This phase does not implement the cancelled GPU proposal or make a GPU speed claim.

## B: controlled Fixed-2 results

All seven arms use exactly 1,710 FULL requests and 1,706 SKIPs, saving 49.94% of requests relative to Dense. FULL output is the same cached detector output in every arm; only the declared tracking configuration changes. Dense's prior measured AP is included as a reference.

| Arm | AP50 | AP50–95 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| Dense reference | 23.87 | 15.73 | 15.65 | 23.19 |
| T0 Fixed-2 original | 20.81 | 14.02 | 14.00 | 18.09 |
| T1 + anchoring | 20.84 | 14.24 | 14.49 | 18.26 |
| T2 + quality gates | 20.86 | 14.27 | 14.52 | 18.28 |
| T3 + existing-track recovery | 21.87 | 14.67 | 14.76 | 19.00 |
| T4 shared threshold 0.10 | 22.72 | 15.09 | 15.12 | 20.01 |
| T5 no score decay | 20.86 | 14.25 | 14.48 | 18.26 |
| T6 hold boxes | 20.48 | 13.54 | 13.32 | 17.73 |

![Accuracy and false-positive trade-off](../runs/argoverse_hd_phase2_diagnostics_v1/figures/confidence_tradeoff.png)

T1 improves over the original Fixed-2 by 0.2226 points at identical detection times. T3 adds 0.4292 points over T1; T4 adds 0.8485. T4 reaches 95.92% of Dense AP with half the requested detector calls, leaving an absolute AP gap of 0.6416 points. This is retention of a weak absolute baseline, not high absolute detection accuracy.

T3 is a two-stage confirmation experiment: high-confidence births remain 0.25, and observations in [0.10,0.25) can confirm existing unmatched tracks. **T4 is not a pure birth-only intervention:** track_min_conf is shared by first-stage association and spawning, so lowering it to 0.10 changes both. Its additional gain cannot be attributed exclusively to new births. No extra unapproved arm was added to remove that confound.

T2's gain over T1 is only 0.0293 points. T5 changes AP by +0.0053 points with identical internal tracker states and output geometry/classes. Its different fixed-threshold counts are score calibration effects, not newly located objects. T6 reduces tracking work but loses 0.7033 AP points: optical flow is useful here.

### Reliability at stated confidence thresholds

Below are official TP/FP counts on the same 1,706 SKIP frames at IoU0.50. Ignored/capped predictions are excluded according to COCO. These counts describe operating points, not AP, and all classes are pooled.

| Arm | TP ≥0.001 | FP ≥0.001 | Precision ≥0.001 % | TP ≥0.25 | FP ≥0.25 |
| --- | --- | --- | --- | --- | --- |
| T0 Fixed-2 original | 7739 | 2387 | 76.43 | 7486 | 2013 |
| T1 + anchoring | 7755 | 2392 | 76.43 | 7492 | 2024 |
| T2 + quality gates | 7765 | 2381 | 76.53 | 7502 | 2013 |
| T3 + existing-track recovery | 9076 | 4550 | 66.61 | 7492 | 2024 |
| T4 shared threshold 0.10 | 10440 | 10791 | 49.17 | 7491 | 2025 |
| T5 no score decay | 7755 | 2392 | 76.43 | 7755 | 2392 |
| T6 hold boxes | 7579 | 2564 | 74.72 | 7335 | 2175 |

T3 and T4 improve low-score recall/AP, while the 0.25 operating point is essentially unchanged. T4 more than quadruples SKIP false positives relative to T1 at the retained-score floor. That is a real trade-off, even though Dense itself emits many low-score false positives. Lowering a threshold is not evidence that the additional outputs are uniformly reliable. Do not promote their confidence merely to make more boxes pass 0.25.

Across all frames, fixed-confidence0.25 class-macro statistics are:

| Arm | Macro precision % | Macro recall % | Macro F1 % |
| --- | --- | --- | --- |
| T0 Fixed-2 original | 64.99 | 17.54 | 24.35 |
| T1 + anchoring | 66.25 | 17.57 | 24.41 |
| T2 + quality gates | 66.29 | 17.58 | 24.43 |
| T3 + existing-track recovery | 66.25 | 17.57 | 24.41 |
| T4 shared threshold 0.10 | 66.24 | 17.57 | 24.41 |
| T5 no score decay | 65.27 | 17.90 | 24.74 |
| T6 hold boxes | 63.91 | 17.37 | 24.07 |

T5 raises thresholded recall and FP together using unchanged geometry. This distinguishes a metric/operating-point effect from recovery of missing detections.

### Per-sequence and object-size results

| Sequence | T0 | T1 | T2 | T3 | T4 | T5 | T6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sequence_00 | 6.47 | 6.61 | 6.67 | 6.75 | 6.89 | 6.59 | 6.41 |
| sequence_01 | 8.13 | 8.48 | 8.53 | 8.82 | 9.10 | 8.50 | 7.38 |
| sequence_02 | 12.62 | 12.80 | 12.80 | 13.00 | 13.32 | 12.78 | 12.06 |
| sequence_03 | 23.64 | 25.51 | 25.56 | 26.13 | 27.24 | 25.53 | 19.52 |
| sequence_04 | 22.88 | 22.88 | 22.88 | 24.00 | 24.31 | 23.08 | 22.88 |

T3 and T4 improve AP over T1 on all five sequences in this diagnostic set. This is descriptive consistency, not a held-out or statistical-significance claim. Class support differs across sequence APs.

| Arm | AP small | AP medium | AP large |
| --- | --- | --- | --- |
| Dense | 1.01 | 15.40 | 53.46 |
| T1_fixed2_anchor | 0.73 | 13.14 | 49.95 |
| T3_existing_track_recovery | 0.78 | 13.59 | 51.35 |
| T4_low_birth_threshold | 0.81 | 14.47 | 51.92 |

Small-object detection remains poor even with T4. The experiments do not isolate resolution or original inference truncation; D was not run. Full class AP is:

| Class | Dense | T1 | T3 | T4 |
| --- | --- | --- | --- | --- |
| person | 16.41 | 14.52 | 14.87 | 15.33 |
| bicycle | 0.04 | 0.02 | 0.02 | 0.03 |
| car | 28.25 | 25.59 | 26.38 | 26.79 |
| motorcycle | 4.90 | 2.69 | 3.52 | 3.95 |
| bus | 54.90 | 52.59 | 53.39 | 54.42 |
| truck | 9.18 | 7.52 | 7.80 | 8.11 |
| traffic_light | 6.11 | 5.41 | 5.60 | 5.80 |
| stop_sign | 6.08 | 5.59 | 5.79 | 6.29 |

## A: separate baseline misses from SKIP losses

On the 1,627 quality-aware SKIP frames there are 26,635 regular targets. Pairing official GT match identities gives:

| Threshold | Both match | Neither matches | Dense only | Adaptive only |
| --- | --- | --- | --- | --- |
| IoU50_conf0.001 | 7201 | 12203 | 7184 | 47 |
| IoU50_conf0.25 | 6685 | 18956 | 668 | 326 |
| IoU75_conf0.001 | 5201 | 18791 | 2429 | 214 |
| IoU75_conf0.25 | 4976 | 20733 | 595 | 331 |

The all-score gap is substantial, but most missing detector evidence has low confidence. Among the 7,184 Dense-only TP50 targets, previous-FULL evidence was:

| Previous-FULL evidence | Target instances |
| --- | --- |
| previous_correct_score_10_to_25 | 2437 |
| previous_correct_score_001_to_10 | 3696 |
| previous_correct_score_at_least_25 | 39 |
| no_correct_previous_Dense_match | 986 |
| annotation_identity_absent_previous_FULL | 26 |

Actual tracker event logs show no correct previous detector-row link for 7,145 of these cases; the 39 with such a link still have an internal track, without a correct current official output. The no-link group also includes previous detector misses and absent annotation identities. This is not proof that every missing target could be recovered by tracking, nor an additive AP decomposition. The controlled B interventions establish that confidence handling does contribute to the AP deficit; they do not uniquely allocate the entire deficit among mechanisms.

Twenty-four predetermined diagnostic cases across all five sequences were visually reviewed with immediate-neighbour context (72 panels). They include small/occluded objects, low-score FULL observations absent on SKIP, shifted propagated extents, bus/car/truck and sign/light disagreements, and background light-like responses. These are illustrations, not prevalence estimates or a new annotation audit. No boxes/classes were relabelled. The images show nearest-overlap boxes for clarity, which need not be the internally admitted source row; runtime logs carry that linkage.

## Geometry and confidence lifecycle diagnostics

The following uses one identical GT cohort for every fixed-schedule arm: both previous FULL and current Dense have an official TP75 with confidence≥0.25, joined using the provided annotation identity. It isolates performance where a strong seed and a strong current detector reference exist, rather than averaging over objects never admitted.

| Arm | Eligible GT instances | Adaptive TP50 | Adaptive TP75 |
| --- | --- | --- | --- |
| T0 Fixed-2 original | 5498 | 5480 | 5098 |
| T1 + anchoring | 5498 | 5489 | 5306 |
| T2 + quality gates | 5498 | 5493 | 5324 |
| T3 + existing-track recovery | 5498 | 5489 | 5308 |
| T4 shared threshold 0.10 | 5498 | 5490 | 5309 |
| T5 no score decay | 5498 | 5489 | 5306 |
| T6 hold boxes | 5498 | 5405 | 4811 |

Runtime summaries separately retain hidden/deleted/unemitted states and geometry of uniquely linked emitted tracks. They are conditional identity/geometry diagnostics; annotation IDs may fragment, and unlinked detector FPs are not silently treated as correct geometry. Key event counts are:

| Arm | Spawn events | Births <0.25 | Low-score confirmations | Hidden-track recoveries | Flow rejections |
| --- | --- | --- | --- | --- | --- |
| T0 Fixed-2 original | 1158 | 0 | 0 | 433 | 403 |
| T1 + anchoring | 1168 | 0 | 0 | 419 | 174 |
| T2 + quality gates | 1164 | 0 | 0 | 420 | 271 |
| T3 + existing-track recovery | 755 | 0 | 3647 | 379 | 212 |
| T4 shared threshold 0.10 | 3368 | 2995 | 0 | 1407 | 336 |
| T5 no score decay | 1168 | 0 | 0 | 419 | 174 |
| T6 hold boxes | 1384 | 0 | 0 | 375 | 0 |

Counts are track-update events across sequences, not unique physical objects. Detailed centre, width/height, IoU, age, visibility and flow-quality distributions are in combined_diagnostics. Confidence decay does not delete tracks in this implementation; the score-only control confirms that distinction.

## C: does scheduling improve the trade-off?

All cells below use the same anchored tracker and legacy adaptive execution path, including propagation before FULL. Archived schedules are forced by the analysis harness, while tracker updates remain normal. Uniform U0 and phase-shifted U1 have identical FULL budgets per sequence. Both phases were pre-specified.

| Budget source | FULL requests | Archived AP % | Uniform U0 AP % | Uniform U1 AP % | Archived−U0 pp | Archived−U1 pp |
| --- | --- | --- | --- | --- | --- | --- |
| Original | 1733 | 14.18 | 14.15 | 14.34 | +0.0310 | -0.1589 |
| Anchoring | 1729 | 14.16 | 14.11 | 14.33 | +0.0537 | -0.1694 |
| Quality-aware | 1789 | 14.21 | 14.38 | 14.18 | -0.1642 | +0.0354 |

These controls test allocation given an archived per-sequence budget and a common engine. Their total budgets are known offline; they are not newly validated online schedulers. A small gain over one phase is insufficient evidence of a superior adaptive policy. The original native Fixed-2+anchoring control is also competitive with fewer calls. Its early-FULL path differs from C, so comparing it with C is a practical system comparison, not a pure schedule intervention.

The schedule × quality-flow factorial is:

| Engine | Archived anchoring schedule (1729) | Archived quality schedule (1789) |
| --- | --- | --- |
| Anchoring only | 14.16 | 14.21 |
| Anchoring + quality gates | 14.19 | 14.21 |

With the quality schedule fixed, adding quality gates changes AP by +0.000000 points. With the anchoring schedule fixed, the change is +0.029444. Changing the schedule without quality gates adds +0.048464 points and 60 calls. The interaction is -0.029444 points. These effects depend on the reference schedule; AP is nonlinear. The historical incremental quality-aware result is therefore not evidence of a generally improved accepted-box estimator.

Same-history pre-FULL shadow outputs are compared with Dense, with initial FULL frames excluded. Correlations use **pre-decision** states, not post-FULL track counts that would leak detector information. Immediate TP/FP changes, including negative benefits and both selected FULL/SKIP cohorts, are retained in per-frame utility CSVs. These are not downstream causal effects or streaming accuracy. Neither motion/flow consistency nor a current detection miss alone establishes the value of another detector call.

At confidence 0.25 and IoU0.50, the immediate mean changes per frame are:

| Schedule | Decision | Frames | Dense − propagated TP/frame | Dense − propagated FP/frame |
| --- | --- | --- | --- | --- |
| Original allocation | FULL | 1728 | +0.380 | +0.321 |
| Original allocation | SKIP | 1683 | +0.191 | +0.228 |
| Quality allocation | FULL | 1784 | +0.345 | +0.324 |
| Quality allocation | SKIP | 1627 | +0.210 | +0.230 |

Selected FULL frames have greater immediate mean TP benefit, but also introduce more FP. Median TP benefit is zero in all four cohorts. This shows some targeting, not a validated optimal allocation or a downstream accuracy gain. For the quality allocation, Spearman correlations with immediate TP benefit are age +0.114, novelty +0.024, failure fraction −0.037, flow quality +0.006 and uncertainty +0.016. These weak marginal associations do not rule out nonlinear signals; temporal dependence and selection remain.

## Computational costs and measurement limits

| Arm | Mean internal tracks | Initial LK points total | Replay step wall ms | Replay step CPU ms | Diagnostic callback wall ms |
| --- | --- | --- | --- | --- | --- |
| T0 Fixed-2 original | 6.98 | 380944 | 3.217 | 5.453 | 0.212 |
| T1 + anchoring | 6.98 | 190464 | 2.729 | 4.574 | 0.191 |
| T2 + quality gates | 6.98 | 190416 | 2.840 | 4.673 | 0.199 |
| T3 + existing-track recovery | 8.84 | 241312 | 2.882 | 4.725 | 0.214 |
| T4 shared threshold 0.10 | 15.67 | 427680 | 2.990 | 5.044 | 0.233 |
| T5 no score decay | 6.98 | 190464 | 2.702 | 4.531 | 0.191 |
| T6 hold boxes | 7.08 | 0 | 1.513 | 1.588 | 0.181 |

There were 36,492 simulated FULL cache lookups and **zero** new YOLO executions. Summed recorded replay-stage wall time across 21 arms was 676.1s, including image decoding/hashing, diagnostic serialization and archive writing. Summed per-arm scoring wall time was 540.0s, including aggregate and independent per-sequence COCO evaluation. These sums exclude preparation, A's Dense rescoring, manual rendering and subsequent combined analysis, and are not an end-to-end benchmark runtime.

Replay-step times subtract measured diagnostic callback time; residual wrapper/cache overhead remains. They exclude decode, hashing, event serialization, scoring and model execution. This is one sequential replay per arm, without timing counterbalancing or live detector contention. CPU time can exceed wall time due to library worker threads. Small timing differences are not proven speedups. Initial LK point totals count grid points passed to forward flow; the backward check also runs.

T3/T4 increase state and point work while preserving the same detector-request budget. The high-confidence admission baseline deliberately tracks far fewer boxes than Dense emits. Lowering admission can narrow the accuracy gap but weakens that tracking-efficiency advantage. Actual combined YOLO+tracking FPS, hardware energy and MPS performance were not remeasured; E was not run. The existing GFLOPs formula covers detector arithmetic only and cannot price this additional tracking work.

## Evidence-based Phase 3 recommendation

1. **Use anchored Fixed-2 as the required working control.** It improves localisation at the same requests; a more complicated scheduler has not established a robust advantage over the pre-specified uniform controls or this practical reference. Preserve stock YOLOv8n.
2. **Investigate continuation separately from births.** T3 is the smaller existing mechanism to carry forward for validation: high-confidence births with lower-score confirmation. T4 demonstrates further headroom but changes association too, increases false outputs and more than doubles internal tracks. It is not a justified blanket deployment threshold. A future birth-only intervention must freeze the high-stage association rule, with an explicit added-track/FP budget.
3. **Treat score calibration and detection reliability separately.** T5 can move boxes across 0.25 without improving geometry. Agree on an application operating point and acceptable miss/false-alarm costs; do not select a policy by AP or thresholded F1 alone. Validate confidence handling on new approved data before deployment.
4. **Keep flow and anchoring; do not replace the tracker or add a scheduler network now.** Holding boxes costs accuracy, and quality gates add little conditional AP. Only pursue a more complicated flow/scale/occlusion model if the linked residual errors justify its cost. No camera/appearance/ROI/CLAHE sweep is supported by these experiments.
5. **A scheduler must predict useful refresh benefit at matched total cost.** Cheap signals should be validated against unseen-object/retention/localisation risk, including invisible low-score evidence, rather than only health of already admitted tracks. Frozen-schedule results are diagnostic; a new prospective closed-loop controller requires separate approval and comparison with both periodic phases. Do not claim superiority from extra calls.

These priorities follow mechanisms rather than transplanting complete systems. [ByteTrack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) motivates continuation from lower scores while preserving high-score births; its MOT results do not promise detection AP gains here. [OC-SORT](https://arxiv.org/html/2203.14360v3) supports observation-centric state correction, while our anchoring is a simpler existing mechanism. [Detect or Track](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/02/AAAI-LuoH.pdf) motivates measuring actual tracking quality and refresh utility; its learned scheduler/tracker is a larger trained system and was not adopted. [Towards Streaming Perception](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470460.pdf) separates frame-aligned AP from latency-aware streaming AP; no streaming result is claimed in this phase.

Remaining uncertainty: original pre-LUT/global top300 losses, causal resolution effects, detailed annotation quality, physical identity through annotation fragmentation, true live-inference total cost, and generalization beyond five already-analysed sequences. No held-out accuracy, statistical significance, calibration guarantee, energy saving or expected Phase 3 AP gain is claimed. D/E and Phase 3 implementation remain unexecuted.

## Root causes ranked by evidence and impact

1. **Weak detector evidence limits absolute performance (confirmed).** Dense AP is 15.73%, with small-object AP 1.01% versus large-object AP 53.46%. Adaptive tracking cannot reliably create new class evidence for objects the detector never identifies. These size strata establish a disparity, not the causal effect of resolution; no resolution sweep was approved. Dense is a reference, not a mathematical upper bound on temporal AP.
2. **Confidence admission/continuation loses usable evidence (confirmed intervention).** T3/T4 recover AP at unchanged detector requests; this is larger than either quality gating or score decay. Their added false positives and tracking work constrain useful gains. T4 cannot separate births from first-stage association.
3. **Refresh allocation has no robust advantage over periodic controls (confirmed on this subset).** Each archived adaptive schedule beats one budget-matched phase and loses to the other. Weak marginal signal/benefit correlations support diagnosing the objective before adding controller complexity; they do not prove all adaptive scheduling is ineffective.
4. **Propagation geometry matters, but is not the only or dominant recoverable issue (supported).** Anchoring improves the common strong-seed cohort at TP75; holding boxes hurts. Quality gates add only small conditional gains. This does not exclude geometry failures on difficult/low-score objects outside the cohort.
5. **Score decay has negligible AP impact here (confirmed control).** Removing it leaves state/geometry unchanged and adds 0.0053 AP points. It changes a chosen confidence operating point, so promoting scores is not a substitute for recovering detections.

## Changes and reproducibility

New files are confined to [the analysis harness](../analysis/argoverse_hd_phase2/run.py), its cache/observation, scoring, pairing, summary/report and rendering helpers, [synthetic harness tests](../tests/test_phase2_harness.py), and this report. The frozen plan, existing dataset, original benchmark, Phase 1 outputs and production algorithm code remain intact. All 73 tests passed, including 5 new cache/wrapper/official-match tests. Two test-development failures (an incorrect unittest module invocation and an assertion looking for a birth on the final SKIP frame) were corrected before any dataset replay; they did not alter algorithms or benchmark results.

Implementation evidence: [FULL admission, two-stage association and spawning](../extracted/adaptive_pipeline/tracking/tracker.py:224) establishes the shared threshold and T3 semantics; [SKIP outputs](../extracted/adaptive_pipeline/tracking/tracker.py:264) applies decay to scores without deleting state. [fixed anchored early FULL](../extracted/adaptive_pipeline/pipeline.py:99) explains its reduced flow work relative to the legacy adaptive path. [legacy controller](../extracted/adaptive_pipeline/adaptive_framing/controller.py:93) computes health/refresh eligibility on visible admitted tracks; its uncertainty trigger uses detector confidence and age. [cache and observation wrappers](../analysis/argoverse_hd_phase2/replay.py:59), [official scoring adapter](../analysis/argoverse_hd_phase2/scoring.py:11) and [same-history utility analysis](../analysis/argoverse_hd_phase2/analyze.py:141) provide the diagnostic implementation. Exact source snapshots accompany the run.

The approved plan is [plan.json](../analysis/argoverse_hd_phase2_plan_v1/plan.json). Exact full configs, archived/uniform schedules and source hashes are retained there and copied into the new run. The execution driver enforces R completion before A/B/C. The primary output is [summary.csv](../runs/argoverse_hd_phase2_diagnostics_v1/combined_diagnostics/summary.csv), with per-arm COCO metrics, per-sequence/class/size scores, official GT match identities, prediction archives, tracker states/hashes, event logs and per-frame timing/utility data. The manual review is documented in [review_notes.json](../runs/argoverse_hd_phase2_diagnostics_v1/A_saved_pairing/manual_visuals_v2/review_notes.json).

The independent [final archive audit](../analysis/argoverse_hd_phase2/verify.py) checks saved arrays, official tensors, aggregate/class/sequence scores, exact forced schedules, every FULL cache output, production snapshots and all 71 frozen source-evidence hashes. It reruns synthetic tests only. The machine-readable audit is [final_verification.json](../runs/argoverse_hd_phase2_diagnostics_v1/final_verification.json); artifact hashes are in [artifact_sha256.json](../runs/argoverse_hd_phase2_diagnostics_v1/artifact_sha256.json).

Commands executed from the repository root:

```sh
PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -q
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.run --stage R --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.run --stage A --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.run --stage B --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.run --stage C --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.manual_review --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.analyze --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.report --out runs/argoverse_hd_phase2_diagnostics_v1
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_phase2.verify --out runs/argoverse_hd_phase2_diagnostics_v1
```

The initial visual render was retained separately and re-rendered with clipped labels for readability; no experiment was rerun. Fresh directories are required. The one-off driver intentionally binds to the approved plan/output path and refuses completed stages; reproduction in another destination requires a newly specified plan/output, not overwriting these results. Runtime/backend, source snapshots and final verification records accompany the outputs. No commit or push was made.
