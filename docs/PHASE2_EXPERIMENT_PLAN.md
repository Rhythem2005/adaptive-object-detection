# Phase 2: bottleneck investigation and proposed experiments

Prepared 19 October 2026. **Awaiting approval. No new scoring, image experiment, pipeline replay, YOLO inference, test run or algorithm change was executed in this phase.** This document reviews existing evidence and specifies experiments. Phase 2's causal validation remains pending; Phase 3 has not started.

## Current assessment

The best-supported direction is to investigate **which detector observations survive into SKIP outputs, at what tracking cost**, before designing a more complicated scheduler. Keep stock YOLOv8n and a strong fixed-schedule control. Treat input resolution as a separate detector diagnostic, not a presumed efficiency improvement.

The [verified Phase 1 results](COCO_PHASE1_RESULTS.md) establish the following on 3,416 frames:

| Method | COCO AP50–95 % | AP50 % | Detector calls | Call savings % |
|---|---:|---:|---:|---:|
| Dense | 15.7327 | 23.8650 | 3416 | 0.00 |
| Fixed-2 | 14.0200 | 20.8149 | 1710 | 49.94 |
| Original adaptive | 13.9814 | 20.6985 | 1733 | 49.27 |
| Adaptive + anchoring | 14.1642 | 20.6936 | 1729 | 49.39 |
| Adaptive + quality-aware flow | 14.2126 | 20.6943 | 1789 | 47.63 |

These are existing measurements, not new Phase 2 results. Standard COCO correction raised AP by about 0.24 points without changing the ranking. Quality-aware adaptive gains 0.1926 AP points over Fixed-2 while spending 79 more detector calls; original adaptive spends 23 more calls and scores 0.0386 points less. Neither comparison isolates a superior scheduler at equal compute.

### Ranked bottlenecks and evidence strength

| Priority | Evidence and mechanism | What is established / uncertain | Smallest informative test | Potential benefit |
|---|---|---|---|---|
| 1: detection reliability | Dense AP is 1.01% on small objects versus 53.46% on large objects; bicycle AP is 0.04%. All methods inherit weak FULL observations. | Size-associated weakness is confirmed. Resolution, class semantics, score ranking and domain conditions are not causally separated. | Paired official error analysis (A); only later a controlled 640/960 detector diagnostic (D). | Accuracy on every frame; efficiency only if the extra detector cost is justified. |
| 2: admission and confirmation | FULL emits raw detections down to 0.001; tracking admits detections at 0.25. Failed FULL confirmation immediately hides a track. | Mechanism is confirmed; its contribution to official AP and the FP/cost of admitting more boxes are unmeasured. | Cached fixed-schedule recovery and birth-threshold controls (B). | Recall without more detector calls, potentially offset by FPs and tracking overhead. |
| 3: refresh allocation | Legacy refreshes are largely confidence/age rules, with 1,450 `too_many_refresh` triggers. Most SKIPs are one frame after FULL. | It behaves close to a phase-changing Fixed-2 schedule. A causal advantage from choosing different frames is unproven. | Common tracker, two uniform controls and exactly matched calls per sequence (C). | Better allocation or simpler scheduling; no guaranteed gain. |
| 4: propagation geometry | Anchoring improves stringent-IoU matches in previous diagnostics. Quality-aware and anchoring outputs are identical on 1,417/1,427 shared SKIP frames. | Anchoring is supported; quality-aware flow's incremental geometry effect is confounded with schedule changes and selected histories. | Fixed schedule B and a schedule × quality-flow factorial in C. | Localisation of already admitted objects. |
| 5: total processing cost | The pipeline performs signals/flow/association in addition to YOLO; lower admission thresholds increase that work. | Detector calls are not total compute, time, power or energy. Existing timing has only one repetition. | Component work/timing in replay, then separate real-inference timing (E). | Prevent apparent AP/call gains that lose actual efficiency. |

Sources for the older descriptive counts are [saved diagnostics](../analysis/argoverse_hd_root_cause_v1/saved_output_diagnostics.json), [history and geometry diagnostics](../analysis/argoverse_hd_root_cause_v1/focused_diagnostics.json), and their [previous report](ARGOVERSE_HD_DIAGNOSIS.md). They contain historical custom-match counts and many-to-one coverage analyses. They are not newly verified COCO loss decompositions. Phase 1 is the source of current AP, area AP and AR.

## Assumptions that need correction or qualification

**A low-score association switch is not a complete solution.** [Tracker.update_full](../extracted/adaptive_pipeline/tracking/tracker.py:224) filters high-score detections, optionally associates lower scores to existing unmatched tracks, and only spawns from the high-score set. The earlier analysis found 6,239 low-confidence previous-FULL candidates among 7,266 Dense-only SKIP coverage losses. Of those, 3,924 never had a correct high-confidence observation at any earlier FULL, and 3,782 had the relevant score below 0.10. These sets overlap. A 0.10 confirmation-only option cannot address most of that candidate pool. Even the history counts do not prove actual internal track presence or recoverable AP: they use annotation identities and overlap, not recorded association events.

**Confidence decay does not delete tracks in this code.** [Tracker.outputs](../extracted/adaptive_pipeline/tracking/tracker.py:264) gates visibility using `missed` and `fail_streak`, then multiplies confidence by decay. It applies no confidence deletion threshold. Disabling decay could change ranking or the fixed-0.25 operating point, but cannot directly resurrect missing tracks. This is why B separates scores from admission and geometry.

**Continuous flow quality is not box correctness.** [Flow propagation](../extracted/adaptive_pipeline/tracking/flow.py:91) estimates median translation and one isotropic scale for both dimensions; quality gates filter that estimate. They do not estimate independent width/height changes or certify object identity. [Anchoring](../extracted/adaptive_pipeline/tracking/tracker.py:125) directly emits the accepted measurement, so changing its KF measurement variance may have limited immediate effect on output geometry. It can still affect velocity, later association and failures. Flow should be assessed on failed/hidden tracks too, not only successful emitted boxes.

**Legacy uncertainty is confidence plus age.** [Controller.decide](../extracted/adaptive_pipeline/adaptive_framing/controller.py:93) checks stored score below 0.65 after two frames in the recovered adaptive configurations. It does not use covariance. `use_local=True,max_local_rois=0` turns a refresh need into FULL. Setting `use_local=False` would remove this branch and change the experiment. Risk/budget options exist but were disabled, and the existing budget mechanism is soft rather than a guarantee of equal wall time. No risk-controller sweep is proposed initially.

**The original global detector cap is still unaudited.** [Detector._to_array](../extracted/adaptive_pipeline/detection/detector.py:45) maps/filters classes after model prediction. [Installed NMS](../.venv/lib/python3.14/site-packages/ultralytics/utils/nms.py:125) chooses the best class and later caps detections globally. Phase 1's negligible cap100-versus-cap300 *evaluation* effect does not exonerate the earlier 80-class inference truncation. Unrecorded boxes cannot be recovered from saved predictions. This is a hypothesis, not a diagnosed cap failure.

**Timing needs a defined scope.** The historical quality-aware mean processing latency is 14.87 ms versus Dense 22.12 ms, about 32.8% lower by arithmetic on existing means, despite 47.63% fewer calls. This is not a new performance measurement. [Pipeline timing](../extracted/adaptive_pipeline/pipeline.py:66) excludes decode/scoring/output I/O. [Telemetry cpu_ms_per_frame](../extracted/adaptive_pipeline/telemetry/metrics.py:78) uses process time over a broader loop, which includes such work; process peak RSS is cumulative. These fields must not be treated as equivalent cost measures. No energy saving has been measured.

**Dense is a reference, not a mathematical accuracy ceiling.** Propagation can occasionally cover an object that current-frame Dense misses. More FULL calls need not improve every object or AP. An accurate diagnosis must also retain adaptive-only successes, duplicates and stale/background detections.

## Research that informs the experiments

| Primary reference | Relevant lesson | Application, cost and limitations here |
|---|---|---|
| [Luo et al., Detect or Track, AAAI 2019](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/02/AAAI-LuoH.pdf), §§3–4 | The paper explicitly discusses AP loss from discarding low-confidence boxes, and constructs scheduling labels from tracking overlap and appearing/disappearing objects. | Supports admission and actual refresh-benefit diagnostics. Its learned shared-feature tracker/scheduler requires training and substantial integration; it is not proposed as a replacement. Tracking all low-score boxes could be costly and propagate FPs. |
| [Zhang et al., ByteTrack, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf), §3 / Algorithm 1 | Lower-score observations can recover unmatched existing tracks; new tracks come from unmatched high-score observations. | Direct motivation for separating confirmation from births in B. Existing second association is much simpler than importing ByteTrack, but its extra assignment work must be measured. MOT accuracy/identity results do not establish detection AP gains on this dataset. |
| [Cao et al., OC-SORT, CVPR 2023](https://arxiv.org/html/2203.14360v3), §4 | Detector observations and re-update after reacquisition can repair accumulated filter errors. | Supports testing anchoring and association history, not calling our anchoring full OC-SORT. Full adoption adds trajectory/association logic and cannot retroactively correct already emitted online outputs or create never-detected objects. |
| [Li et al., Towards Streaming Perception, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470460.pdf), §3 | Streaming evaluation queries the latest available output against the world at query time, combining latency and accuracy. | Our frame-aligned COCO AP plus throughput is not streaming AP. First isolate current failures; a later timestamp/output-buffer experiment needs separate approval. Additional instrumentation and scheduling costs must be included. |
| [Kalal et al., Forward–Backward Error, ICPR 2010](https://cmp.felk.cvut.cz/ftp/articles/matas/kalal-2010-fb_track-icpr.pdf) | Forward/backward consistency selects more reliable trajectories. | It motivates point reliability checks, not a guarantee that an entire box is localised correctly. Assess box geometry against annotations independently. Retain the existing estimator during diagnosis. |
| [Bolya et al., TIDE, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480562.pdf) | Error types and their AP effects interact; diagnosis should go beyond one aggregate score. | Motivates labelled error cohorts and explicit rematching. No new TIDE dependency is needed for initial tests. Overlap-based candidate labels will not be presented as a formal TIDE decomposition or additive contributions to AP. |

Dataset history diagnostics also need care: the [authors' format documentation](https://github.com/mtli/sAP/blob/master/doc/data_setup.md) warns of fragmented/noisy track IDs across annotation segments. Noisy GT identities cannot establish physical births or exact tracker identity switches by themselves.

## Proposed initial execution: saved-output analysis and cached replay only

Request approval for A, R, B and C below. **Zero new YOLO inferences.** The replay uses the 3,416 existing images and existing Dense predictions, resets at all five original boundaries, and preserves every frame. The prepared specifications include all source settings and hashes; no algorithms are edited.

Fresh proposed output root: `../runs/argoverse_hd_phase2_diagnostics_v1/`. It has not been created. The harness must reject an existing destination.

### A — Pair official matches and inspect failure cohorts

- Inputs: original five prediction archives and telemetry, provided annotations/manifest, Phase 1 normalized annotation copy/ID map and official match evidence; all frames and classes.
- Use the Phase 1 standard COCO protocol, preserving crowds and all prediction rows. Retain matched GT IDs as well as TP/FP/ignore flags. Use IoU50 and IoU75 for interpretable counts alongside pooled AP50–95.
- Compare Dense and each sparse method on **the exact same SKIP image IDs**. Keep both miss sets, Dense-only and adaptive-only successes, then examine size within each class as well as the joint size/class distribution, confidence bins [0.001,0.10), [0.10,0.25), [0.25,1], and time since FULL. One-to-one matching and crowd rules remain official; auxiliary nearest-overlap classifications are labelled as candidate explanations.
- Revisit admission-related history separately for previous correct low-score observations, detector misses, apparent new annotation identities and strong prior seeds. Do not assume those groups sum to AP loss or that GT-to-prediction overlap proves tracker association.
- Manual audit: at most 100 cases (four per candidate category × five categories × five sequences), deterministically ordered by a hash of sequence/image/case identity. Categories: Dense miss, low-score candidate loss, strong-seed propagation loss, wrong-class overlap, duplicate/stale/background candidate. Inspect up to ±2 frames. Keep sparse strata sparse; do not replace them with favourable cases. This is diagnostic review, not a new annotation set or benchmark selection.
- Outputs: `A_saved_pairing/`, including cohort definitions, denominators, official matches, paired counts, illustrative case IDs and ambiguity notes.

This resolves how much current-frame detection versus SKIP output loss is visible under the verified matcher. Exact internal causes remain for replay.

### R — Reproduce the four existing temporal runs using cached detections

- Four complete passes: `fixed2_093`, `C_fullfallback`, `adaptive_observed`, `adaptive_quality`, using their **saved full configurations**, not current defaults. Dense's original mapped output is returned only on requested FULL frames; record lookups as simulated detector requests, never actual new inference calls.
- No ROI inference is possible. A crop call must fail, not substitute a whole-frame cache or silently disable a controller branch. Preserve the recovered zero-ROI FULL fallback in adaptive runs. No CLAHE/refinement or model warmup/inference occurs.
- Require identical frame inputs, FULL/SKIP decisions, call counts and bitwise prediction arrays against each original run. Stop and investigate on disagreement; do not silently relax tolerances or continue interpreting ablations. Official aggregate metrics must reproduce Phase 1.
- Synthetic adapter/wrapper/scorer tests run only after approval, before real-data replay. The cache adapter must reject wrong image IDs, missing/extra frames, wrong taxonomy and accidental model/crop inference. Instrumentation must not change returned values, state or decisions.
- Outputs: `R_reference_parity/`.

### B — Isolate tracking factors with an identical Fixed-2 schedule

Every arm uses all 3,416 frames and the same FULL frames as recovered Fixed-2. Common settings: mapped Dense cache, `conf_decay=0.93`, `track_min_conf=0.25`, association IoU 0.3, output NMS 0.7, grid flow width 640, no quality-based score decay, no low-score association unless specified. All other settings come from the saved Fixed-2 config. No parameter search follows disappointing results.

| Arm | Change from recovered Fixed-2 | Question |
|---|---|---|
| T0 original | None; reuse the validated R control | Reference. |
| T1 anchor | `observation_anchoring=True` | Does anchoring help at exactly the same detection times? |
| T2 quality | T1 + `quality_aware_flow=True` | Do quality gates improve geometry/retention independently of scheduling? |
| T3 existing-track recovery | T1 + `low_score_association=True,association_low_conf=0.10` | Can lower-score observations confirm existing tracks without low-score births? |
| T4 lower birth threshold | T1 + `track_min_conf=0.10`; low-score association stays off | How much recall is lost at admission, and what FP/track-count/cost penalty results? |
| T5 no score decay | T1 + `conf_decay=1.0` | Is score ranking responsible for any metric change when geometry/track state is held constant? |
| T6 hold | T1 + `propagation="hold"` | How much does flow buy at a one-frame gap compared with a cheap existing control? |

T4 is a diagnostic, not a proposed deployment setting. T6 changes propagation and potentially later association, not just current-frame centre displacement. No comparison will mislabel it as a perfectly isolated displacement estimator test.

Log admission, match/recovery, miss/hide/delete and NMS outcomes, track IDs and source detector rows; record pre/post boxes, scores, age, covariance summaries, flow validity and available spread/FB/dispersion diagnostics. Compare centre displacement normalised by GT diagonal, log width/height ratio errors and IoU against annotation boxes for eligible seeds, including failure/unmatched counts. Separate seed error from incremental propagation error, and report the selected denominators. Do not claim unbiased global geometry accuracy from only emitted boxes.

Record track/point counts and component CPU/wall work, with diagnostic recording/serialization timed separately. Cached replay throughput is **not** deployed pipeline FPS: detector execution, contention and warmup are absent. T5 must keep decisions, geometry and track state identical to T1; otherwise its interpretation fails.

Outputs: `B_tracking_factors/`. Six new full passes plus the reused T0 control.

### C — Separate schedule allocation from propagation quality

Use a common anchored engine from saved `adaptive_observed`, with `policy=adaptive,controller_mode=legacy`, and a harness-only controller that forces a prescribed FULL/SKIP sequence. All other detector/tracker settings stay fixed. Each FULL updates normal tracker/controller state. No future GT is used for decisions.

Three archived schedules are compared with **two** deterministic uniform controls at exactly the same FULL counts in each sequence:

| Schedule budget | seq00 | seq01 | seq02 | seq03 | seq04 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Original adaptive | 466 | 438 | 235 | 259 | 335 | 1733 |
| Anchoring adaptive | 466 | 435 | 234 | 259 | 335 | 1729 |
| Quality-aware adaptive | 486 | 468 | 237 | 263 | 335 | 1789 |

For length N and budget B, U0 uses `floor(j*N/B)`, j=0..B−1. U1 starts with frame0, then uses `floor((j+0.5)*N/B)`, j=1..B−1. Two phases reduce dependence on one accidental alignment. These are explicit offline count-matched controls; knowing an archive-derived per-sequence budget does **not** make them a deployable online scheduler. Equal calls also do not guarantee equal tracking cost.

Nine cells: common anchored engine × (three archived schedules and six corresponding uniform controls). Add two cells with quality-aware flow on the archived anchoring and quality schedules. The latter complete a 2×2 schedule × propagation-quality comparison because the common-engine cells already contain both schedules. Total: **11 full passes**. Keep the original Fixed-2 reference as a separate practical control.

Important implementation constraint: [pipeline early FULL handling](../extracted/adaptive_pipeline/pipeline.py:99) skips flow on scheduled FULL frames for anchored fixed policies. C uses the **same legacy adaptive execution path** for every cell, preserving propagation before FULL. Otherwise changing the schedule path would also change the internal pre-detection state. Forced archived cells must reproduce corresponding validated natural runs where configurations agree.

Capture non-mutating pre-FULL shadow outputs before detector confirmation, and compare the actual detector output with that same-history alternative at IoU 50/75 (both TP and FP changes). This estimates immediate refresh benefit, not total downstream benefit or additive AP loss. A fixed-policy FULL fast path that omitted flow is not a valid SKIP counterfactual without a separate cloned computation; do not mislabel it.

Assess trigger/flow/age/size signals against these observed benefits across the full cohort, not only selected SKIP frames. Never give Dense detections or GT to a tested decision rule on SKIP. Report the official AP and per-sequence deltas of the complete forced schedules. The quality factorial identifies conditional effects and interactions, not a unique additive decomposition of nonlinear AP. Frozen schedules establish allocation effects given a common tracker; they do not validate a redesigned closed-loop scheduler.

Outputs: `C_schedule_factors/`. No risk-policy/network training, new scheduling algorithm or detector inference.

## Common validity and decision rules

- Pin Phase 1 `pycocotools==2.0.11`, positive annotation IDs in a copy, standard maxDets100 and unchanged taxonomy/coordinates. Extend the analysis scorer to arbitrary arm names; the current five-method CLI must not be tricked by renaming arms as old methods. Preserve historical metrics separately.
- Primary comparison: pooled COCO AP50–95 against simulated detector request count and tracking work. Also report AP50/75, AP small/medium/large, AR100, class/sequence results, exact-confidence 0.25 P/R/F1, FP and miss persistence, duplicates, hiding/recovery rates and first-detection delay where annotation identity is sufficiently reliable.
- Do not improve apparent AP by dropping difficult classes, frames or low scores from evaluation. A score-only gain is not evidence of more reliable geometry or discovery.
- Report all five sequence deltas and class support. These five sequences have already been analysed and are development data, not a fresh held-out set. Do not treat 3,416 correlated frames as independent samples or declare significance from tiny differences. No formal noninferiority claim or universal AP-loss tolerance is assumed.
- Prefer a Pareto improvement; if accuracy and cost trade off, report the trade-off explicitly. A Phase 3 candidate should outperform a corresponding fixed control or offer a clearly justified cost/reliability benefit. More complexity must earn its cost. If fixed scheduling remains strongest, retain it.
- Hash data, caches, configurations and source snapshots before/after; save fresh per-arm evidence and failure logs. Record no-inference status explicitly. Any disagreement that threatens validity stops dependent experiments.

## Conditional follow-ups — not included in the initial approval request

**D: paired resolution/raw-output diagnostic.** A prepared manifest selects exactly 64 evenly spaced frames from each sequence (320 total), using `fid=(j*(N−1)+31)//63`, j=0..63. It includes image IDs, original paths and existing image hashes. The frame selection is fixed before any new inference. These isolated frames are suitable for Dense resolution comparison, not temporal replay.

If separately approved: stock checkpoint at 640 and 960, CPU FP32, confidence 0.001, NMS 0.7, original global max_det 300, same class map and no augmentation/CLAHE/crops. Run both on the same selected frames: **640 measured forward calls**, plus three excluded full-frame warmups per size (six). Retain 80-class post-NMS outputs before LUT and saturation information through a diagnostic wrapper; do not change NMS or caps. Verify the new 640 output against the corresponding saved Dense rows first. If parity fails, diagnose it before attributing differences to resolution. Raw pre-cap recording would require a separate semantics-preserving validation; without it, saturation is only evidence of a possible cap effect, not a quantified loss.

Fresh proposed outputs: `runs/argoverse_hd_phase2_resolution_v1/{imgsz640,imgsz960}/`. No 1280 arm or class-filter/cap sweep initially. If saturation matters, specify a separate experiment rather than combining it with resolution changes.

The existing padded-area cost formula predicts about **2.19× detector arithmetic** for 960×608 versus 640×416. At half as many calls, this would still be about 1.10× Dense at 640 detector arithmetic before tracking overhead. Approximately 77% fewer 960 calls would be needed to halve Dense at 640 detector arithmetic. These are estimates from the current cost model, not measured latency or energy. A resolution gain alone may therefore be incompatible with the efficiency objective.

**E: real-inference processing-cost validation.** Specify after A–C identify meaningful controls/candidates, then obtain separate approval. Use the same stock checkpoint, device and inference settings for every compared method, repeated runs with counterbalanced method order, equal warmup policy, and both pipeline-only and decode-inclusive timings. Record thread/device/runtime settings and separate diagnostic overhead. Use actual device synchronisation if a GPU is selected. Historical MPS availability was false; cached CPU replay says nothing about future GPU speed. Any MPS change requires its own compatibility check and a newly measured Dense control, not a comparison against historical CPU timings. Streaming AP requires a further timestamped-output protocol; it cannot be manufactured from cached replay FPS.

## How results would determine the Phase 3 direction

These are conditional decisions, not promised outcomes or permission to implement improvements:

| Result of Phase 2 validation | Direction worth considering next |
|---|---|
| T3 gains recall/AP with limited added FP and association work | Retain high-confidence births and validate recovery of existing tracks; a full tracker replacement is unnecessary. |
| Only T4 helps, with substantial FP or track-count growth | Admission is a genuine trade-off. Investigate a bounded birth/confirmation policy separately; do not ship a globally lowered threshold on AP alone. |
| T1 helps at identical schedules; T2 adds little | Keep observation anchoring; avoid more quality-gate complexity unless it improves cost or failure prediction. |
| T6 preserves most accuracy while materially reducing tracking work | Simpler short-gap propagation deserves a measured fixed-schedule control; optical flow should justify its cost. |
| Archived schedules fail to beat both count-matched uniform controls consistently | Prefer periodic scheduling as the working baseline. Do not train a scheduler just to force an adaptive winner. |
| Schedule choices show repeatable benefit and cheap signals predict it | Test an explicitly budget-matched prospective closed-loop policy in a separately approved phase. Frozen schedules alone are insufficient. |
| Resolution improves small-object accuracy but removes the compute saving | Report that detector limitation and cost conflict. Do not adopt larger input automatically or change the stock model constraint. |

The five existing sequences cannot provide a new held-out validation claim. Before deployment or claims of generalization, agree on accuracy-loss tolerance and cost requirements and obtain separately approved evaluation data. No additional data is requested or downloaded in this phase.

## Prepared artifacts and approval boundary

The [machine-readable plan](../analysis/argoverse_hd_phase2_plan_v1/plan.json) is marked `AWAITING_USER_APPROVAL_NOT_RUN`. Its directory contains 13 complete config specifications, nine archived/uniform schedule files, a conditional 320-frame manifest and a source hash inventory. These are preparation artifacts, not experiment outputs. Existing reports/results/data and uncommitted algorithm work were preserved.

Initial requested scope: A + four R parity passes + six new B passes + eleven C passes = **21 complete five-sequence replays, 71,736 processed frame steps and zero new YOLO inferences**, plus synthetic harness validation and at most 100 diagnostic cases. Source cache lookups do not count as actual inference. D/E are excluded.

The cached/forced-schedule diagnostic runner does not yet exist. The existing Argoverse runner only supports the recovered five presets and hardcodes historical scoring. After approval, prepare the analysis-only adapter, non-mutating wrappers and arbitrary-arm scorer, run synthetic validation, then execute the approved scope. Production detection/tracking/scheduling files remain unchanged. No fabricated command for an unimplemented driver is presented as runnable. Exact invocation and fresh destinations will be reported before execution.

At the end of the approved experiments, Phase 2 should deliver measured evidence for: detection versus propagation losses; admission versus confirmation versus score effects; geometry benefit per unit tracking work; schedule effects at equal calls; and a conditional Phase 3 recommendation. If evidence is negative or inconclusive, report that rather than enabling additional features. Phase 3 implementation, training, new datasets, commits and pushes remain outside this plan.
