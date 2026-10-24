# D/E: detector diagnosis and real-inference performance — proposal

Prepared 20 October 2026. **Awaiting approval. No D/E inference, scoring, replay, tests or algorithm changes have been performed for this proposal.** Read-only inspection verified the completed optimization archive, local checkpoint, hardware, installed sources and original D manifest. Existing code/results/uncommitted work remain intact. Phase 4 is not started.

## Finalized practical reference and unresolved questions

Retain `runs/argoverse_hd_next_research_v1/candidate_config.json` as the practical reference: stock YOLOv8n; CPU FP32; imgsz640/conf0.001/NMS0.7/global max_det300; anchored Fixed-2; high births0.25; existing-track recovery down to0.10; 3x3 LK grid, minimum4 valid points, width640; confidence decay0.93; fixed_signal_fastpath=True. No historical default is changed.

The completed archive's 275 artifact hashes match. Its audit records 1,181 preserved prior-evidence hashes, exact reproduction of three controls, exact fastpath prediction/detail/tracker-state/COCO-tensor parity, 96 synthetic tests, six scored and six timed cache passes, and no inference. The checkpoint remains SHA256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.

Dense remains 15.732713% official AP; practical reference14.692246%; unrestricted weak births15.091065%; stricter matching+grid3 14.682026%. All fixed methods request1,710 vs Dense3,416. The fastpath reduced cached wall step2.759414->1.233757ms and CPU4.546395->2.874584ms; this is not a measured live-inference speedup. Stricter association reduces low-score FP and TP; weak births increase both, with more track/flow work. Neither persistence nor the tested scheduler has established a superior all-metric solution.

Phase1 rules out a wholesale taxonomy/coordinate/matcher error as the main explanation. Dense small-object AP1.008774% vs large53.457692% establishes a size-associated deficit, not its cause. Phase2/3 identify rejected weak evidence and imperfect propagation as additional losses. Latest source-overlap diagnostics identify coarse localisation, duplicate/matching and class-confusion candidates among persistent weak FULL observations. Those descriptive overlaps do not establish physical identities or prove annotation errors. The next question is detector evidence quality and actual total cost, before another tracker/scheduler modification.

## Original D/E and what changes in this proposal

The authoritative narrower D specification is `docs/PHASE2_EXPERIMENT_PLAN.md:144`:64 predetermined frames per sequence,320 total; paired640/960; stock weights/settings; six excluded full-frame warmups; verify640 against saved Dense; inspect all80 classes before mapping. Its manifest is `analysis/argoverse_hd_phase2_plan_v1/conditional_resolution_manifest.json`, SHA256 `619c9a9644d93c7c7e657dcc0db587bbddb138029c56561fc7e3fec2779ea1b2`. The earlier diagnosis's1280 suggestion is not included. The manifest's indices exactly satisfy `(j*(N-1)+31)//63`, j=0..63 for lengths901,901,469,475,670; all320 paths exist. No reselection or favourable-case sampling.

Original E (`docs/PHASE2_EXPERIMENT_PLAN.md:152`) requests repeated real inference, common settings, pipeline-only and decode-inclusive timings, with candidate selection deferred until A–C. Completed later phases now provide the relevant candidates.

This proposal adds: a bounded paired observer check for D, a semantics-preserving pre-cap observation when feasible, an explicitly diagnostic640 cap-removal shadow score, and five fully position-balanced E repetitions. It does not authorize a cap/class-filter/resolution sweep, higher-resolution temporal pipeline, new detector, quantization/export, training, or tracker/scheduler redesign.

## Feasibility and common controls

Read-only hardware inventory: Apple M4 MacBook Air,10 CPU cores (4 performance/6 efficiency),16GB memory; approximately59GiB free storage. CPU is the chosen backend for every method. GPU/MPS is not required or part of the comparison.

Installed runtime: Python3.14.6; PyTorch2.14.0; torchvision0.29.0; Ultralytics8.4.149; NumPy2.5.3; SciPy1.18.1; OpenCV5.0.0.93; pycocotools2.0.11. Record versions, actual tensor device/dtype, NMS implementation, model fusion/backend state, resolved predictor arguments, CPU/OpenCV threads, thread environment, power mode and run order after approval. No dependency upgrade/download. Freeze common threads (PyTorch intra-op8, inter-op1, OpenCV10) before measured work; historical inter-op settings were not recorded, so historical latency is not a paired control. Stop if the640 bridge fails rather than silently changing tolerance/settings.

All real calls use one BGR image per request, batch1, FP32, CPU, same local checkpoint, conf0.001, class-aware NMS0.7, global max_det300, all80 COCO classes in inference, then the existing8-class LUT. Augmentation, CLAHE, ROI/local inference, refinement, model compilation/export and quantization remain off. Explicitly record rect=True and actual preprocessed tensor dimensions; expected stride-padded shapes are640x416 and960x608, subject to verification.

The installed NMS chooses its implementation partly on whether torchvision has already been imported. The D observer must wrap the already-selected branch and preserve import order; it must not import torchvision merely to switch backends. Installed NMS mutates box-coordinate storage in place: any raw tensor observation must clone before mutation and never modify native inputs/results. Retain stable row/index alignment.

Official scoring retains all8 categories, empty frames, crowds, positive copied annotation IDs, original-pixel coordinates and COCO maxDets[1,10,100]. AP is pooled over selected images and categories, not averaged sequence AP. Supplementary exact P/R/F1/TP/FP at0.001,0.10,0.25/IoU50 and75 remain separate estimands. No optimized threshold or changed confidence promotion.

## D — paired detector evidence diagnosis

### D0: validate bridge and observer

Use the first and last selected image of each sequence:10 manifest-fixed frames. At each size, run a canonical unobserved call and an observed call on each image. Observed calls count toward the320 primary predictions per size; there are20 additional unobserved validation calls across both sizes.

First compare canonical640 with corresponding saved Dense predictions; then compare observed vs unobserved outputs, including class/score/order/boxes, all80 post-NMS output and mapping when observable. Aim for bitwise equality. Observer synthetic checks cover empty/crowd-independent detector outputs, class mapping, ties, >300 retained indices, tensor aliasing and coordinate restoration. No actual benchmark method changes. Any discrepancy is recorded and investigated; do not proceed to resolution attribution or E with an unexplained bridge failure.

Three excluded full-frame warmups per size: six explicit warmup calls. Do not use `YOLODetector.warmup`, which additionally performs a batched ROI call. Record any implicit framework/model forward separately from explicit detector requests.

### D1: paired640/960 predictions

Complete one primary prediction for every selected image at each resolution. Retain native pre-NMS decoded head boxes/all80 class scores when accessible, native80-class post-NMS outputs, mapped outputs, confidence ranks, actual shapes/gain/padding and per-stage diagnostic timings. Stream raw artifacts to disk rather than retaining all tensors. Do not select a different class or alter NMS/caps.

Score the saved original640 cache on the same320-image subset once as the bridge reference; this subset score is not the full3,416-frame15.7327% baseline. Score live640 and960 officially, pooled and per sequence/class/COCO size. Report gains/losses atIoU50 and75; original-pixel size and resized min-side bins; confidence distributions; FP/TP at fixed thresholds; duplicate/wrong-class/coarse-overlap candidates; and paired gained/lost GT matches. Raw candidate coverage is diagnostic, not AP or guaranteed recoverable recall. Raw class scores are model outputs, not assumed calibrated probabilities or pre-sigmoid logits.

### D2: native cap audit and bounded shadow

Where the native NMS primitive already returns an index list before `i[:max_det]`, observe that original list without changing it. Reconstruct aligned native survivors and verify its first300 rows reproduce canonical80-class post-NMS output exactly, before/after the same coordinate restoration. Record candidates lost at the original global top300 before dataset LUT, including provided-class survival and unsupported-class consumption. This distinguishes the inference cap from COCO's per-category max100.

If validated, make **one diagnostic640 shadow score** from those same already-computed native NMS survivors with the global300 truncation omitted, then map classes and apply standard COCO max100. No extra YOLO or alternate NMS call, pre-NMS class filtering, best-of8 class selection, new threshold or production setting change. Label it a counterfactual cap audit, not a new deployed benchmark/configuration. Report exactly which targets/FPs differ. It can quantify a cap effect that saturation counts alone cannot.

If non-mutating pre-cap observation/reconstruction cannot be validated, disable that observer, preserve failures, and report the cap question unresolved. A validated canonical640/960 comparison may still proceed; no inferred cap-loss number or shadow score is substituted. Observer failure that affects canonical prediction parity blocks dependent work until resolved. Diagnostic timing with tensor capture is not an uninstrumented deployment latency claim.

### D accounting and interpretation

640 primary calls (320x2 sizes) +20 additional canonical observer-validation calls +6 explicit warmups = **666 planned detector requests**, plus separately counted internal framework forwards if any. The observer validation images are reused, not predicted a third time. No temporal replay on the sparsely sampled320 images, no1280 arm and no full-dataset960 run.

Estimated execution2–5 minutes plus diagnostic scoring/I/O, potentially longer if raw/NMS observation is expensive. Estimates use historical640 detector-call timing around22ms; they are not a fresh performance result. Store roughly1–3GB raw artifacts if decoded heads have expected shapes; reserve up to5GB combined D/E output. Actual shapes/calls/bytes are audited, not assumed.

The expected padded-pixel ratio is `(960*608)/(640*416)=2.1923`. This is an arithmetic proxy, not latency or energy. At half the calls it is approximately1.096 times Dense640 detector arithmetic before tracking. Resolution gain alone therefore does not justify a higher-resolution adaptive pipeline. Interpret its effect jointly with small-object recovery, high-IoU quality, confidence/FP changes and the cap audit. Shared misses remain detector/domain/evidence hypotheses; this experiment will not uniquely separate blur, domain shift and model capacity.

## E — real-inference accuracy and processing cost

Run all3,416 frames, ordered within five original sequences, with state reset at every boundary. No cached detections answer an actual detector request in E. Reuse historical saved predictions/scoring as expected references and post-run validators, not as replacements for timed inference.

| Label | Frozen source | Only permitted config difference | Expected official AP % | Calls/pass |
|---|---|---|---:|---:|
| A Dense640 | original benchmark baseline config | explicit common CPU/thread/inference controls |15.732713|3416|
| B practical reference | latest candidate_config.json | none beyond common controls |14.692246|1710|
| C pre-fastpath ablation | B | fixed_signal_fastpath=False |14.692246|1710|
| D high-AP weak births | Phase3 V1_separate_birth10 config | fixed_signal_fastpath=True and common controls |15.091065|1710|
| E precision-oriented | latest V_strict_grid3 config | fixed_signal_fastpath=True and common controls |14.682026|1710|

All fixed arms receive the same applicable preprocessing optimization except explicit C. D retains its archived4x4 flow and birth/association settings; E retains its stricter low-stage matching and3x3 flow. Do not hybridize them or change thresholds. Fastpath activation for D/E must reproduce their archived predictions/state/official metrics before accepting the corresponding live result. These are reference values to verify, not promised new measurements. Keep original adaptive scheduling results as historical context; no new adaptive method has earned another runtime sweep.

Five blocks use column-permuted Latin orders `ABDCE`, `BCEDA`, `CDAEB`, `DEBAC`, `EACBD`; each method occupies every position once and predecessor diversity improves over simple cyclic ordering. This is not full predecessor-pair balance or a thermal/significance guarantee. Keep workloads/power mode fixed, run sequentially with no competing scoring/tests, use a common30-second pause between blocks, record block trends and any interruptions. Do not select a best repetition. If a block is interrupted, preserve it and report incomplete data; no unapproved extra repeat.

Use a fresh process/model per arm/pass to isolate predictor state and process peak memory. Exclude and separately record model loading and three full-frame warmups per pass (75 total). Warmups use the same first image/shape and do not enter tracker state or official scoring. Underlying framework forward counts must be audited separately.

### Timing scope

In the **same pass**, collect nested external clocks:

1. Decode: local image read+JPEG decode, one `cv2.imread` per processed frame.
2. Pipeline service: external `AdaptivePipeline.step` entry to returned prediction/details, including actual YOLO preprocessing/forward/NMS/mapping, required signals/flow/association/controller and output construction.
3. Decode-inclusive processing: immediately before image read to prediction readiness in memory, including necessary validity checks. Report mean/median/P95/P99, total service time and throughput derived from the same scope.
4. Entire invocation wall time, setup/warmup, validation, serialization and offline scoring separately. Do not relabel sum-of-step FPS as entire-application throughput.

Measure CPU time in matching scopes and process peak RSS. Provide FULL/SKIP/per-sequence distributions, actual detector invocations, detector-call milliseconds, residual non-detector step time, tracking/flow component telemetry, requests saved, and frame-age in indices. No additional instrumented profiling replay is required. Do not subtract nested timings twice or call nominal GFLOPs measured compute/energy.

Avoid heavy hooks/raw tensor capture/GT matching/hashing in timed E steps. Hash input files before/after passes outside timing; compare stored output arrays and schedules after the pass. Score the first complete run of each arm officially; if later outputs are identical, reuse those accuracy scores and report timing replication. If output parity fails, preserve the mismatch and diagnose it rather than pool differing accuracy as a timing repeat. Initial correctness checks are part of the first planned pass, not extra complete cache or real-inference runs.

Images are decoded sequentially; full decoded residency would use approximately22GiB on a16GB machine. Output arrays are modest enough to buffer per pass; write completed arrays/timing logs outside measured prediction service. Report ordinary filesystem-cache conditions, not a forced cold-storage experiment. There is no playback throttling, frame dropping, asynchronous scheduling or future-frame information.

The historical field `fps_end_to_end` is derived from internal `t_total_ms` and excludes decoding; it also omits details constructed after the internal timestamp. Historical CPU process time includes broader loop work. These old figures are context only, not paired evidence for a new total speedup. Actual online capture-to-output latency, deadline misses and streaming AP require a separate timestamped-input/scheduler protocol and are not asserted here.

### E accounting and costs

Five arms x five blocks = **25 full-sequence passes /85,400 processed frames**. Scheduled measured detector requests =5x(3416+4x1710)=**51,280**; explicit excluded warmups75; total **51,355**. Combined D/E planned application-level requests: **52,021**. Internal framework forwards/setup costs are recorded separately and do not disappear from accounting.

Planning estimate: about19 minutes of detector-call time at the old22ms/call rate, plus decode, tracking, process setup,2 minutes block pauses, scoring and audit. Allow **30–60 minutes E machine execution** on this CPU, with a90-minute planning checkpoint if thermal/runtime conditions differ substantially. D and implementation/analysis are additional. These are estimates, not predicted FPS or promised total speedup. Refine the remaining-time estimate after the first block without changing the fixed arm/repeat scope. Reserve up to5GB fresh D/E artifacts; no existing results are deleted.

## Gates and decisions

- **Validity first:** checkpoint/taxonomy/coordinates/actual settings must match; canonical640 bridge and observer parity must be explained; E shared FULL-frame detections and all-frame outputs must reproduce relevant references. Do not silently accept metric changes as numerical noise. Seek bitwise equality; if not attainable, quantify classes, rows, scores, coordinates, threshold/IoU crossings and official-score changes and stop dependent comparisons until scientifically resolved. No arbitrary tolerance is a new approval.
- **D resolution:** quantify paired size/class/sequence gains and FP cost. A larger AP on this sample is detector headroom evidence, not proof of a better efficient temporal pipeline or a generalization claim.
- **D cap:** require validated native pre-truncation reconstruction. A standard-protocol shadow AP change and matched target changes can support or weaken cap loss as an explanation; mere top300 saturation cannot. Compare its magnitude with the current1.0405-point practical-reference gap without treating contributions as additive.
- **E reproduction:** preserve verified B accuracy/precision/call counts; C must have identical outputs. D/E optional common-fastpath activations must preserve their original metrics. Score fidelity takes precedence over a faster result.
- **E efficiency:** evaluate B versus newly measured A using both pipeline-only and decode-inclusive mean/CPU/tail cost; evaluate B versus C for actual fastpath contribution. Report all five paired block changes and uncertainty descriptively. Do not use correlated frames as independent experimental repetitions.
- **E trade-offs:** compare high-AP D and lower-FP E at the same request budget. Use official AP/AP50/AP75/AR100, macro/micro P/R/F1, FP/TP counts and costs together. No unspecified accuracy-loss margin, FP target, energy claim or single weighted score selects a winner. A result that loses accuracy or increases FP is a disclosed trade-off.
- **Stop/limits:** unexplained reproducibility failure, memory pressure or invalid observer pauses dependent work. Do not add1280, cap/class sweeps, higher-resolution temporal methods, repeated failed runs or new tracker/scheduler changes to rescue a result. Preserve failures and request approval for expanded experiments.

## Fresh locations and execution status

Proposed, not created: `runs/argoverse_hd_detector_diagnosis_D_v1/` and `runs/argoverse_hd_real_runtime_E_v1/`. Original Phase2 proposed resolution directory is unused and will not be overwritten. Freeze complete configs, invocation commands, source snapshots, manifests and actual runtime before execution after approval. New analysis-only adapters/validation/timing harness are to be implemented after approval; no nonexistent runner command is represented as already usable.

Existing COCO scoring and manifest helpers should be reused. Do not launch the old monolithic benchmark runner unchanged: it uses historical custom evaluation, incomplete external timing and extra ROI warmup. No rerun of completed A/R/B/C or cached optimization passes is necessary. Review/feasibility read-only checks and new plan metadata do not constitute D/E execution.

Sources: original D/E definitions in `docs/PHASE2_EXPERIMENT_PLAN.md:144`; final reference/results in `docs/NEXT_RESEARCH_RESULTS.md`; detector mapping/warmup in `extracted/adaptive_pipeline/detection/detector.py:36`; timing scopes in `extracted/adaptive_pipeline/run.py:49`, `telemetry/metrics.py:70` and `pipeline.py:216`; installed Ultralytics `utils/nms.py:125`/`:163`, `engine/predictor.py:349` and `utils/torch_utils.py:350`. Official [Ultralytics predict arguments](https://docs.ultralytics.com/modes/predict/) support explicit shape/device/inference controls; [PyTorch benchmarking guidance](https://docs.pytorch.org/docs/2.14/benchmark_utils.html) motivates warmup/thread control and replicated timing; [Towards Streaming Perception](https://arxiv.org/abs/2005.10420) distinguishes offline processing/AP from streaming output accuracy.
