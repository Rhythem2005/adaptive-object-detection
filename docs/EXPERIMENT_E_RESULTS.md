# Experiment E — live CPU accuracy and processing cost

**Retain B as the practical reference.** Its decode-inclusive mean falls from24.0495 to16.1027ms/frame (33.0435% less wall time;41.581→62.101FPS), with36.5029% less process CPU and49.9415% fewer detector requests. AP remains14.692246% versus Dense15.732713%; the1.040467-point gap is unresolved. FULL latency is higher than Dense; inexpensive SKIP frames provide the aggregate saving.

Completed under the approved 640-only scope. Stock YOLOv8n, CPU FP32, unchanged detector/tracker/scheduler. All 25 passes reproduce their archived prediction arrays and details byte for byte, schedules exactly, and temporal sequence-end tracker hashes exactly. The five first passes reproduce official COCO metrics, per-sequence metrics, operating points and precision/recall/score tensors exactly. Later passes reuse these scores only after exact prediction fidelity.

Each pass processes all 3,416 original 1920×1200 images in five ordered sequences (901,901,469,475,670 frames), with tracking state reset at every boundary. The shared provided-annotation evaluation contains 55,412 regular objects and 2,299 crowds across eight categories; no pseudo-labels, empty-frame removal or altered taxonomy.

## Accuracy, precision and request budget

AP is official COCO maxDets [1,10,100], all eight mapped categories, original-pixel coordinates, crowds/empty frames retained. P/R/F1 below are micro at confidence≥0.25 and IoU50; FP0.10 uses the same official matching at confidence≥0.10. These fixed operating points are separate from AP and COCO AR100.

|Arm|AP%|AP50%|AP75%|AR100%|P25%|R25%|F1%|TP25|FP25|TP10|FP10|Calls/pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|A Dense640|15.732713|23.865010|15.654390|23.191434|76.7966|28.3116|41.3713|15688|4740|21002|21535|3416|
|B Practical fastpath|14.692246|21.897441|14.787367|19.012228|77.8110|27.6745|40.8280|15335|4373|19561|15239|1710|
|C Pre-fastpath|14.692246|21.897441|14.787367|19.012228|77.8110|27.6745|40.8280|15335|4373|19561|15239|1710|
|D Weak births (high AP)|15.091065|22.715169|15.117125|20.010943|77.7603|27.6565|40.8014|15325|4383|20790|20680|1710|
|E Strict association|14.682026|21.867362|14.785965|18.984982|77.8110|27.6745|40.8280|15335|4373|19420|14755|1710|

Category-macro operating points at confidence0.25/IoU50 (not COCO AP):

|Arm|Macro P%|Macro R%|Macro F1%|Micro P10%|Micro R10%|
|---|---|---|---|---|---|
|A|62.4565|18.0835|25.0018|49.3735|37.9015|
|B|66.2819|17.5831|24.4270|56.2098|35.3010|
|C|66.2819|17.5831|24.4270|56.2098|35.3010|
|D|66.2396|17.5702|24.4073|50.1326|37.5189|
|E|66.2819|17.5831|24.4270|56.8252|35.0466|

All fixed arms save 49.941452% of detector requests. B/C output fidelity makes their comparison an accuracy-preserving preprocessing ablation. D retains the archived weak-birth/4×4 flow settings; E retains stricter low-stage association/3×3 flow. D/E use the same verified fixed-signal fastpath as B. No confidence, cap, resolution or class-filter change was introduced.

## Timing and CPU cost

Numbers pool equal-size passes over five blocks. FPS uses 3,416 divided by the summed relevant frame scope per pass; it is sequential file processing, without playback pacing. Pipeline service externally includes step return and diagnostic detail construction. Decode-inclusive adds local JPEG read/decode and shape/type validation. CPU time is process time across threads, so it can exceed elapsed wall time. Residual non-detector service includes signals, flow/Kalman propagation, association/output construction, controller and Python overhead; it is not pure tracker time.

|Arm|Decode mean ms|Step mean ms|Decode-inclusive mean ms|Processing FPS|Step CPU ms/frame|Inclusive CPU ms/frame|Non-detector wall / CPU ms/frame|Loop FPS|
|---|---|---|---|---|---|---|---|---|
|A|5.1947|18.8548|24.0495|41.581|47.9335|53.0524|0.0307 / 0.0278|41.571|
|B|5.1425|10.9602|16.1027|62.101|28.6097|33.6867|1.3657 / 3.0692|62.079|
|C|5.0729|12.4476|17.5205|57.076|29.1066|34.1776|2.8737 / 4.6142|57.057|
|D|5.0691|11.3294|16.3985|60.981|29.2745|34.3416|1.6517 / 3.5862|60.960|
|E|5.0646|11.0128|16.0773|62.199|28.6969|33.7601|1.3539 / 3.0549|62.178|

Pooled frame distributions (ms; correlated frames are not independent experimental repetitions):

|Arm|Decision (frames over5passes)|Step mean / median / P95 / P99|Inclusive mean / median / P95 / P99|Step CPU mean|
|---|---|---|---|---|
|A|FULL (17080)|18.8548 / 18.7883 / 19.7903 / 21.3065|24.0495 / 24.0019 / 25.3617 / 26.6865|47.9335|
|B|FULL (8550)|20.0231 / 19.9454 / 21.1933 / 22.0548|25.1433 / 25.1268 / 26.4245 / 27.3118|53.2138|
|B|SKIP (8530)|1.8761 / 1.8577 / 2.1869 / 2.3359|7.0410 / 7.1499 / 7.7714 / 8.0110|3.9479|
|C|FULL (8550)|21.5092 / 21.2959 / 22.5568 / 24.1539|26.5675 / 26.4804 / 27.7993 / 29.5251|52.7401|
|C|SKIP (8530)|3.3647 / 3.3375 / 3.7101 / 3.9569|8.4522 / 8.5721 / 9.1194 / 9.4232|5.4176|
|D|FULL (8550)|20.2416 / 20.1120 / 21.3312 / 22.7200|25.3028 / 25.2278 / 26.5828 / 27.9895|53.5557|
|D|SKIP (8530)|2.3963 / 2.2464 / 3.2317 / 3.6436|7.4733 / 7.4643 / 8.5310 / 8.9643|4.9364|
|E|FULL (8550)|20.1401 / 20.0382 / 21.1296 / 22.2709|25.1952 / 25.1520 / 26.4102 / 27.4664|53.3936|
|E|SKIP (8530)|1.8641 / 1.8442 / 2.1601 / 2.3105|6.9381 / 7.0866 / 7.5321 / 7.7117|3.9422|

Measured internal telemetry, averaged across all frames; nested scopes are not added twice:

|Arm|Detector FULL mean ms|Signal mean ms/frame|Propagation mean ms/frame|Decision mean ms/frame|Association/output mean ms/frame|
|---|---|---|---|---|---|
|A|18.8241|0.0000|0.0000|0.0000|0.0000|
|B|19.1666|0.5097|0.6251|0.0038|0.1918|
|C|19.1254|2.0221|0.6190|0.0040|0.1924|
|D|19.3328|0.5105|0.8585|0.0039|0.2435|
|E|19.2951|0.5087|0.6203|0.0039|0.1860|

Block-paired reductions; positive values mean less cost. Each arm occupied each order position exactly once; predecessor pairs are not fully balanced. the five blocks are repetitions, not five independent datasets. No significance or thermal-causality claim is made.

|Comparison|Step reduction% (five blocks)|Inclusive reduction% (five blocks)|CPU inclusive reduction% (five blocks)|
|---|---|---|---|
|B_vs_A|39.78, 43.75, 41.76, 42.55, 41.46|32.47, 33.19, 32.93, 33.90, 32.71|35.54, 37.13, 36.45, 37.32, 36.06|
|C_vs_A|32.02, 34.64, 34.66, 34.42, 34.14|26.45, 27.49, 27.54, 27.30, 26.96|34.71, 35.98, 36.11, 35.61, 35.47|
|D_vs_A|38.01, 40.08, 40.41, 40.74, 40.28|31.05, 31.65, 31.99, 32.56, 31.81|34.58, 35.12, 35.57, 35.58, 35.48|
|E_vs_A|39.93, 41.82, 41.51, 42.43, 42.24|32.52, 33.12, 32.73, 33.88, 33.48|35.54, 36.39, 35.70, 37.24, 36.94|
|B_vs_C|11.42, 13.95, 10.88, 12.40, 11.11|8.19, 7.86, 7.44, 9.08, 7.87|1.28, 1.80, 0.53, 2.65, 0.91|
|D_vs_B|-2.95, -6.53, -2.33, -3.14, -2.01|-2.11, -2.31, -1.39, -2.04, -1.34|-1.49, -3.20, -1.37, -2.78, -0.90|
|E_vs_B|0.25, -3.44, -0.44, -0.21, 1.32|0.07, -0.11, -0.30, -0.03, 1.15|-0.01, -1.18, -1.17, -0.13, 1.38|

## Performance drift, memory and execution conditions

Block orders: `ABDCE`, `BCEDA`, `CDAEB`, `DEBAC`, `EACBD`; 30-second pauses separate blocks. Fresh interpreter/model per pass; three excluded full-frame warmups, no ROI warmup. No concurrent scoring/tests/heavy diagnostics during timed work; first-block official scoring occurs between blocks. Read/hash validation is outside timing and can warm filesystem caches. Storage is not forced cold. An archive-schema guard stopped after first-block Dense scoring: Phase1 stores four extra axis-metadata arrays. The continuation validated all those arrays and the three result tensors exactly, completed the remaining first-block scores, then resumed blocks2–5 with the frozen worker. No timing pass was repeated. This investigation lengthened the first inter-block gap; all raw timestamps and the interruption traceback are retained.

|Arm|Inclusive mean ms by block1→5|Last/first change%|Step mean ms by block1→5|Loop peak RSS MiB range|Whole worker peak RSS MiB range|
|---|---|---|---|---|---|
|A|24.0383, 23.9761, 23.9840, 24.2721, 23.9770|-0.26|18.5635, 18.8795, 18.8776, 19.0885, 18.8651|542.8–561.2|575.8–594.2|
|B|16.2322, 16.0178, 16.0861, 16.0433, 16.1343|-0.60|11.1781, 10.6191, 10.9936, 10.9671, 11.0431|525.8–536.1|552.7–562.9|
|C|17.6810, 17.3843, 17.3799, 17.6446, 17.5127|-0.95|12.6194, 12.3404, 12.3353, 12.5192, 12.4238|536.8–540.6|563.8–567.5|
|D|16.5744, 16.3880, 16.3103, 16.3699, 16.3498|-1.36|11.5078, 11.3120, 11.2501, 11.3118, 11.2654|532.6–537.2|560.0–564.3|
|E|16.2208, 16.0352, 16.1337, 16.0477, 15.9494|-1.67|11.1506, 10.9842, 11.0423, 10.9896, 10.8972|532.3–536.7|559.0–563.6|

Peak RSS is the macOS process high-water mark (bytes converted to MiB), includes imports/model and buffered predictions; the whole-worker peak additionally includes archive validation and serialization. Loop peak was read before those postprocessing steps. It is not incremental tracker allocation or total system memory. Startup/postwarmup values and per-pass costs are retained in `COMPLETE.json`.

`pmset` power and warning outputs are retained at every pass boundary. They consistently report AC Power, while the battery percentage declined from55% at the first accepted pass to36% at completion and was reported as discharging. Captured AC settings show lowpowermode0. The physical charging condition was not independently measured or changed. This Mac did not expose usable thermal warning/CPU power status through `pmset -g therm`; no chip temperature, frequency, joules or throttling event was measured. Drift can reflect heating, OS contention, order and workload; it cannot uniquely identify thermal effects.

## Per-sequence results

Per-sequence AP is evaluated separately with official category support rules; aggregate AP is pooled COCO AP, not its average. Latency below pools five repeated passes for that sequence.

|Sequence (frames)|Arm|AP%|AP50%|P25%|R25%|F1%|TP25 / FP25|Inclusive mean / P95 ms|Step mean ms|CPU inclusive ms/frame|
|---|---|---|---|---|---|---|---|---|---|---|
|sequence_00 (901)|A|7.391448|11.861156|67.2253|31.9540|43.3178|2888 / 1408|23.5491 / 24.5497|18.5357|52.2327|
|sequence_00 (901)|B|6.793922|10.945498|68.1348|31.3233|42.9167|2831 / 1324|15.7972 / 25.6467|10.8082|33.1356|
|sequence_00 (901)|C|6.793922|10.945498|68.1348|31.3233|42.9167|2831 / 1324|17.2176 / 26.9329|12.3041|33.6712|
|sequence_00 (901)|D|6.894196|11.256355|68.0385|31.2790|42.8561|2827 / 1328|15.9692 / 25.5812|11.0617|33.5502|
|sequence_00 (901)|E|6.777941|10.900935|68.1348|31.3233|42.9167|2831 / 1324|15.7249 / 25.5213|10.8200|33.1081|
|sequence_01 (901)|A|9.731668|18.203215|75.3931|30.2587|43.1852|2877 / 939|23.6101 / 24.5131|18.8471|52.5388|
|sequence_01 (901)|B|8.858258|16.472037|77.0828|29.2911|42.4510|2785 / 828|15.5896 / 25.3857|10.8497|33.1140|
|sequence_01 (901)|C|8.858258|16.472037|77.0828|29.2911|42.4510|2785 / 828|17.1123 / 26.9339|12.4393|33.7533|
|sequence_01 (901)|D|9.097534|17.134161|76.9720|29.2491|42.3901|2781 / 832|15.9679 / 25.6244|11.2907|33.7918|
|sequence_01 (901)|E|8.840478|16.408063|77.0828|29.2911|42.4510|2785 / 828|15.6204 / 25.4531|10.9499|33.2081|
|sequence_02 (469)|A|14.024540|21.520144|86.4463|19.6604|32.0351|3138 / 492|24.5274 / 25.4558|18.9508|53.7361|
|sequence_02 (469)|B|13.018822|19.327304|87.2727|19.2469|31.5384|3072 / 448|16.6517 / 26.3994|11.1007|34.4238|
|sequence_02 (469)|C|13.018822|19.327304|87.2727|19.2469|31.5384|3072 / 448|18.0504 / 27.9698|12.6014|34.8816|
|sequence_02 (469)|D|13.321093|20.157571|87.2159|19.2344|31.5179|3070 / 450|16.9898 / 26.4614|11.5483|35.2086|
|sequence_02 (469)|E|12.982561|19.217627|87.2727|19.2469|31.5384|3072 / 448|16.6104 / 26.4150|11.1694|34.5336|
|sequence_03 (475)|A|29.160920|52.033453|79.0198|35.6981|49.1791|3160 / 839|24.6613 / 26.3335|19.2043|53.9288|
|sequence_03 (475)|B|26.172489|46.952660|79.7419|34.9074|48.5582|3090 / 785|16.6021 / 26.4535|11.2032|34.4103|
|sequence_03 (475)|C|26.172489|46.952660|79.7419|34.9074|48.5582|3090 / 785|17.8832 / 27.5358|12.5813|34.7019|
|sequence_03 (475)|D|27.242759|49.337590|79.7419|34.9074|48.5582|3090 / 785|17.0482 / 26.5303|11.7416|35.4738|
|sequence_03 (475)|E|26.187937|46.949090|79.7419|34.9074|48.5582|3090 / 785|16.4809 / 26.2174|11.1832|34.4386|
|sequence_04 (670)|A|24.661504|34.876819|77.3416|30.0755|43.3094|3625 / 1062|24.5451 / 25.6147|18.9795|53.7454|
|sequence_04 (670)|B|23.996202|33.339033|78.2618|29.5113|42.8606|3557 / 988|16.4654 / 26.3116|11.0425|34.1691|
|sequence_04 (670)|C|23.996202|33.339033|78.2618|29.5113|42.8606|3557 / 988|17.8487 / 27.6208|12.4493|34.5649|
|sequence_04 (670)|D|24.309510|33.846657|78.2618|29.5113|42.8606|3557 / 988|16.6802 / 26.5202|11.2962|34.7356|
|sequence_04 (670)|E|24.014180|33.365425|78.2618|29.5113|42.8606|3557 / 988|16.5065 / 26.4044|11.1261|34.3570|

## Reproducibility and actual execution cost

Accepted run: 25 passes /85,400 frames,51,280 measured detector requests +75 excluded warmups =51,355 requests and root forwards. Three preserved pre-sequence startup failures used nine additional warmups, so **51,364 actual E requests/forwards including failed startup validation**. No extra full passes, 960 inference, temporal replay from caches, cap changes, training or Phase4 work.

Accepted-run elapsed wall span: 2007.676s from the first execution marker to completion, including workers, first-block scoring, the evaluation-schema investigation interruption, hash/output validation, serialization and four pauses. Stopped startup orchestrator wall: 10.870s. Implementation, synthetic tests, preparation hashing and final report/audit are separate development costs.

Initial startup guards exposed: Ultralytics lazy imports reset OpenCV threads; installed predictor removed `half` in favour of `quantize`; AutoBackend renamed the older `pt` flag to `format`. These were harness compatibility errors, resolved without changing algorithms/inference settings. All stopped attempts remain in E_v1/E_v2/E_v3 and are hashed by `startup_corrections.json`. The corrected 110-test synthetic suite passed before accepted execution;112 tests passed after adding the axis-metadata fidelity checks, before the continuation. One attempted test-module command failed to import the non-package tests path; the discovery command corrected it. Draft report finalization also hit an exclusive-create guard; the preserved analysis JSON is reused only after exact equality verification, with no rescoring or inference. Development logs and the initial report draft are retained.

Mean per-pass wall scopes (seconds); subprocess cost includes interpreter/import startup, unlike the worker invocation clock. Validation/serialization/scoring are excluded from per-frame service throughput. These are nested or disjoint scopes as named, not quantities to sum indiscriminately.

|Arm|Setup|Warmups|Processing loop|Post-loop validation|Serialization|Worker invocation|Subprocess including imports|Loop CPU seconds|
|---|---|---|---|---|---|---|---|---|
|A|0.014|0.539|82.173|0.971|0.455|85.038|86.197|181.249|
|B|0.014|0.557|55.026|0.947|0.350|57.839|59.029|115.099|
|C|0.014|0.544|59.870|0.870|0.349|62.364|63.565|116.775|
|D|0.014|0.517|56.037|0.868|0.362|58.516|59.708|117.336|
|E|0.014|0.547|54.939|0.865|0.347|57.430|58.636|115.349|

Five first-pass official scoring calls took 150.154s in scorer wall clocks; later identical passes were not rescored.

Commands (from repository root):

```sh
PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_runtime_E.run prepare
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_runtime_E.run run
# First-block schema guard stopped scoring, not timing; validate metadata and continue remaining passes:
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_runtime_E.resume
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_runtime_E.report
```

Historical E_v1/v2/v3 preparation commands used earlier harness snapshots and are preserved; exclusive-create guards prevent overwriting/repeating completed execution. Each subprocess invocation is recorded in `passes/*/SUBPROCESS.json`. The E_v4 execution plan, frozen configs, source snapshot, source/prior hashes, input/GT manifest, raw per-frame CSVs, saved predictions/details, official scoring, power logs and summaries provide the reproducibility record.

Actual common runtime: macOS-26.1-arm64-arm-64bit-Mach-O; Python3.14.6, PyTorch2.14.0, Ultralytics8.4.149, OpenCV5.0.0.93, pycocotools2.0.11. Fused native PyTorch CPU backend, FP32, input1×3×416×640, class-aware torchvision NMS, all80 detector classes before the existing8-class map. PyTorch intra8/inter-op1, OpenCV10/OpenCLoff; all recorded runtime fields match across25 processes. Decode, flow, tracker, scheduler and offline COCO scoring run on CPU.

New files are analysis-only E runner/continuation/report/audit and synthetic tests, fresh E run directories and this report. No production algorithm/config file or prior result was edited; prior protected hashes are checked before/after execution and in final audit. No commit/push.

## Interpretation, limits and next direction

**Confirmed efficiency benefit, with an accuracy trade-off.** B reduces pipeline wall time41.8707% and inclusive wall33.0435%; five paired inclusive reductions span32.47–33.90%, and CPU-inclusive reductions35.54–37.32%. At confidence0.25, precision rises1.0145 points versus Dense, with367 fewer FPs and353 fewer TPs; recall falls0.6370 points. Fewer FPs accompany lower recall; E did not optimize accuracy.

**FULL and SKIP differ.** B FULL service averages20.0231ms versus Dense18.8548ms; inclusive FULL averages25.1433ms versus24.0495ms. FULL-inclusive P95 is26.4245ms versus Dense25.3617ms. SKIP service averages1.8761ms (inclusive7.0410ms), generating the throughput gain. A lower aggregate mean does not establish lower latency on every frame or streaming deadline compliance.

**Tracking is a bounded, visible overhead.** B spends1.3657ms/frame on all non-detector service, including0.5097ms signal/flow preparation, 0.6251ms propagation/Kalman and0.1918ms update/output telemetry. The residual is12.46% of service and8.48% of inclusive processing; its CPU residual is3.0692ms/frame. This includes orchestration/output work, not pure LK or association. Eliminating that entire residual could remove only8.48% of current inclusive wall—an accounting upper bound, not an achievable optimization forecast. Detector and decode account for about59.58% and31.94% of inclusive time.

**The fastpath survives real inference.** B/C predictions/details and accuracy/schedules are identical; their only frozen config difference is fixed_signal_fastpath. Signal cost falls2.0221→0.5097ms/frame. Live step wall falls11.9493% and inclusive wall8.0920%, with all five paired inclusive improvements positive. Inclusive CPU falls only1.4364% (paired0.53–2.65%); detector-phase CPU charging also differs between live arms. These multithreaded process clocks do not identify instruction/energy savings or explain CPU redistribution between phases. The old cache-only55% wall reduction is not a live-pipeline speedup.

**Weak births buy AP at a precision/work cost.** D adds0.398819 AP points over B, remaining0.641648 below Dense. At0.10 it adds1,229TPs and5,441FPs (+35.70% FP), reducing precision56.2098→50.1326%. At0.25 it loses10TPs and adds10FPs. It takes1.8367% more inclusive wall and1.9440% more CPU than B; SKIP service is27.73% higher. Propagation increases0.6251→0.8585ms/frame, consistent with the previously archived larger track/flow workload. Retain it as the accuracy-oriented comparison, not the practical default.

**Strict association offers a lower-score precision trade-off, without a demonstrated runtime win.** E removes484FPs and141TPs at0.10 versus B; AP falls0.010221 points. Its0.25 TP/FP counts are identical. Inclusive wall differs only0.1577% in E’s favour, paired signs vary, and CPU is0.2180% higher. This is not a reliable efficiency advantage.

**Drift is small and not monotonic.** First→last inclusive means change−0.26%,−0.60%,−0.95%,−1.36%,−1.67% forA–E. No method shows progressive mean slowdown. This does not prove absence of heating/throttling: temperature was unavailable, the first scoring/validation gap was longer, and OS/power effects were not isolated.

**Next direction: detector evidence quality and scale under a measured compute budget, with held-out validation.** The completed [D diagnosis](DETECTOR_D_RESULTS.md) found paired320-frame640→960 AP16.504911→21.683948%, stronger small/medium evidence and a negligible640 cap-removal effect. These remain diagnostic sample results, not full-dataset960 or temporal results. Together with E, this supports studying where better detector/localisation evidence is worth its real cost before another tracker/scheduler redesign. Do not infer that half-rate960 is cheaper: D’s padded-pixel arithmetic exceeds Dense640 at half the calls before tracking, and its diagnostic timing does not price a temporal pipeline. Keep stock weights, B as the practical control, D as the high-AP control, official AP plus fixed-threshold FP/TP and actual inclusive wall/CPU as co-primary outcomes. Require previously unused sequences before claiming generalization. Any scale-allocation/960 temporal experiment or algorithm change needs separate approval; none was implemented. Phase4 remains unstarted.

Implementation references: [E timing worker](../analysis/argoverse_hd_runtime_E/run.py:198), [schema-aware continuation](../analysis/argoverse_hd_runtime_E/resume.py:21), [official scorer](../analysis/argoverse_hd_phase2/scoring.py:40), [pipeline timing/output scopes](../extracted/adaptive_pipeline/pipeline.py:216), [detector wrapper](../extracted/adaptive_pipeline/detection/detector.py:36). Exact arm settings are in E_v4/configs/A.json throughE.json; absolute archived source paths are in execution_plan.json.

Five previously studied sequences are an in-sample benchmark. Repetition tests runtime stability, not new-scene generalization. CPU process time is a computational-cost proxy, not energy or FLOP measurement. Timed steps include small common passive counter/CPU-timer overhead; no heavy tensor capture, hashing, serialization or GT scoring occurs inside their scope. Sequential file throughput is not capture-to-output latency, streaming AP or deadline compliance; synchronous processing may outrun or lag nominal video arrival. CPU-only conclusions do not establish MPS/GPU/export performance. No confidence tuning or candidate selection on new holdout data occurred.
