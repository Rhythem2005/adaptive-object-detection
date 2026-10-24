# Branch B: latest-frame processing — bounded validation results

**Result: the latest-frame mechanism works in the approved timestamp-paced replay. All predeclared criteria passed in both repetitions.** This validates the original freshness objective at the application-buffer level. It does not establish real-camera sensor-to-result latency or tracking accuracy across dropped frames. The frozen offline Fixed-2 results remain separate and unchanged.

## Scope and controls

- One experiment, eight fresh worker processes: sequence_00 source frames 0–299, 300 images per pass; 15 and 90 FPS time-scaled replay, two repetitions per policy/rate. This is an early single-sequence diagnostic, not held-out validation or a five-sequence benchmark.
- Ground truth: 300 images, 1,512 regular boxes and 30 crowds. Regular targets in this prefix are car (1,310), truck (188) and bus (14). Its AP is not comparable to the full five-sequence 15.732713% Dense640 AP.
- Independent producer decodes each JPEG at its mapped source-timestamp deadline. Original integer nanosecond timestamps and source frame IDs are retained. No preload, artificial inference delay, hidden producer drop, or source backpressure is used. Publication jitter is logged; arrival-timing acceptance checks passed in all eight workers.
- Unchanged stock YOLOv8n, CPU FP32, batch 1, imgsz 640 (observed tensor 1×3×416×640), confidence 0.001, NMS IoU 0.7, global max_det 300; unchanged COCO-to-Argoverse mapping. Torch intra/inter-op 8/1; OpenCV 10 threads, OpenCL off. The same Dense policy="always" pipeline branch runs in both arms. Optical flow, tracking, and temporal detector scheduling are bypassed.
- Checkpoint SHA256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- Balanced order: R1 15 FIFO→latest, then 90 latest→FIFO; R2 90 FIFO→latest, then 15 latest→FIFO. Three full-frame detector warmups per worker precede the replay epoch. No retry, tuning, extra configuration or 960-resolution experiment occurred.
- FIFO holds at most the finite 300-frame source population; reaching capacity raises rather than throttling the producer. Latest holds one pending frame. Both drain pending work at EOF. No presentation/UI is timed; result-ready means the unchanged detection pipeline has returned an output to the headless consumer.

## Capture-to-result latency and frame accounting

Latency starts at the mapped source-arrival deadline **before JPEG decode**, and ends when the detection output becomes available. It includes decode, queue wait and processing; it excludes startup, warmup, post-run scoring/serialization and physical display latency.

| Source FPS | Rep | Policy | Captured | Processed | Overwritten | Peak pending | P50 ms | P95 ms | Max ms |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | 1 | fifo | 300 | 300 | 0 | 1 | 31.90 | 32.88 | 34.82 |
| 15 | 1 | latest | 300 | 300 | 0 | 1 | 31.54 | 32.63 | 34.01 |
| 90 | 1 | fifo | 300 | 300 | 0 | 138 | 1455.23 | 2428.87 | 2551.73 |
| 90 | 1 | latest | 300 | 171 | 129 | 1 | 31.29 | 35.70 | 84.33 |
| 15 | 2 | fifo | 300 | 300 | 0 | 1 | 31.80 | 33.34 | 82.27 |
| 15 | 2 | latest | 300 | 300 | 0 | 1 | 31.46 | 32.82 | 85.35 |
| 90 | 2 | fifo | 300 | 300 | 0 | 131 | 1317.42 | 2282.55 | 2392.85 |
| 90 | 2 | latest | 300 | 175 | 125 | 1 | 30.45 | 35.32 | 41.12 |

Overload P95 reductions were **98.5302%** and **98.4528%**, exceeding the predeclared 50% threshold. Latest discarded **129/300 (43.0%)** and **125/300 (41.67%)** pending frames. It selected the newest published source frame at every atomic dequeue, including frame 299 after EOF; it never cancels an in-flight inference.

At 15 FPS neither policy dropped a frame. Latest P95 was 0.25 and 0.52 ms lower than FIFO, within the allowed FIFO+5 ms limit. Occasional maxima of 82–85 ms remain: no zero-latency or hard real-time deadline guarantee is claimed. In overload R1, a 76.24 ms service-time spike produced the 84.33 ms latest-frame maximum despite its short queue.

Every worker ended with `captured = completed + overwritten = 300`, `consumed = completed`, pending=0, pending bytes=0, and a joined capture thread. There were no source/decode failures or hidden source drops. Source IDs strictly increased; IDs were never renumbered after dropping. For all four overloaded passes, **299 of the 300 publications occurred during active inference intervals**, directly confirming that capture continued while YOLO was busy.

## Frame age, throughput, CPU and memory

Active output FPS uses the common first-to-last source-arrival window; drain-inclusive output FPS additionally includes post-EOF processing. The source window is about 19.93 s at 15 FPS and 3.32 s at 90 FPS. These are actual replay output rates, not offline processing-capacity estimates.

| FPS / rep / policy | Frame age at inference P95 ms | Active output FPS | Output FPS through drain | Post-EOF drain ms | Replay-loop CPU s | CPU %* | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 / 1 / fifo | 11.49 | 15.00 | 15.03 | 20.12 | 19.083 | 95.5 | 495.08 |
| 15 / 1 / latest | 11.48 | 15.00 | 15.03 | 19.42 | 18.841 | 94.3 | 495.38 |
| 90 / 1 / fifo | 2409.82 | 48.46 | 51.07 | 2545.66 | 16.358 | 277.5 | 1866.97 |
| 90 / 1 / latest | 16.65 | 50.87 | 51.02 | 23.32 | 9.700 | 287.7 | 533.50 |
| 15 / 2 / fifo | 11.83 | 15.00 | 15.03 | 19.60 | 19.090 | 95.5 | 526.34 |
| 15 / 2 / latest | 11.43 | 15.00 | 15.03 | 19.46 | 18.835 | 94.2 | 503.38 |
| 90 / 2 / fifo | 2264.18 | 50.57 | 52.49 | 2386.69 | 16.033 | 279.6 | 1784.11 |
| 90 / 2 / latest | 16.42 | 51.77 | 52.03 | 34.89 | 9.782 | 289.1 | 536.25 |

*CPU 100% denotes one fully occupied core. Process CPU time includes producer decode, the consumer and backend threads across the replay/drain window; it excludes model startup/warmups and scoring. Per-service process CPU fields also include concurrent producer work and must not be described as detector-only CPU time. RSS is process-lifetime peak, including startup, not an incremental image-buffer size.

Under overload latest used **40.70% and 38.99% less total replay-loop CPU** for the same 300 source arrivals, because it ran fewer detections. It did not make each YOLO invocation intrinsically faster: mean service times were FIFO/latest 19.53/19.53 ms (R1) and 19.00/19.15 ms (R2). Its advantage is freshness and avoided obsolete work, not a new detector or higher intrinsic detector capacity.

FIFO peak occupancy was 138 and 131 decoded images; latest remained one. Peak RSS was approximately 1,867/1,784 MiB for FIFO and 534/536 MiB for latest during overload. Both stayed below the 4,096 MiB guard. This short finite-source test does not establish long-duration RSS stability.

## Accuracy and fidelity

All **2,146 processed prediction arrays and their prediction-detail arrays** matched archived Dense640 outputs byte for byte (dtype, shape and content); 24 additional forwards were warmups. Actual forward hooks verified CPU FP32 and identical input tensor shapes. Every processed step was FULL. No cached detection substituted for inference.

For detector-quality comparisons, only the intersection of processed source IDs is used. Once both emitted arrays were proven identical to the archived array, the official pycocotools result was computed once for that common population and reused; this is exact identity, not a statistical equivalence claim.

| Compared population | Common frames | FIFO, latest and archived Dense COCO AP % |
|---|---:|---:|
| 15 FPS, R1 | 300 | 23.600778 |
| 90 FPS, R1 | 171 | 22.232948 |
| 15 FPS, R2 | 300 | 23.600778 |
| 90 FPS, R2 | 175 | 23.522809 |

At 15 FPS the shared 300-frame evaluation has TP=888, FP=387, precision=69.6471% and recall=58.7302% at confidence 0.25 / IoU 0.50, identical for both policies. The 90 FPS common subsets differ between repetitions; differences in their AP must not be interpreted as detector improvement. AP on each policy’s unmatched frame population is not used to compare detector quality.

A separate **replay-time-aligned held-output diagnostic** evaluates the last completed boxes, unchanged, at the same 300 source-arrival query times. Before the first result the output is empty. It uses the official COCO bbox scorer but is **not official streaming AP**, conventional per-frame detector AP, or a tracking metric. It measures consequences of stale/absent delivered outputs on this sample.

| Condition | FIFO aligned AP % | Latest aligned AP % | FIFO visible-frame age P95 ms | Latest visible-frame age P95 ms |
|---|---:|---:|---:|---:|
| 15 FPS, R1 | 18.3806 | 18.3806 | 66.67 | 66.67 |
| 90 FPS, R1 | 0.7750 | 5.5339 | 1477.78 | 55.56 |
| 15 FPS, R2 | 18.3598 | 18.3794 | 66.67 | 66.67 |
| 90 FPS, R2 | 0.7477 | 6.1763 | 1400.01 | 44.44 |

The aligned diagnostic still has low absolute AP during overload: even the latest completed output trails the current scene. Dropping obsolete work removes backlog; it cannot remove inference duration or recover objects visible only on discarded frames. Normal-condition aligned AP is lower than ordinary AP because queries occur at the next source arrival, before its result is available. No-result queries (1 at 15 FPS, 3–4 at 90 FPS) are retained, not excluded from AP; visible-age summaries exclude these undefined ages and report their counts separately.

## Implementation and validation evidence

- Extended [LatestFrameBuffer](../extracted/adaptive_pipeline/adaptive_framing/frame_manager.py:37), preserving its legacy tuple API and one-pending-frame default. Added immutable packet metadata, atomic selection evidence, closure/error signaling, accounting and a bounded FIFO control. Frame ownership transfers to the buffer; producers must not mutate a published image.
- Extended [VideoCaptureThread](../extracted/adaptive_pipeline/adaptive_framing/capture.py:16) with a stored thread, cancellation, join, release attempts, buffer closure and surfaced errors. [TimestampReplayCapture](../extracted/adaptive_pipeline/adaptive_framing/capture.py:113) reuses its lifecycle and scales verified source timestamp offsets; late releases are logged rather than retimestamped or hidden.
- Added the isolated [stream consumer and entry point](../extracted/adaptive_pipeline/run_stream.py:46). It preserves source IDs/timestamps through output readiness and drains EOF atomically. It restricts this validation to frozen CPU Dense640. The existing offline `run.py`, pipeline, configuration defaults, detection, optical flow and tracking algorithms were not edited.
- Added [one-shot validation harness](../analysis/latest_frame_validation.py:50): immutable scope/order/criteria, hard limits, source hashes, official scoring, no retries and separate timing scopes.
- Added [synthetic tests](../tests/test_streaming.py:1): producer progress while inference is blocked, replacement vs FIFO order, last-frame EOF delivery, failure draining, cancellation, release errors, monotonic metadata, configuration and memory guards. **130 tests passed** in the full test discovery run (18 new streaming tests).

Commands (already executed; outputs are immutable and commands intentionally refuse reuse of the same output location):

```sh
PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=extracted:. .venv/bin/python -m analysis.latest_frame_validation prepare
PYTHONPATH=extracted:. .venv/bin/python -m analysis.latest_frame_validation run
```

All eight exact subprocess commands are retained in `../runs/branch_b_latest_frame_v1/passes/*/SUBPROCESS.json`. Config, source hashes, input hashes, original timestamps, counters, per-frame clocks, raw predictions, official tensors and query-to-result assignments are retained under this root.

## Budget, preservation and limitations

- Actual execution plus scoring/preservation gates: **120.26 wall seconds**, below 600 s. Requests **2,170 = 2,146 measured + 24 warmup**, below 2,424. Scoring took 8.17 wall s / 8.16 process CPU s, outside all timed replay scopes. Replay-loop process CPU summed across the eight passes was 127.723 s. Startup/warmup CPU was not separately instrumented; this sum is not total experiment CPU.
- Outputs were approximately 9.8 MB before this report and final manifests, below the 250 MB limit. No decoded images or videos were copied into the output directory.
- Independent post-run preservation audit checked **2,578 prior archived files with zero mismatches**, plus 106 protected preexisting source/report/config/checkpoint/index files. All execution-source hashes still match. Existing uncommitted changes were retained. No commit/push, training, scheduler optimization or paused 960 + Fixed-3 execution occurred.
- The Mac remained on AC power (battery charging 26% before / 27% after). Thermal/power-level telemetry was unavailable from pmset; temperatures and throttling were not measured. Two balanced repetitions are sufficient for this bounded demonstration, not a general hardware performance characterization.
- Images were hashed before execution, warming the filesystem cache. The measured decode stage is JPEG decoding with file reads, not camera/video-codec/network acquisition or cold-disk throughput.
- Real camera acquisition was not run. Generic camera timestamps are host receive times after decode, not sensor exposure timestamps. Upstream camera/driver/decoder backlog is backend-dependent and is not eliminated merely by a one-slot application buffer. Continuous camera reads avoid added artificial pacing, but camera buffering, timestamp calibration and sensor-to-display latency remain unverified.
- Generic VideoCapture file pacing uses declared file FPS and is marked as such; it is not validated variable-frame-rate playback. This experiment instead uses original verified dataset timestamps. Camera/device reads that cannot be interrupted can still delay shutdown; join surfaces a timeout rather than claiming guaranteed hardware cancellation.
- The old `run.py --realtime` demo is unchanged and still discards capture timestamps in its caller. The validated timestamp/EOF consumer is the separate `run_stream.py` path. Do not attribute these new measurements to the legacy demo or to frozen offline Fixed-2.
- Variable frame-gap optical flow/tracker behavior, physical presentation latency, other scenes, native-camera source FPS and long-running live deployments were not validated. Do not attach the frozen Fixed-2 AP to this dropped-frame stream.

## Finalization recommendation

**Freeze this implementation and its evidence as the controlled demonstration of the original latest-frame adaptive-framing objective.** Incoming frames continue while YOLO is busy, only the newest pending frame survives, and the consumer processes that newest available frame when free. It materially reduces backlog age without changing detector outputs. This is sufficient to finalize the application-level research objective with explicit replay/camera limitations; it is not certification of physical camera deployment or arbitrary-gap tracking. Keep the separate verified Fixed-2 configurations as the established offline accuracy–computation results. No further research phase is begun.
