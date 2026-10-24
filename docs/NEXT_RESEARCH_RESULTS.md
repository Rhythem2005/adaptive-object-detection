# Evidence-driven optimization: findings and recommendation

**Recommended candidate: anchored Fixed-2, low-score continuation, 3x3 flow and the guarded fixed-signal fastpath.** It preserves the prior 14.692246% COCO AP exactly while reducing measured cache-step overhead by 55.29% and CPU time by 36.77%. Detector requests remain 1,710 (49.94% fewer than Dense). This stage establishes a processing-efficiency improvement, not a new accuracy gain or an end-to-end acceleration.

The detector remains stock YOLOv8n. The only production changes are an opt-in fixed_signal_fastpath flag, validation guards and omission of unused thumbnail/luminance/illumination updates for supported fixed configurations. The flag defaults off. Tracking, flow, detector settings and evaluator semantics are unchanged. Tentative-birth and output-filter prototypes stay isolated in analysis and were not promoted or run on the dataset after their diagnostic gates failed.

The controlled timing comparison is against the identical recovery+3x3 configuration. The unused 160-pixel INTER_AREA thumbnail resize costs 1.392818 ms/frame in this measurement, compared with 0.056593 ms for thumbnail luminance. Required flow preparation remains enabled. The main removed cost is resizing for unused signals, not a replacement flow algorithm or GPU transfer. Timer wrappers are common to both arms and included in the measured scope.

## What improved, and what did not

- Fastpath: every output, prediction detail, tracker-state hash and official COCO tensor matches its control; there is no additional frame-buffering delay. Mean cached step 2.759414 -> 1.233757 ms; P95 3.514942 -> 1.970289 ms; CPU 4.546395 -> 2.874584 ms. All three paired wall-time changes are between -55.43% and -55.11%. These repeated measurements remain specific to this hardware, OpenCV build and cache-processing scope.
- Stricter low-score association plus 3x3: AP 14.682026%, a loss of 0.010221 percentage points versus recovery+3x3. At confidence0.10/IoU50, FP falls 15,239 -> 14,755 (-3.18%) while TP falls 19,561 -> 19,420 (-141). Precision rises 56.2098% -> 56.8252%; recall falls 35.3010% -> 35.0466%. Counts at confidence0.25 are identical. Mean tracks fall 8.85246 -> 8.44965 and flow points 135,855 -> 129,663. This is a precision/recall/work-proxy trade-off; its latency was not separately priced.
- The alternate Fixed-2 phase scores 14.616755% AP versus 14.692246%, with identical request counts. It increases TP25 but also FP25. This does not establish a new online scheduler; its budget uses known sequence lengths.
- Simple two-observation confirmation fails the predeclared precision-gain screen. Birth-source precision improves only 3.30 percentage points while rejecting 41.52% of source true detections; FULL-output precision improves 2.29 points while rejecting 23.46%. Consequently five conditional arms, including Dense postprocessing, are skipped. No AP outcome is claimed for these unrun variants, and the screen does not rule out all temporal confirmation approaches.

## Why persistence was insufficient

Of 2,954 retained weak birth-source FP50 events, descriptive GT-overlap checks identify 1,500 with maximum regular-GT IoU in [0.10,0.50), 655 with same-class overlap>=0.50 (duplicate/matching-conflict candidates), 212 with only other-class overlap>=0.50, and 587 with lower overlap. Cars account for 2,246 retained FP events. These are errors in the FULL detector source observations, before propagation. Persistence cannot by itself establish correct class, uniqueness or box extent.

These categories are overlap diagnostics, not verified physical root-cause labels: coarse overlap does not prove a particular object identity, and low overlap does not prove background or annotation error. Counts are observation events, not unique objects. They nevertheless challenge an explanation centred only on unstable background lights or optical-flow drift. Any later candidate-quality work should explicitly distinguish localisation, duplication and classification from object absence.

## Historical official controls

| Previously scored method | AP % | Requests | P25 micro % | R25 micro % | FP25 |
|---|---:|---:|---:|---:|---:|
| baseline | 15.732713 | 3416 | 76.7966 | 28.3116 | 4740 |
| fixed2_093 | 14.020016 | 1710 | 77.8020 | 27.6474 | 4371 |
| C_fullfallback | 13.981381 | 1733 | 77.9876 | 27.7232 | 4336 |
| adaptive_observed | 14.164180 | 1729 | 77.7980 | 27.7359 | 4386 |
| adaptive_quality | 14.212644 | 1789 | 77.8511 | 27.6944 | 4366 |

Phase2/3 anchoring, recovery, unrestricted weak births, stricter association, Fixed-2 phase and Fixed-3 references are additionally retained in summary.json/summary.csv. The high-AP weak-birth control reproduces 15.091065% AP, but has 20,680 all-frame FP10 versus 15,239 for recovery+3x3 and 428,144 versus 135,855 flow points. Historical flow-point differences are work proxies; no unmeasured latency is assigned to that arm.

## Interpretation and next decision

Use the exported candidate_config.json for the strongest measured practical efficiency configuration. If fewer low-score false positives are worth the explicit recall/AP loss, retain V_strict_grid3 as a separate precision-oriented option. No deployment miss/false-alarm cost or accepted noninferiority margin was supplied, so this report does not silently select that trade-off.

The selected candidate remains 1.040467 AP percentage points below Dense. This stage supplies no new evidence that it can surpass Dense at half the requests. The most defensible next research question is reliable box/candidate quality, with explicit duplicate/localisation/classification diagnostics and cost limits, rather than stronger persistence thresholds or a more elaborate scheduler. A harder NMS threshold, appearance model, confidence promotion or new propagation algorithm is not justified merely by these overlaps and has not been implemented.

[ByteTrack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) supports lower-score continuation conditioned on established tracks; its tracking results do not validate weak births here. [OC-SORT](https://arxiv.org/abs/2203.14360) motivates preserving observations but does not establish a need for more tracker complexity. [Towards Streaming Perception](https://arxiv.org/abs/2005.10420) explains why cached processing cost and offline COCO AP cannot establish deployed streaming accuracy.

All five sequences are development evidence already reviewed in earlier phases. Results are descriptive; independent generalization, real-inference throughput, energy, and any resolution/raw-detector hypothesis remain unresolved. No new inference or D/E work was performed. Phase4 is not begun. Further algorithm experiments require separate approval.

## Detailed measurements and reproducibility

# Evidence-driven optimization: measured results

This stage uses saved stock YOLOv8n detections only. All five sequences / 3,416 frames / 55,412 regular targets and 2,299 crowd annotations are retained. Official COCO evaluation is unchanged. No inference, training, D/E, Phase 4, commit or push occurred.

## Controls and diagnostic gates

All three controls reproduce predictions, details, tracker-state hashes, official precision/recall/score tensors and aggregate/class/sequence metrics exactly.

| Gate | Original TP/FP | Retained TP/FP | Original precision % | Retained precision % | TP retention % | Passed |
|---|---:|---:|---:|---:|---:|---|
| birth | 1462/6172 | 855/2954 | 19.151 | 22.447 | 58.482 | False |
| output | 2698/8444 | 2065/5726 | 24.215 | 26.505 | 76.538 | False |

The gate requires >=50% TP retention, >=5 percentage-point precision gain and >=50 source TP events. These predeclared engineering thresholds are not significance tests. Diagnostics use recovery+grid3 unmatched weak observations, with official Dense source labels; intervention changes future candidate membership. protocol_amendment.json explains why immediate-birth events would have introduced selection bias. No candidate replay preceded this correction. Stable background can persist and real objects can fail unwarped IoU confirmation. Ground-truth labels are never consumed by a runtime candidate.

Skipped arms: {"V_birth_grid3": "Failed pre-specified saved-evidence birth gate", "V_confirmed_grid3": "Failed pre-specified saved-evidence birth gate", "V_output_grid3": "Failed pre-specified saved-evidence output gate", "V_confirmed_output_grid3": "Failed pre-specified saved-evidence both gates", "V_dense_output": "Failed pre-specified saved-evidence output gate"}

## Official metrics and operating points

AP/AR and precision cells below are percentages. P25 is micro precision at exact confidence 0.25 and IoU50. FP10 is all-frame FP at exact confidence 0.10. These are different estimands from COCO AP.

| Method | AP | AP50 | AP75 | AR100 | TP25 | FP25 | P25 | TP10 | FP10 | Requests | Mean tracks | Flow points |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense existing | 15.732713 | 23.865010 | 15.654390 | 23.191434 | 15688 | 4740 | 76.7966 | 21002 | 21535 | 3416 | — | — |
| F_fastpath | 14.692246 | 21.897441 | 14.787367 | 19.012228 | 15335 | 4373 | 77.8110 | 19561 | 15239 | 1710 | 8.85246 | 135855 |
| F_phase1 | 14.616755 | 21.820146 | 14.731607 | 19.176962 | 15437 | 4417 | 77.7526 | 19580 | 15189 | 1710 | 8.84778 | 135900 |
| R_birth10 | 15.091065 | 22.715169 | 15.117125 | 20.010943 | 15325 | 4383 | 77.7603 | 20790 | 20680 | 1710 | 15.68940 | 428144 |
| R_grid3 | 14.692246 | 21.897441 | 14.787367 | 19.012228 | 15335 | 4373 | 77.8110 | 19561 | 15239 | 1710 | 8.85246 | 135855 |
| R_recovery | 14.671818 | 21.874537 | 14.763550 | 19.001518 | 15326 | 4382 | 77.7654 | 19549 | 15236 | 1710 | 8.84485 | 241312 |
| V_strict_grid3 | 14.682026 | 21.867362 | 14.785965 | 18.984982 | 15335 | 4373 | 77.8110 | 19420 | 14755 | 1710 | 8.44965 | 129663 |

Full macro/micro P/R/F1 at exact thresholds, size/class AP, all prior Phase 3 controls, and source-event/paired-match diagnostics are in summary.json and summary.csv. AP is pooled, not an average of sequence scores.

## Per-sequence COCO AP (%)

| Method | sequence_00 | sequence_01 | sequence_02 | sequence_03 | sequence_04 |
|---|---:|---:|---:|---:|---:|
| Dense | 7.39145 | 9.73167 | 14.02454 | 29.16092 | 24.66150 |
| F_fastpath | 6.79392 | 8.85826 | 13.01882 | 26.17249 | 23.99620 |
| F_phase1 | 6.79014 | 8.92762 | 13.02335 | 26.33039 | 23.98280 |
| R_birth10 | 6.89420 | 9.09753 | 13.32109 | 27.24276 | 24.30951 |
| R_grid3 | 6.79392 | 8.85826 | 13.01882 | 26.17249 | 23.99620 |
| R_recovery | 6.75326 | 8.82011 | 12.99925 | 26.13030 | 24.00088 |
| V_strict_grid3 | 6.77794 | 8.84048 | 12.98256 | 26.18794 | 24.01418 |

Sequence class support differs; sequence AP must not be averaged to recover pooled AP. The five sequences have already influenced development. No untouched test set or generalization claim is implied.

## Cached processing cost and parity

Selected timed reference: R_grid3. Fastpath changes only unused thumbnail/LAB/illumination preparation; required 640-wide flow preparation still executes on FULL and SKIP. The raw model input, detector schedule, output arrays, details, tracker arrays and official tensors are identical. Inactive luminance/controller-thumbnail/illumination telemetry is intentionally absent or no longer updated; no enabled decision or enhancement consumes it.

| Arm | Mean wall ms | Median | P95 | P99 | Mean CPU ms | Mean signal ms | Required flow preparation ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| R_grid3 | 2.759414 | 2.830563 | 3.514942 | 3.733891 | 4.546395 | 1.997199 | 0.520975 |
| F_fastpath | 1.233757 | 1.154688 | 1.970289 | 2.125408 | 2.874584 | 0.482467 | 0.480707 |

Wall change: -55.289%; CPU change: -36.772%. Paired round wall changes: [-55.42950257276774, -55.32589057000262, -55.11093633714316]. Three AB/BA/AB repetitions include identical lightweight component timer wrappers; their overhead is included. No heavy event observers execute. Decode/hash/output validation/serialization and model execution are outside the timed step. CPU time includes worker threads. These are not end-to-end FPS, energy measurements or streaming AP.

## Protocol correction and development failures

Before the first gate or new variant, the candidate cohort was corrected to avoid conditioning on immediate weak births. Only the unused gate function in the driver changed; controls and all algorithm source bytes remained identical, and original source/manifest versions are retained. Two diagnostic helper-import errors were corrected before a gate result was written; they caused no pipeline replay or scoring. The initial pytest command was unavailable; the existing unittest runner was used without installing dependencies. See protocol_amendment.json, execution_source_revision.json and development_failures.json.

## Reproduction and boundaries

Commands (repository root; every completed stage refuses overwrite):

```text
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.run --stage prepare
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.run --stage controls
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.run --stage gate
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.run --stage variants
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.run --stage finalists
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.run --stage timing
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.summarize
```

Exact configs, conditional gates, forced schedules, source snapshots/hashes and pre-existing dirty files are archived here. plan.json is the predeclared scope; finalist_specs.json contains the selected full configurations. Historical results and dataset files were not overwritten. The fastpath defaults off; causal birth/output adapters remain isolated in analysis, rather than silently replacing production algorithms.

This report is descriptive. Summary JSON retains paired gained/lost official GT matches at multiple confidence/IoU thresholds. Those are not additive AP contributions. Flow-point counts are work proxies, not latency measurements. Candidate selection uses the declared TP/FP gate followed by AP and point work; the measured frontier has no repeated CPU price for every arm. D/E remain deferred.

Final reporting/audit commands:

```text
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.complete_report
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.verify
```

The first successful corrected gate used the diagnostic module directly, as recorded in protocol_amendment.json; the canonical `--stage gate` command now delegates to the same corrected implementation. Original manifests/source and failed development sources are retained. Final verification and execution_costs.json record synthetic checks, immutable prior hashes, completed pass counts and stage costs.
