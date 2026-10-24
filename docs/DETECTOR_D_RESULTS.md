# Experiment D: paired stock YOLOv8n detector diagnosis

**On the predetermined320-frame diagnostic subset, increasing input size640->960 raises official COCO AP 16.504911%->21.683948% (+5.179036 percentage points).** The validated640 global-cap-removal shadow changes AP by -0.00001131 points, a negligible negative difference. This supports input scale as a material detector limitation here and does not support the original global300 cap as an explanation for640's low AP on this subset.

The practical reference remains the verified640 anchored Fixed-2 recovery+3x3 configuration with fixed_signal_fastpath=True. It was not modified or replayed. No Experiment E, temporal960 pipeline, Phase4, retraining, pseudo-label dataset generation, commit or push occurred. E requires separate user approval.

## Scope and fidelity

Exactly64 manifest-frozen images from each of five sequences:320 images,5,832 regular annotations and228 crowd annotations. Images are1920x1200. The same stock checkpoint, CPU FP32, batch1, conf0.001, class-aware NMS0.7, native80-class inference and global300 cap are used at both sizes. Existing8-class LUT and original-pixel coordinate semantics are preserved. PyTorch intra-op8/inter-op1 and OpenCV10 threads; OpenCL off; actual input tensors1x3x416x640 and1x3x608x960.

All320 live640 mapped predictions match saved Dense bitwise. Ten canonical unobserved images per size match their observed predictions and native80-class rows exactly. Every observed image reconstructs native top300 output before and after coordinate restoration exactly from native pre-truncation survivor indices. Final audit repeats this reconstruction from all640 saved sidecars. Native NMS backend is torchvision.ops.nms; import/backend selection and original returned tensors/indices remain unchanged. No alternate NMS call or class selection is used for the shadow.

Official pycocotools2.0.11, all8 categories, copied positive annotation IDs, crowd/empty-frame handling and maxDets[1,10,100] remain unchanged. Saved/live640 subset predictions are identical and their official scoring is performed once. AP is pooled rather than averaged across sequence AP. These320-frame scores must not be compared directly with full-dataset Dense15.7327% as though only resolution differed.

## Official scores (%)

| Case | AP50-95 | AP50 | AP75 | AR100 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|---:|
| imgsz640 | 16.504911 | 25.821678 | 16.291748 | 23.503631 | 0.909103 | 16.955672 | 55.172034 |
| imgsz960 | 21.683948 | 32.743186 | 21.819548 | 32.619239 | 4.256600 | 28.121681 | 56.818895 |
| shadow640_no_global300 | 16.504900 | 25.821630 | 16.291745 | 23.504061 | 0.909103 | 16.955666 | 55.171994 |

At exact confidence0.25 and IoU50, micro operating points are:

| Case | TP | FP | Precision % | Recall % | F1 % |
|---|---:|---:|---:|---:|---:|
| imgsz640 | 1602 | 429 | 78.877400 | 27.469136 | 40.747806 |
| imgsz960 | 2188 | 435 | 83.415936 | 37.517147 | 51.756357 |
| shadow640_no_global300 | 1602 | 429 | 78.877400 | 27.469136 | 40.747806 |

Precision, recall and F1 above are micro fixed-threshold/IoU50 values, not COCO AR100 or the historical best-F1 operating point. Full macro/per-class operating counts and thresholds0.001/0.10/0.25 atIoU50/75 are retained in summary.json and official score files.

## Paired gains and losses

| 960 vs640 | Both TP | Gained | Lost | Neither |
|---|---:|---:|---:|---:|
| IoU50_conf0.001 | 2989 | 1007 | 170 | 1666 |
| IoU75_conf0.001 | 1522 | 860 | 151 | 3299 |
| IoU50_conf0.1 | 2040 | 770 | 108 | 2914 |
| IoU75_conf0.1 | 1315 | 653 | 93 | 3771 |
| IoU50_conf0.25 | 1515 | 673 | 87 | 3557 |
| IoU75_conf0.25 | 1168 | 563 | 52 | 4049 |

Gained/lost cells join official GT IDs on identical images. They are not additive AP contributions and do not establish physical identity continuity. Higher resolution has losses as well as gains.

## Sequence and class AP (%)

| Sequence | 640 | 960 | Delta points |
|---|---:|---:|---:|
| sequence_00 | 8.820095 | 13.640069 | +4.819974 |
| sequence_01 | 10.606881 | 14.748091 | +4.141210 |
| sequence_02 | 14.248624 | 19.631494 | +5.382870 |
| sequence_03 | 29.743577 | 41.171602 | +11.428025 |
| sequence_04 | 25.657631 | 27.034755 | +1.377124 |

| Class | 640 | 960 | Delta points |
|---|---:|---:|---:|
| person | 17.225518 | 27.487249 | +10.261731 |
| bicycle | 0.230664 | 0.952223 | +0.721559 |
| car | 27.845828 | 38.394405 | +10.548577 |
| motorcycle | 7.736755 | 7.559349 | -0.177406 |
| bus | 53.834547 | 58.229899 | +4.395352 |
| truck | 10.588105 | 11.258479 | +0.670374 |
| traffic_light | 7.007110 | 11.668307 | +4.661197 |
| stop_sign | 7.570764 | 17.921672 | +10.350908 |

All five sequences improve pooled within-sequence AP, while motorcycle AP falls slightly. That category has only22 regular targets in this subset; bicycle85 and stop_sign41 also have limited support. Sequence class support differs under official COCO rules. No statistical significance or held-out generalization is asserted.

## What native detector evidence reveals

| GT diagnostic category atIoU50/score>0.001 | 640 | 960 |
|---|---:|---:|
| official_TP50 | 3159 | 3996 |
| same_class_native_preNMS_overlap50 | 123 | 151 |
| other_class_native_preNMS_overlap50 | 318 | 332 |
| no_confident_native_preNMS_overlap50 | 2232 | 1353 |

For an officially unmatched GT, the diagnostic checks native best-of80 class assignments and pre-NMS box overlap. At640,2,232 regular targets have no native confident box withIoU>=0.50 even irrespective of predicted class; this falls to1,353 at960. The large resolution effect therefore occurs before the tracker and is consistent with improved detector box/class evidence. It is not explained by fixing LUT, evaluation or global300 truncation.

These are spatial candidate diagnostics, not one-to-one detection recall or proven causal error labels. A nearby other-class box is only a confusion candidate; a correct-class pre-NMS overlap can still be suppressed, conflict in matching or rank below an evaluation cap. Raw GT-class score fields in GT_candidate_diagnostics.jsonl.gz are oracle diagnostics only; no ground-truth class selection is applied to predictions. Disjoint descriptive size bins use area<=1024,1024<area<=9216,area>9216, unlike COCO boundary overlap. Small-object AP remains low even at960, so resolution does not solve all detector limitations.

## Global cap audit

| Input | Images with >300 native survivors | Discarded all80 | Discarded supported8 |
|---|---:|---:|---:|
| 640 | 9 | 236 | 193 |
| 960 | 99 | 5186 | 4353 |

The640 shadow preserves the same native NMS ordering and merely omits global300 truncation before the LUT, then applies official per-category max100. AP 16.504900% versus 16.504911%; exact-threshold TP50/TP75 at0.001,0.10 and0.25 are unchanged. At0.001/IoU50 it adds178 evaluated FPs and zero TPs. Some extra predictions are ignored/capped by official evaluation, so193 discarded supported rows do not imply193 evaluated changes.

The global cap binds more often at960 (99/320 images,4,353 supported rows discarded), but **no960 cap-removal score was approved or run**. Its AP effect remains unquantified. Do not infer that a negligible640 cap effect applies to960 or to untested datasets. The earlier max_nms30000 candidate cap cannot bind for the observed decoded head lengths5,460 and11,970.

## Actual execution costs and instrumentation failure

Successful validated execution:666 application requests =640 primary+20 canonical validation+6 excluded full-frame warmups. The root-model counter directly records333 forwards per size,666 total. Validated inference/capture stage wall 69.772s; process CPU 107.651s. This includes load/decode/hash/validation/raw serialization. Official scoring for three cases totals 14.841s; saved-data diagnostic analysis 5.146s. These stage sums exclude development, preparation/audit, and the failed startup duration.

One preceding640 warmup prediction completed before a counter-instrumentation IndexError. Ultralytics deep-copies its predictor model; a bound callback had copied the external counter object. No NMS observer/counterfactual or validation image had run in that attempt. The failed source, request ledger and traceback summary are preserved. A closure-based callback plus a deepcopy regression test fixed accounting;103 synthetic tests passed. The successful run then received the normal equal three full-frame warmups per size. **Actual completed detector application requests including this failure are667**, one above the planned666. The failed warmup elapsed/root-forward count was not retained by its broken counter; do not represent the666 directly counted forwards as complete measurement of that failed attempt. See counter_instrumentation_amendment.json.

| Size | Canonical validation mean predict ms (10 images) | Observed primary mean predict ms (320 images) |
|---|---:|---:|
| 640 | 19.453458 | 20.724263 |
| 960 | 39.258925 | 40.624023 |

Predict clocks cover model.predict preprocessing/forward/NMS/results, with observation overhead for primary calls; they exclude subsequent NumPy/LUT copying and raw-file serialization. Primary decode sums 3.039s and raw serialization 44.596s are separately recorded. These are one640-then960 diagnostic run with small canonical samples, not counterbalanced E timings, end-to-end throughput, thermal analysis or streaming AP.

## Implications and next approval

Confirmed on this sample: input resolution materially limits detector accuracy; stronger box/class evidence raises AP, recall and high-confidence precision at960. The original640 global cap has negligible scored impact. Existing mapping/evaluation remain consistent. Still unresolved: exact allocation among spatial resolution, scale-dependent model behaviour, domain/blur/occlusion effects; full-dataset960 performance;960 cap loss; temporal960 accuracy and total runtime. No single resolution is declared the deployment winner.

Expected input-pixel arithmetic ratio is2.1923 for960x608 vs640x416. At half as many960 calls, detector arithmetic would be about1.096 times Dense640 before tracking. This is not measured latency/energy and cannot justify claiming both higher AP and lower total cost. The diagnostic timings also show substantial per-call cost growth but do not price a960 adaptive system.

Retain the practical640 fastpath reference. Experiment E remains pending approval and stays CPU-only at640: prioritize new Dense versus practical reference measurements; use the pre-fastpath ablation, high-AP weak-birth and stricter-precision configurations as explanations of the trade-off. Explicitly report FULL versus SKIP latency, decode-inclusive throughput, paired block differences, and performance drift/possible thermal effects. Drift alone cannot establish thermal causation; record run order/power/interruptions and do not silently pool it away. No new960 temporal method, cap change, tracker or scheduler is added to E.

## Reproduction and artifacts

```text
PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -q
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.run --stage prepare
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.run --stage infer
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.run --stage score
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.analyze
PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.report
```

Completed stages refuse overwrites; fresh reproduction needs a separately specified destination. The original failed first invocation and subsequent correction are documented rather than replayed. plan.json, runtime.json, requests.jsonl, canonical_validation*.npz, native80.npz, raw/*.npz, official tensors/matches, summary.csv/summary.json, GT_candidate_diagnostics.jsonl.gz and execution_costs.json retain the full evidence. Final source snapshots and artifact hashes accompany COMPLETE.json. Scoring helper new_YOLO_inferences=0 denotes that scoring stage only; whole-D actual requests are667, not zero.
