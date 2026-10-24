"""Finalize D only, audit original evidence, and stop before Experiment E."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import time

import numpy as np
from analysis.argoverse_hd_phase2.common import archive, dump, load, sha
from .run import BASE, ROOT, LATEST, BENCH, verify


def main():
    verify();assert (ROOT/'ANALYSIS_COMPLETE.json').exists()
    if (ROOT/'COMPLETE.json').exists():raise FileExistsError('D already finalized')
    doc=BASE/'docs/DETECTOR_D_RESULTS.md'
    if doc.exists():raise FileExistsError(doc)
    started=time.perf_counter();x=load(ROOT/'summary.json');frames=load(ROOT/'frames.json')
    proposal=load(BASE/'analysis/argoverse_hd_DE_plan_v1/plan.json')
    assert sha(BASE/'yolov8n.pt')==proposal['weights_sha256']
    reference=archive(BENCH/'baseline/preds.npz');live=archive(ROOT/'imgsz640/preds.npz')
    for f in frames:
        assert sha(f['image'])==f['image_sha256']
        np.testing.assert_array_equal(live[f['key']],reference[f['key']])
    # Independently reconstruct every native/capped/mapped array from saved sidecars.
    from .observer import map_rows
    from ultralytics.utils import ops
    import torch
    for size in [640,960]:
        config=load(ROOT/'plan.json')['configs'][str(size)]
        mapping={int(k):v for k,v in config['class_map'].items()}
        pred=archive(ROOT/f'imgsz{size}/preds.npz');native=archive(ROOT/f'imgsz{size}/native80.npz')
        shadow=archive(ROOT/f'imgsz{size}/precap_mapped.npz')
        canonical=archive(ROOT/f'imgsz{size}/canonical_validation.npz')
        canonical80=archive(ROOT/f'imgsz{size}/canonical_validation80.npz')
        assert pred.keys()==native.keys()==shadow.keys()==live.keys()
        for f in frames:
            key=f['key']
            with np.load(ROOT/f'imgsz{size}/raw'/f"{f['image_id']:06d}.npz") as a:
                original=torch.from_numpy(a['pre_nms_rows'][a['native_survivor_indices']].copy())
                np.testing.assert_array_equal(original[:300].numpy(),a['capped_letterbox'])
                original[:,:4]=ops.scale_boxes(a['input_tensor_shape'].tolist(),original[:,:4],(1200,1920,3))
                np.testing.assert_array_equal(original.numpy(),a['native_survivors_original'])
                np.testing.assert_array_equal(original[:300].numpy(),native[key])
                np.testing.assert_array_equal(map_rows(native[key],mapping),pred[key])
                np.testing.assert_array_equal(map_rows(original.numpy(),mapping),shadow[key])
                expected=[416,640] if size==640 else [608,960]
                assert a['input_tensor_shape'].tolist()==expected
                assert a['raw_head'].dtype==np.float32 and a['raw_head'].shape[1]==84
                assert np.isfinite(a['raw_head']).all()
            if key in canonical:
                np.testing.assert_array_equal(pred[key],canonical[key])
                np.testing.assert_array_equal(native[key],canonical80[key])
        args=load(ROOT/f'imgsz{size}/resolved_predictor_args.json')
        for field,value in {'device':'cpu','imgsz':size,'conf':.001,'iou':.7,'max_det':300,'rect':True,
            'augment':False,'agnostic_nms':False,'classes':None,'batch':1}.items():assert args[field]==value,(field,args.get(field))
    events=[json.loads(line) for line in (ROOT/'requests.jsonl').read_text().splitlines()]
    complete=[r for r in events if r['status']=='complete'];begun=[r for r in events if r['status']=='started']
    assert len(complete)==len(begun)==666
    assert [r['request'] for r in complete]==list(range(2,668))
    assert sum(r['observed_root_forwards'] for r in complete)==666
    kinds={k:sum(r['kind']==k for r in complete) for k in ['observed_primary','canonical_validation','excluded_full_frame_warmup']}
    assert kinds=={'observed_primary':640,'canonical_validation':20,'excluded_full_frame_warmup':6}
    for r in complete:
        assert r['last_forward']['device']=='cpu' and r['last_forward']['dtype']=='torch.float32'
        assert r['last_forward']['shape']==([1,3,416,640] if r['size']==640 else [1,3,608,960])
    failure=load(ROOT/'counter_instrumentation_amendment.json')
    assert failure['failed_attempt_completed_application_requests']==1
    assert len((ROOT/'failures/attempt1_counter_deepcopy/requests.jsonl').read_text().splitlines())==1
    assert (ROOT/'failures/attempt1_counter_deepcopy/INFERENCE_FAILURE.json').exists()
    before=ROOT/'pre_execution/extracted/adaptive_pipeline'
    for p in before.rglob('*.py'):
        assert sha(p)==sha(BASE/'extracted/adaptive_pipeline'/p.relative_to(before))
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE)==(ROOT/'pre_execution/git_head.txt').read_bytes()
    prior_counts={}
    for name in ['argoverse_hd_coco_phase1_v1','argoverse_hd_phase2_diagnostics_v1','argoverse_hd_phase3_v1','argoverse_hd_next_research_v1']:
        root=BASE/'runs'/name;manifest=load(root/'artifact_sha256.json')
        for rel,digest in manifest.items():assert sha(root/rel)==digest,(name,rel)
        prior_counts[name]=len(manifest)
    assert not (BASE/'runs/argoverse_hd_real_runtime_E_v1').exists()
    costs=dict(x['execution_costs'])
    costs['analysis_wall_seconds']=load(ROOT/'ANALYSIS_COMPLETE.json')['analysis_wall_seconds']
    costs['failed_startup_warmup_wall_time']='not recorded; first prediction completed before external-counter failure'
    costs['failed_startup_root_forward_count']='unavailable from failed bound-method counter; counted separately as one completed detector request'
    costs['primary_decode_seconds']=0.;costs['primary_raw_serialization_seconds']=0.
    for size in [640,960]:
        with (ROOT/f'imgsz{size}/frames.csv').open() as stream:rows=list(csv.DictReader(stream))
        costs['primary_decode_seconds']+=sum(float(r['decode_seconds']) for r in rows)
        costs['primary_raw_serialization_seconds']+=sum(float(r['serialization_seconds']) for r in rows)
    costs['artifact_bytes_before_final_metadata']=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    dump(ROOT/'execution_costs.json',costs)
    metrics=x['metrics'];a=metrics['imgsz640']['COCO'];b=metrics['imgsz960']['COCO'];c=metrics['shadow640_no_global300']['COCO']
    delta=(b['AP']-a['AP'])*100;shadow_delta=(c['AP']-a['AP'])*100
    report=['# Experiment D: paired stock YOLOv8n detector diagnosis','',
        f"**On the predetermined320-frame diagnostic subset, increasing input size640->960 raises official COCO AP {a['AP']*100:.6f}%->{b['AP']*100:.6f}% (+{delta:.6f} percentage points).** The validated640 global-cap-removal shadow changes AP by {shadow_delta:.8f} points, a negligible negative difference. This supports input scale as a material detector limitation here and does not support the original global300 cap as an explanation for640's low AP on this subset.",'',
        'The practical reference remains the verified640 anchored Fixed-2 recovery+3x3 configuration with fixed_signal_fastpath=True. It was not modified or replayed. No Experiment E, temporal960 pipeline, Phase4, retraining, pseudo-label dataset generation, commit or push occurred. E requires separate user approval.','',
        '## Scope and fidelity','',
        'Exactly64 manifest-frozen images from each of five sequences:320 images,5,832 regular annotations and228 crowd annotations. Images are1920x1200. The same stock checkpoint, CPU FP32, batch1, conf0.001, class-aware NMS0.7, native80-class inference and global300 cap are used at both sizes. Existing8-class LUT and original-pixel coordinate semantics are preserved. PyTorch intra-op8/inter-op1 and OpenCV10 threads; OpenCL off; actual input tensors1x3x416x640 and1x3x608x960.','',
        'All320 live640 mapped predictions match saved Dense bitwise. Ten canonical unobserved images per size match their observed predictions and native80-class rows exactly. Every observed image reconstructs native top300 output before and after coordinate restoration exactly from native pre-truncation survivor indices. Final audit repeats this reconstruction from all640 saved sidecars. Native NMS backend is torchvision.ops.nms; import/backend selection and original returned tensors/indices remain unchanged. No alternate NMS call or class selection is used for the shadow.','',
        'Official pycocotools2.0.11, all8 categories, copied positive annotation IDs, crowd/empty-frame handling and maxDets[1,10,100] remain unchanged. Saved/live640 subset predictions are identical and their official scoring is performed once. AP is pooled rather than averaged across sequence AP. These320-frame scores must not be compared directly with full-dataset Dense15.7327% as though only resolution differed.','',
        '## Official scores (%)','',
        '| Case | AP50-95 | AP50 | AP75 | AR100 | AP small | AP medium | AP large |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in metrics:
        m=metrics[name]['COCO']
        report.append('| '+name+' | '+' | '.join(f'{100*m[k]:.6f}' for k in ['AP','AP50','AP75','AR100','AP_small','AP_medium','AP_large'])+' |')
    report+=['','At exact confidence0.25 and IoU50, micro operating points are:','',
        '| Case | TP | FP | Precision % | Recall % | F1 % |','|---|---:|---:|---:|---:|---:|']
    for name in metrics:
        p=x['operating_points'][name]['0.25']['50']
        report.append(f"| {name} | {p['TP']} | {p['FP']} | {100*p['precision']:.6f} | {100*p['recall']:.6f} | {100*p['F1']:.6f} |")
    report+=['','Precision, recall and F1 above are micro fixed-threshold/IoU50 values, not COCO AR100 or the historical best-F1 operating point. Full macro/per-class operating counts and thresholds0.001/0.10/0.25 atIoU50/75 are retained in summary.json and official score files.','',
        '## Paired gains and losses','',
        '| 960 vs640 | Both TP | Gained | Lost | Neither |','|---|---:|---:|---:|---:|']
    for key,values in x['paired_official_matches']['960_vs640'].items():
        q=values['all'];report.append(f"| {key} | {q.get('both',0)} | {q.get('gained',0)} | {q.get('lost',0)} | {q.get('neither',0)} |")
    report+=['','Gained/lost cells join official GT IDs on identical images. They are not additive AP contributions and do not establish physical identity continuity. Higher resolution has losses as well as gains.','',
        '## Sequence and class AP (%)','',
        '| Sequence | 640 | 960 | Delta points |','|---|---:|---:|---:|']
    for seq,m in metrics['imgsz640']['per_sequence'].items():
        old=m['AP']*100;new=metrics['imgsz960']['per_sequence'][seq]['AP']*100
        report.append(f'| {seq} | {old:.6f} | {new:.6f} | {new-old:+.6f} |')
    report+=['','| Class | 640 | 960 | Delta points |','|---|---:|---:|---:|']
    for name,m in a['per_class'].items():
        old=m['AP']*100;new=b['per_class'][name]['AP']*100
        report.append(f'| {name} | {old:.6f} | {new:.6f} | {new-old:+.6f} |')
    report+=['','All five sequences improve pooled within-sequence AP, while motorcycle AP falls slightly. That category has only22 regular targets in this subset; bicycle85 and stop_sign41 also have limited support. Sequence class support differs under official COCO rules. No statistical significance or held-out generalization is asserted.','',
        '## What native detector evidence reveals','',
        '| GT diagnostic category atIoU50/score>0.001 | 640 | 960 |','|---|---:|---:|']
    labels=['official_TP50','same_class_native_preNMS_overlap50','other_class_native_preNMS_overlap50','no_confident_native_preNMS_overlap50']
    for label in labels:report.append(f"| {label} | {x['raw_evidence']['640']['counts']['all'].get(label,0)} | {x['raw_evidence']['960']['counts']['all'].get(label,0)} |")
    report+=['','For an officially unmatched GT, the diagnostic checks native best-of80 class assignments and pre-NMS box overlap. At640,2,232 regular targets have no native confident box withIoU>=0.50 even irrespective of predicted class; this falls to1,353 at960. The large resolution effect therefore occurs before the tracker and is consistent with improved detector box/class evidence. It is not explained by fixing LUT, evaluation or global300 truncation.','',
        'These are spatial candidate diagnostics, not one-to-one detection recall or proven causal error labels. A nearby other-class box is only a confusion candidate; a correct-class pre-NMS overlap can still be suppressed, conflict in matching or rank below an evaluation cap. Raw GT-class score fields in GT_candidate_diagnostics.jsonl.gz are oracle diagnostics only; no ground-truth class selection is applied to predictions. Disjoint descriptive size bins use area<=1024,1024<area<=9216,area>9216, unlike COCO boundary overlap. Small-object AP remains low even at960, so resolution does not solve all detector limitations.','',
        '## Global cap audit','',
        '| Input | Images with >300 native survivors | Discarded all80 | Discarded supported8 |','|---|---:|---:|---:|']
    for size in ['640','960']:
        v=x['cap_audit'][size];report.append(f"| {size} | {v['frames_with_native_survivors_gt300']} | {v['discarded_total80']} | {v['discarded_supported8']} |")
    report+=['',
        f"The640 shadow preserves the same native NMS ordering and merely omits global300 truncation before the LUT, then applies official per-category max100. AP {100*c['AP']:.6f}% versus {100*a['AP']:.6f}%; exact-threshold TP50/TP75 at0.001,0.10 and0.25 are unchanged. At0.001/IoU50 it adds178 evaluated FPs and zero TPs. Some extra predictions are ignored/capped by official evaluation, so193 discarded supported rows do not imply193 evaluated changes.",'',
        'The global cap binds more often at960 (99/320 images,4,353 supported rows discarded), but **no960 cap-removal score was approved or run**. Its AP effect remains unquantified. Do not infer that a negligible640 cap effect applies to960 or to untested datasets. The earlier max_nms30000 candidate cap cannot bind for the observed decoded head lengths5,460 and11,970.','',
        '## Actual execution costs and instrumentation failure','',
        f"Successful validated execution:666 application requests =640 primary+20 canonical validation+6 excluded full-frame warmups. The root-model counter directly records333 forwards per size,666 total. Validated inference/capture stage wall {costs['stage_wall_seconds_with_load_decode_hash_validation_capture_serialization']:.3f}s; process CPU {costs['stage_process_seconds_with_load_decode_hash_validation_capture_serialization']:.3f}s. This includes load/decode/hash/validation/raw serialization. Official scoring for three cases totals {costs['scoring_wall_seconds']:.3f}s; saved-data diagnostic analysis {costs['analysis_wall_seconds']:.3f}s. These stage sums exclude development, preparation/audit, and the failed startup duration.",'',
        'One preceding640 warmup prediction completed before a counter-instrumentation IndexError. Ultralytics deep-copies its predictor model; a bound callback had copied the external counter object. No NMS observer/counterfactual or validation image had run in that attempt. The failed source, request ledger and traceback summary are preserved. A closure-based callback plus a deepcopy regression test fixed accounting;103 synthetic tests passed. The successful run then received the normal equal three full-frame warmups per size. **Actual completed detector application requests including this failure are667**, one above the planned666. The failed warmup elapsed/root-forward count was not retained by its broken counter; do not represent the666 directly counted forwards as complete measurement of that failed attempt. See counter_instrumentation_amendment.json.','',
        '| Size | Canonical validation mean predict ms (10 images) | Observed primary mean predict ms (320 images) |','|---|---:|---:|']
    for size in [640,960]:
        t=costs['diagnostic_prediction_wall_by_kind']
        report.append(f"| {size} | {t[str(size)+'_canonical_validation']['mean_ms']:.6f} | {t[str(size)+'_observed_primary']['mean_ms']:.6f} |")
    report+=['',
        f"Predict clocks cover model.predict preprocessing/forward/NMS/results, with observation overhead for primary calls; they exclude subsequent NumPy/LUT copying and raw-file serialization. Primary decode sums {costs['primary_decode_seconds']:.3f}s and raw serialization {costs['primary_raw_serialization_seconds']:.3f}s are separately recorded. These are one640-then960 diagnostic run with small canonical samples, not counterbalanced E timings, end-to-end throughput, thermal analysis or streaming AP.",'',
        '## Implications and next approval','',
        'Confirmed on this sample: input resolution materially limits detector accuracy; stronger box/class evidence raises AP, recall and high-confidence precision at960. The original640 global cap has negligible scored impact. Existing mapping/evaluation remain consistent. Still unresolved: exact allocation among spatial resolution, scale-dependent model behaviour, domain/blur/occlusion effects; full-dataset960 performance;960 cap loss; temporal960 accuracy and total runtime. No single resolution is declared the deployment winner.','',
        'Expected input-pixel arithmetic ratio is2.1923 for960x608 vs640x416. At half as many960 calls, detector arithmetic would be about1.096 times Dense640 before tracking. This is not measured latency/energy and cannot justify claiming both higher AP and lower total cost. The diagnostic timings also show substantial per-call cost growth but do not price a960 adaptive system.','',
        'Retain the practical640 fastpath reference. Experiment E remains pending approval and stays CPU-only at640: prioritize new Dense versus practical reference measurements; use the pre-fastpath ablation, high-AP weak-birth and stricter-precision configurations as explanations of the trade-off. Explicitly report FULL versus SKIP latency, decode-inclusive throughput, paired block differences, and performance drift/possible thermal effects. Drift alone cannot establish thermal causation; record run order/power/interruptions and do not silently pool it away. No new960 temporal method, cap change, tracker or scheduler is added to E.','',
        '## Reproduction and artifacts','',
        '```text',
        'PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -q',
        'PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.run --stage prepare',
        'PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.run --stage infer',
        'PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.run --stage score',
        'PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.analyze',
        'PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_detector_D.report',
        '```','',
        'Completed stages refuse overwrites; fresh reproduction needs a separately specified destination. The original failed first invocation and subsequent correction are documented rather than replayed. plan.json, runtime.json, requests.jsonl, canonical_validation*.npz, native80.npz, raw/*.npz, official tensors/matches, summary.csv/summary.json, GT_candidate_diagnostics.jsonl.gz and execution_costs.json retain the full evidence. Final source snapshots and artifact hashes accompany COMPLETE.json. Scoring helper new_YOLO_inferences=0 denotes that scoring stage only; whole-D actual requests are667, not zero.']
    text='\n'.join(report)+'\n';(ROOT/'report.md').write_text(text);doc.write_text(text)
    final=ROOT/'final_source';shutil.copytree(BASE/'analysis/argoverse_hd_detector_D',final/'analysis',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(BASE/'tests/test_detector_D.py',final/'test_detector_D.py')
    dump(ROOT/'final_analysis_sources.json',{str(p):sha(p) for p in (BASE/'analysis/argoverse_hd_detector_D').glob('*.py')})
    for cmd,name in [(['git','status','--short'],'final_git_status.txt'),(['git','diff','--binary'],'final_git_diff.patch')]:
        (ROOT/name).write_bytes(subprocess.check_output(cmd,cwd=BASE))
    result={'saved_live640_bitwise_all320':True,'observer_canonical_exact10_per_size':True,'all640_native_top300_reconstructions_exact':True,
        'independent_saved_sidecar_reconstruction_and_mapping_exact':True,'actual_CPU_FP32_shapes_and_common_predictor_settings_verified':True,
        'selected320_input_hashes_unchanged':True,'all_existing_production_files_unchanged':True,'prior_artifact_hashes_unchanged':prior_counts,
        'Git_HEAD_unchanged':True,'successful_application_requests':666,'actual_application_requests_including_failed_warmup':667,
        'directly_counted_root_forwards':666,'failed_warmup_root_counter_unavailable':True,'synthetic_tests_passed':103,
        'official_scored_cases':list(x['metrics']),'no_extra_NMS_calls_for_shadow':True,'shadow640_fidelity_valid':True,
        'E_started':False,'Phase4_training_commit_push':False,'finalization_audit_wall_seconds':time.perf_counter()-started}
    dump(ROOT/'final_verification.json',result)
    dump(ROOT/'COMPLETE.json',{'D_complete':True,'E_requires_separate_approval':True,'completed_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'actual_application_requests_including_failed_warmup':667})
    hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file()}
    dump(ROOT/'artifact_sha256.json',hashes)
    print('D final audit passed; report:',doc,'; E not started',flush=True)


if __name__=='__main__':main()
