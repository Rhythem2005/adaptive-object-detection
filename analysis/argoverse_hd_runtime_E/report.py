"""Post-run E analysis only. No model imports, inference, replay or rescoring."""
import csv
from pathlib import Path
import statistics

import numpy as np

from analysis.argoverse_hd_phase2.common import archive, clean, dump, load, sha
from .run import BASE, ROOT, BLOCKS, verify_sources

NAMES={'A':'Dense640','B':'Practical fastpath','C':'Pre-fastpath',
       'D':'Weak births (high AP)','E':'Strict association'}
SCOPES=['decode_ms','step_wall_ms','decode_inclusive_ms','decode_cpu_ms','step_cpu_ms',
        'decode_inclusive_cpu_ms','t_detect_ms','detector_cpu_ms','t_signal_ms',
        't_propagate_ms','t_decide_ms','t_track_ms','non_detector_ms','non_detector_cpu_ms']


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return {'n':0}
    return dict(n=len(a),sum=float(a.sum()),mean=float(a.mean()),
        p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),
        p99=float(np.percentile(a,99)),maximum=float(a.max()))


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,row))+'|' for row in rows])


def preserve_record(path, value):
    """A report wording revision may reuse identical analysis, never replace it."""
    if path.exists():
        assert load(path)==clean(value),f'Report would change preserved analysis: {path}'
    else:
        dump(path,value)


def main():
    assert (ROOT/'EXECUTION_COMPLETE.json').exists()
    verify_sources()
    corrections=load(ROOT/'startup_corrections.json')
    for path,digest in corrections['preserved_files'].items():assert sha(path)==digest,path
    frameplan=load(ROOT/'frames.json')
    sequences=list(dict.fromkeys(r['video'] for r in frameplan))
    allrows={m:[] for m in NAMES}; passes={m:[] for m in NAMES}; stats={}; runtimes=[]
    for block,order in enumerate(BLOCKS,1):
        for position,m in enumerate(order,1):
            folder=ROOT/'passes'/f'{block:02d}_{m}'
            meta=load(folder/'COMPLETE.json');runtime=load(folder/'runtime.json')
            assert runtime['actual_device']=='cpu' and not runtime['model_fp16']
            assert (runtime['torch_intra'],runtime['torch_interop'],runtime['opencv_threads'])==(8,1,10)
            runtimes.append(runtime)
            with (folder/'frames.csv').open() as f: rows=list(csv.DictReader(f))
            assert len(rows)==3416
            for row in rows:
                for field in SCOPES[:-2]:row[field]=float(row[field])
                row['non_detector_ms']=row['step_wall_ms']-row['t_detect_ms']
                row['non_detector_cpu_ms']=row['step_cpu_ms']-row['detector_cpu_ms']
                row['block']=block;row['position']=position
            allrows[m].extend(rows)
            per=dict(block=block,position=position,meta=meta,
                scopes={scope:distribution([r[scope] for r in rows]) for scope in SCOPES},
                decisions={decision:{scope:distribution([r[scope] for r in rows if r['decision']==decision])
                    for scope in SCOPES} for decision in ['FULL','SKIP']},
                per_sequence={seq:{scope:distribution([r[scope] for r in rows if r['video']==seq])
                    for scope in SCOPES} for seq in sequences})
            per['processing_fps']=1000/per['scopes']['decode_inclusive_ms']['mean']
            per['loop_fps']=3416/meta['loop_wall_s']
            passes[m].append(per)
    # These settings must be identical across every fresh process, not just asserted in each worker.
    for key in ['versions','torch_intra','torch_interop','opencv_threads','opencv_opencl',
        'thread_environment','actual_device','model_fp16','model_format','model_backend_class',
        'model_fused','resolved_predict_args','nms_backend','checkpoint_sha256','component_devices']:
        assert all(r[key]==runtimes[0][key] for r in runtimes),key
    accuracy={m:load(ROOT/'accuracy'/m/'metrics.json') for m in NAMES}
    image_video={r['image_id']:r['video'] for r in frameplan}
    target_counts={seq:0 for seq in sequences}
    for annotation in load(ROOT/'ground_truth.json')['annotations']:
        if not annotation.get('iscrowd',0):target_counts[image_video[annotation['image_id']]]+=1
    assert sum(target_counts.values())==55412
    for m in NAMES:
        rows=allrows[m]
        matches=archive(ROOT/'accuracy'/m/'official_matches.npz')
        fixed={};per_seq_fixed={seq:{} for seq in sequences}
        for conf in ['0.001','0.1','0.25']:
            counts=matches['counts_conf'+conf]
            tp,fp=counts[:,0,:].sum(axis=0)
            fixed[conf]=dict(TP=int(tp),FP=int(fp),precision=float(tp/(tp+fp)),
                recall=float(tp/55412),F1=float(2*tp/(tp+fp+55412)))
            for seq in sequences:
                mask=np.array([r['video']==seq for r in frameplan]);stp,sfp=counts[mask,0,:].sum(axis=0)
                per_seq_fixed[seq][conf]=dict(TP=int(stp),FP=int(sfp),
                    precision=float(stp/(stp+sfp)) if stp+sfp else None,
                    recall=float(stp/target_counts[seq]),F1=float(2*stp/(stp+sfp+target_counts[seq])))
        stats[m]=dict(name=NAMES[m],accuracy=accuracy[m],fixed_points=fixed,
            pooled={s:distribution([r[s] for r in rows]) for s in SCOPES},
            decisions={d:{s:distribution([r[s] for r in rows if r['decision']==d])
                for s in SCOPES} for d in ['FULL','SKIP']},
            per_sequence={seq:dict(scopes={s:distribution([r[s] for r in rows if r['video']==seq]) for s in SCOPES},
                fixed=per_seq_fixed[seq],COCO=accuracy[m]['per_sequence'][seq]) for seq in sequences},
            passes=passes[m],
            process_memory_mib={k:distribution([p['meta'][k] for p in passes[m]]) for k in
                ['startup_peak_mib','postwarmup_peak_mib','measured_loop_peak_mib','entire_worker_peak_mib']})
    comparisons={}
    for candidate,reference in [('B','A'),('C','A'),('D','A'),('E','A'),('B','C'),('D','B'),('E','B')]:
        comparison={}
        for scope in ['step_wall_ms','decode_inclusive_ms','step_cpu_ms','decode_inclusive_cpu_ms','non_detector_ms']:
            ratios=[1-c['scopes'][scope]['mean']/r['scopes'][scope]['mean']
                for c,r in zip(passes[candidate],passes[reference])]
            comparison[scope]=dict(paired_reduction_fraction=ratios,mean=float(np.mean(ratios)),
                minimum=min(ratios),maximum=max(ratios),
                pooled_reduction=1-stats[candidate]['pooled'][scope]['mean']/stats[reference]['pooled'][scope]['mean'])
        comparisons[candidate+'_vs_'+reference]=comparison
    preserve_record(ROOT/'summary.json',dict(methods=stats,comparisons=comparisons,runtime=runtimes[0],
        scope='CPU FP32 live E, 5 blocks; accuracy first-pass scored, all25 exact archives; no streaming AP/energy/temperature.',
        accounting=load(ROOT/'EXECUTION_COMPLETE.json'),startup_corrections=corrections))
    lines=['# Experiment E — live CPU accuracy and processing cost','',
        '**Retain B as the practical reference.** Its decode-inclusive mean falls from24.0495 to16.1027ms/frame '
        '(33.0435% less wall time;41.581→62.101FPS), with36.5029% less process CPU and49.9415% fewer detector requests. '
        'AP remains14.692246% versus Dense15.732713%; the1.040467-point gap is unresolved. '
        'FULL latency is higher than Dense; inexpensive SKIP frames provide the aggregate saving.','',
        'Completed under the approved 640-only scope. Stock YOLOv8n, CPU FP32, unchanged detector/tracker/scheduler. '
        'All 25 passes reproduce their archived prediction arrays and details byte for byte, schedules exactly, '
        'and temporal sequence-end tracker hashes exactly. The five first passes reproduce official COCO metrics, '
        'per-sequence metrics, operating points and precision/recall/score tensors exactly. Later passes reuse these scores only after exact prediction fidelity.','',
        'Each pass processes all 3,416 original 1920×1200 images in five ordered sequences (901,901,469,475,670 frames), '
        'with tracking state reset at every boundary. The shared provided-annotation evaluation contains 55,412 regular objects '
        'and 2,299 crowds across eight categories; no pseudo-labels, empty-frame removal or altered taxonomy.','',
        '## Accuracy, precision and request budget','',
        'AP is official COCO maxDets [1,10,100], all eight mapped categories, original-pixel coordinates, crowds/empty frames retained. '
        'P/R/F1 below are micro at confidence≥0.25 and IoU50; FP0.10 uses the same official matching at confidence≥0.10. '
        'These fixed operating points are separate from AP and COCO AR100.','']
    lines.append(table(['Arm','AP%','AP50%','AP75%','AR100%','P25%','R25%','F1%','TP25','FP25','TP10','FP10','Calls/pass'],[
        [m+' '+NAMES[m],*[f"{accuracy[m]['COCO'][k]*100:.6f}" for k in ['AP','AP50','AP75','AR100']],
         *[f"{stats[m]['fixed_points']['0.25'][k]*100:.4f}" for k in ['precision','recall','F1']],
         stats[m]['fixed_points']['0.25']['TP'],stats[m]['fixed_points']['0.25']['FP'],
         stats[m]['fixed_points']['0.1']['TP'],stats[m]['fixed_points']['0.1']['FP'],3416 if m=='A' else 1710] for m in NAMES]))
    lines += ['', 'Category-macro operating points at confidence0.25/IoU50 (not COCO AP):','']
    lines.append(table(['Arm','Macro P%','Macro R%','Macro F1%','Micro P10%','Micro R10%'],[
        [m,*[f"{accuracy[m]['fixed_conf25'][k]*100:.4f}" for k in ['macro_precision','macro_recall','macro_F1']],
         *[f"{stats[m]['fixed_points']['0.1'][k]*100:.4f}" for k in ['precision','recall']]] for m in NAMES]))
    lines += ['', 'All fixed arms save 49.941452% of detector requests. B/C output fidelity makes their comparison an accuracy-preserving preprocessing ablation. '
        'D retains the archived weak-birth/4×4 flow settings; E retains stricter low-stage association/3×3 flow. '
        'D/E use the same verified fixed-signal fastpath as B. No confidence, cap, resolution or class-filter change was introduced.','',
        '## Timing and CPU cost','',
        'Numbers pool equal-size passes over five blocks. FPS uses 3,416 divided by the summed relevant frame scope per pass; '
        'it is sequential file processing, without playback pacing. Pipeline service externally includes step return and diagnostic detail construction. '
        'Decode-inclusive adds local JPEG read/decode and shape/type validation. CPU time is process time across threads, so it can exceed elapsed wall time. '
        'Residual non-detector service includes signals, flow/Kalman propagation, association/output construction, controller and Python overhead; it is not pure tracker time.','']
    lines.append(table(['Arm','Decode mean ms','Step mean ms','Decode-inclusive mean ms','Processing FPS','Step CPU ms/frame','Inclusive CPU ms/frame','Non-detector wall / CPU ms/frame','Loop FPS'],[
        [m,*[f"{stats[m]['pooled'][s]['mean']:.4f}" for s in ['decode_ms','step_wall_ms','decode_inclusive_ms']],
         f"{1000/stats[m]['pooled']['decode_inclusive_ms']['mean']:.3f}",
         *[f"{stats[m]['pooled'][s]['mean']:.4f}" for s in ['step_cpu_ms','decode_inclusive_cpu_ms']],
         ' / '.join(f"{stats[m]['pooled'][s]['mean']:.4f}" for s in ['non_detector_ms','non_detector_cpu_ms']),
         f"{3416/statistics.mean(p['meta']['loop_wall_s'] for p in passes[m]):.3f}"] for m in NAMES]))
    lines += ['', 'Pooled frame distributions (ms; correlated frames are not independent experimental repetitions):','']
    lines.append(table(['Arm','Decision (frames over5passes)','Step mean / median / P95 / P99','Inclusive mean / median / P95 / P99','Step CPU mean'],[
        [m,f"{d} ({stats[m]['decisions'][d]['step_wall_ms']['n']})",
         ' / '.join(f"{stats[m]['decisions'][d]['step_wall_ms'][k]:.4f}" for k in ['mean','p50','p95','p99']),
         ' / '.join(f"{stats[m]['decisions'][d]['decode_inclusive_ms'][k]:.4f}" for k in ['mean','p50','p95','p99']),
         f"{stats[m]['decisions'][d]['step_cpu_ms']['mean']:.4f}"]
        for m in NAMES for d in ['FULL','SKIP'] if stats[m]['decisions'][d]['step_wall_ms']['n']]))
    lines += ['', 'Measured internal telemetry, averaged across all frames; nested scopes are not added twice:','']
    lines.append(table(['Arm','Detector FULL mean ms','Signal mean ms/frame','Propagation mean ms/frame','Decision mean ms/frame','Association/output mean ms/frame'],[
        [m,f"{stats[m]['decisions']['FULL']['t_detect_ms']['mean']:.4f}",
         *[f"{stats[m]['pooled'][s]['mean']:.4f}" for s in ['t_signal_ms','t_propagate_ms','t_decide_ms','t_track_ms']]] for m in NAMES]))
    lines += ['', 'Block-paired reductions; positive values mean less cost. Each arm occupied each order position exactly once; predecessor pairs are not fully balanced. '
        'the five blocks are repetitions, not five independent datasets. No significance or thermal-causality claim is made.','']
    lines.append(table(['Comparison','Step reduction% (five blocks)','Inclusive reduction% (five blocks)','CPU inclusive reduction% (five blocks)'],[
        [name,*[', '.join(f'{v*100:.2f}' for v in comparisons[name][scope]['paired_reduction_fraction'])
          for scope in ['step_wall_ms','decode_inclusive_ms','decode_inclusive_cpu_ms']]] for name in comparisons]))
    lines += ['', '## Performance drift, memory and execution conditions','',
        'Block orders: `ABDCE`, `BCEDA`, `CDAEB`, `DEBAC`, `EACBD`; 30-second pauses separate blocks. '
        'Fresh interpreter/model per pass; three excluded full-frame warmups, no ROI warmup. '
        'No concurrent scoring/tests/heavy diagnostics during timed work; first-block official scoring occurs between blocks. '
        'Read/hash validation is outside timing and can warm filesystem caches. Storage is not forced cold. '
        'An archive-schema guard stopped after first-block Dense scoring: Phase1 stores four extra axis-metadata arrays. '
        'The continuation validated all those arrays and the three result tensors exactly, completed the remaining first-block scores, '
        'then resumed blocks2–5 with the frozen worker. No timing pass was repeated. '
        'This investigation lengthened the first inter-block gap; all raw timestamps and the interruption traceback are retained.','']
    lines.append(table(['Arm','Inclusive mean ms by block1→5','Last/first change%','Step mean ms by block1→5','Loop peak RSS MiB range','Whole worker peak RSS MiB range'],[
        [m,', '.join(f"{p['scopes']['decode_inclusive_ms']['mean']:.4f}" for p in passes[m]),
         f"{(passes[m][-1]['scopes']['decode_inclusive_ms']['mean']/passes[m][0]['scopes']['decode_inclusive_ms']['mean']-1)*100:.2f}",
         ', '.join(f"{p['scopes']['step_wall_ms']['mean']:.4f}" for p in passes[m]),
         f"{min(p['meta']['measured_loop_peak_mib'] for p in passes[m]):.1f}–{max(p['meta']['measured_loop_peak_mib'] for p in passes[m]):.1f}",
         f"{min(p['meta']['entire_worker_peak_mib'] for p in passes[m]):.1f}–{max(p['meta']['entire_worker_peak_mib'] for p in passes[m]):.1f}"] for m in NAMES]))
    lines += ['', 'Peak RSS is the macOS process high-water mark (bytes converted to MiB), includes imports/model and buffered predictions; '
        'the whole-worker peak additionally includes archive validation and serialization. Loop peak was read before those postprocessing steps. '
        'It is not incremental tracker allocation or total system memory. Startup/postwarmup values and per-pass costs are retained in `COMPLETE.json`.','',
        '`pmset` power and warning outputs are retained at every pass boundary. They consistently report AC Power, while the battery percentage declined '
        'from55% at the first accepted pass to36% at completion and was reported as discharging. Captured AC settings show lowpowermode0. '
        'The physical charging condition was not independently measured or changed. This Mac did not expose usable thermal warning/CPU power status through `pmset -g therm`; '
        'no chip temperature, frequency, joules or throttling event was measured. Drift can reflect heating, OS contention, order and workload; it cannot uniquely identify thermal effects.','',
        '## Per-sequence results','',
        'Per-sequence AP is evaluated separately with official category support rules; aggregate AP is pooled COCO AP, not its average. '
        'Latency below pools five repeated passes for that sequence.','']
    lines.append(table(['Sequence (frames)','Arm','AP%','AP50%','P25%','R25%','F1%','TP25 / FP25','Inclusive mean / P95 ms','Step mean ms','CPU inclusive ms/frame'],[
        [f"{seq} ({sum(r['video']==seq for r in frameplan)})",m,
         f"{stats[m]['per_sequence'][seq]['COCO']['AP']*100:.6f}",
         f"{stats[m]['per_sequence'][seq]['COCO']['AP50']*100:.6f}",
         f"{stats[m]['per_sequence'][seq]['fixed']['0.25']['precision']*100:.4f}",
         f"{stats[m]['per_sequence'][seq]['fixed']['0.25']['recall']*100:.4f}",
         f"{stats[m]['per_sequence'][seq]['fixed']['0.25']['F1']*100:.4f}",
         f"{stats[m]['per_sequence'][seq]['fixed']['0.25']['TP']} / {stats[m]['per_sequence'][seq]['fixed']['0.25']['FP']}",
         ' / '.join(f"{stats[m]['per_sequence'][seq]['scopes']['decode_inclusive_ms'][k]:.4f}" for k in ['mean','p95']),
         f"{stats[m]['per_sequence'][seq]['scopes']['step_wall_ms']['mean']:.4f}",
         f"{stats[m]['per_sequence'][seq]['scopes']['decode_inclusive_cpu_ms']['mean']:.4f}"] for seq in sequences for m in NAMES]))
    total=load(ROOT/'EXECUTION_COMPLETE.json')
    lines += ['', '## Reproducibility and actual execution cost','',
        f"Accepted run: {total['passes']} passes /85,400 frames,51,280 measured detector requests +75 excluded warmups =51,355 requests and root forwards. "
        'Three preserved pre-sequence startup failures used nine additional warmups, so **51,364 actual E requests/forwards including failed startup validation**. '
        'No extra full passes, 960 inference, temporal replay from caches, cap changes, training or Phase4 work.','',
        f"Accepted-run elapsed wall span: {total['wall_s']:.3f}s from the first execution marker to completion, including workers, first-block scoring, "
        'the evaluation-schema investigation interruption, hash/output validation, serialization and four pauses. '
        f"Stopped startup orchestrator wall: {corrections['failed_orchestrator_wall_s']:.3f}s. "
        'Implementation, synthetic tests, preparation hashing and final report/audit are separate development costs.','',
        'Initial startup guards exposed: Ultralytics lazy imports reset OpenCV threads; installed predictor removed `half` in favour of `quantize`; '
        'AutoBackend renamed the older `pt` flag to `format`. These were harness compatibility errors, resolved without changing algorithms/inference settings. '
        'All stopped attempts remain in E_v1/E_v2/E_v3 and are hashed by `startup_corrections.json`. '
        'The corrected 110-test synthetic suite passed before accepted execution;112 tests passed after adding the axis-metadata fidelity checks, '
        'before the continuation. One attempted test-module command failed to import the non-package tests path; '
        'the discovery command corrected it. Draft report finalization also hit an exclusive-create guard; '
        'the preserved analysis JSON is reused only after exact equality verification, with no rescoring or inference. '
        'Development logs and the initial report draft are retained.','',
        'Mean per-pass wall scopes (seconds); subprocess cost includes interpreter/import startup, unlike the worker invocation clock. '
        'Validation/serialization/scoring are excluded from per-frame service throughput. These are nested or disjoint scopes as named, not quantities to sum indiscriminately.','']
    lines.append(table(['Arm','Setup','Warmups','Processing loop','Post-loop validation','Serialization','Worker invocation','Subprocess including imports','Loop CPU seconds'],[
        [m,*[f"{statistics.mean(p['meta'][k] for p in passes[m]):.3f}" for k in ['setup_wall_s','warmup_wall_s',
            'loop_wall_s','validation_wall_s','serialization_wall_s','invocation_wall_s']],
         f"{statistics.mean(load(ROOT/'passes'/f'{block:02d}_{m}'/'SUBPROCESS.json')['wall_including_interpreter_s'] for block in range(1,6)):.3f}",
         f"{statistics.mean(p['meta']['loop_cpu_s'] for p in passes[m]):.3f}"] for m in NAMES]))
    scoring_cost=load(ROOT/'scoring_cost.json')
    lines += ['',f"Five first-pass official scoring calls took {scoring_cost['all_five_scoring_wall_s']:.3f}s in scorer wall clocks; later identical passes were not rescored.",'',
        'Commands (from repository root):','',
        '```sh','PYTHONPATH=extracted:. .venv/bin/python -m unittest discover -s tests -v',
        'PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_runtime_E.run prepare',
        'PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_runtime_E.run run',
        '# First-block schema guard stopped scoring, not timing; validate metadata and continue remaining passes:',
        'PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_runtime_E.resume',
        'PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_runtime_E.report','```','',
        'Historical E_v1/v2/v3 preparation commands used earlier harness snapshots and are preserved; exclusive-create guards prevent overwriting/repeating completed execution. '
        'Each subprocess invocation is recorded in `passes/*/SUBPROCESS.json`. The E_v4 execution plan, frozen configs, source snapshot, '
        'source/prior hashes, input/GT manifest, raw per-frame CSVs, saved predictions/details, official scoring, power logs and summaries provide the reproducibility record.','',
        'Actual common runtime: '+runtimes[0]['platform']+'; Python3.14.6, PyTorch2.14.0, Ultralytics8.4.149, OpenCV5.0.0.93, pycocotools2.0.11. '
        'Fused native PyTorch CPU backend, FP32, input1×3×416×640, class-aware torchvision NMS, all80 detector classes before the existing8-class map. '
        'PyTorch intra8/inter-op1, OpenCV10/OpenCLoff; all recorded runtime fields match across25 processes. '
        'Decode, flow, tracker, scheduler and offline COCO scoring run on CPU.','',
        'New files are analysis-only E runner/continuation/report/audit and synthetic tests, fresh E run directories and this report. '
        'No production algorithm/config file or prior result was edited; prior protected hashes are checked before/after execution and in final audit. '
        'No commit/push.','',
        '## Interpretation, limits and next direction','',
        '**Confirmed efficiency benefit, with an accuracy trade-off.** B reduces pipeline wall time41.8707% and inclusive wall33.0435%; '
        'five paired inclusive reductions span32.47–33.90%, and CPU-inclusive reductions35.54–37.32%. '
        'At confidence0.25, precision rises1.0145 points versus Dense, with367 fewer FPs and353 fewer TPs; recall falls0.6370 points. '
        'Fewer FPs accompany lower recall; E did not optimize accuracy.','',
        '**FULL and SKIP differ.** B FULL service averages20.0231ms versus Dense18.8548ms; '
        'inclusive FULL averages25.1433ms versus24.0495ms. FULL-inclusive P95 is26.4245ms versus Dense25.3617ms. '
        'SKIP service averages1.8761ms (inclusive7.0410ms), generating the throughput gain. '
        'A lower aggregate mean does not establish lower latency on every frame or streaming deadline compliance.','',
        '**Tracking is a bounded, visible overhead.** B spends1.3657ms/frame on all non-detector service, including0.5097ms signal/flow preparation, '
        '0.6251ms propagation/Kalman and0.1918ms update/output telemetry. The residual is12.46% of service and8.48% of inclusive processing; '
        'its CPU residual is3.0692ms/frame. This includes orchestration/output work, not pure LK or association. Eliminating that entire residual '
        'could remove only8.48% of current inclusive wall—an accounting upper bound, not an achievable optimization forecast. '
        'Detector and decode account for about59.58% and31.94% of inclusive time.','',
        '**The fastpath survives real inference.** B/C predictions/details and accuracy/schedules are identical; their only frozen config difference is fixed_signal_fastpath. '
        'Signal cost falls2.0221→0.5097ms/frame. Live step wall falls11.9493% and inclusive wall8.0920%, with all five paired inclusive improvements positive. '
        'Inclusive CPU falls only1.4364% (paired0.53–2.65%); detector-phase CPU charging also differs between live arms. These multithreaded process clocks '
        'do not identify instruction/energy savings or explain CPU redistribution between phases. The old cache-only55% wall reduction is not a live-pipeline speedup.','',
        '**Weak births buy AP at a precision/work cost.** D adds0.398819 AP points over B, remaining0.641648 below Dense. At0.10 it adds1,229TPs '
        'and5,441FPs (+35.70% FP), reducing precision56.2098→50.1326%. At0.25 it loses10TPs and adds10FPs. '
        'It takes1.8367% more inclusive wall and1.9440% more CPU than B; SKIP service is27.73% higher. '
        'Propagation increases0.6251→0.8585ms/frame, consistent with the previously archived larger track/flow workload. '
        'Retain it as the accuracy-oriented comparison, not the practical default.','',
        '**Strict association offers a lower-score precision trade-off, without a demonstrated runtime win.** E removes484FPs and141TPs at0.10 versus B; '
        'AP falls0.010221 points. Its0.25 TP/FP counts are identical. Inclusive wall differs only0.1577% in E’s favour, paired signs vary, '
        'and CPU is0.2180% higher. This is not a reliable efficiency advantage.','',
        '**Drift is small and not monotonic.** First→last inclusive means change−0.26%,−0.60%,−0.95%,−1.36%,−1.67% forA–E. '
        'No method shows progressive mean slowdown. This does not prove absence of heating/throttling: temperature was unavailable, '
        'the first scoring/validation gap was longer, and OS/power effects were not isolated.','',
        '**Next direction: detector evidence quality and scale under a measured compute budget, with held-out validation.** '
        f"The completed [D diagnosis]({BASE/'docs/DETECTOR_D_RESULTS.md'}) found paired320-frame640→960 AP16.504911→21.683948%, "
        'stronger small/medium evidence and a negligible640 cap-removal effect. These remain diagnostic sample results, not full-dataset960 or temporal results. '
        'Together with E, this supports studying where better detector/localisation evidence is worth its real cost before another tracker/scheduler redesign. '
        'Do not infer that half-rate960 is cheaper: D’s padded-pixel arithmetic exceeds Dense640 at half the calls before tracking, and its diagnostic timing '
        'does not price a temporal pipeline. Keep stock weights, B as the practical control, D as the high-AP control, official AP plus fixed-threshold FP/TP '
        'and actual inclusive wall/CPU as co-primary outcomes. Require previously unused sequences before claiming generalization. '
        'Any scale-allocation/960 temporal experiment or algorithm change needs separate approval; none was implemented. Phase4 remains unstarted.','',
        f"Implementation references: [E timing worker]({BASE/'analysis/argoverse_hd_runtime_E/run.py'}:198), "
        f"[schema-aware continuation]({BASE/'analysis/argoverse_hd_runtime_E/resume.py'}:21), "
        f"[official scorer]({BASE/'analysis/argoverse_hd_phase2/scoring.py'}:40), "
        f"[pipeline timing/output scopes]({BASE/'extracted/adaptive_pipeline/pipeline.py'}:216), "
        f"[detector wrapper]({BASE/'extracted/adaptive_pipeline/detection/detector.py'}:36). "
        'Exact arm settings are in E_v4/configs/A.json throughE.json; absolute archived source paths are in execution_plan.json.','',
        'Five previously studied sequences are an in-sample benchmark. Repetition tests runtime stability, not new-scene generalization. '
        'CPU process time is a computational-cost proxy, not energy or FLOP measurement. Timed steps include small common passive counter/CPU-timer overhead; '
        'no heavy tensor capture, hashing, serialization or GT scoring occurs inside their scope. '
        'Sequential file throughput is not capture-to-output latency, streaming AP or deadline compliance; synchronous processing may outrun or lag nominal video arrival. '
        'CPU-only conclusions do not establish MPS/GPU/export performance. No confidence tuning or candidate selection on new holdout data occurred.','']
    report=BASE/'docs/EXPERIMENT_E_RESULTS.md'
    report.write_text('\n'.join(lines))
    preserve_record(ROOT/'REPORT_GENERATED.json',dict(report=str(report),summary=str(ROOT/'summary.json')))
    print(report)


if __name__=='__main__':main()
