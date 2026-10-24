"""Saved-output analysis only. Run after timed execution; no detector imports."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from analysis.argoverse_hd_phase2.common import load,dump,sha,clean
from analysis.branch_b_expanded_v1.continue_local import BASE,ROOT,read_rows,verify_pass
from analysis.latest_frame_fixed2_validation import held_outputs,score_arrays
from adaptive_pipeline.run_stream import distribution


def csv_out(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def offline():
    e=load(BASE/'runs/argoverse_hd_real_runtime_E_v4/summary.json')
    o=load(BASE/'runs/argoverse_hd_coco_phase1_v1/fixed2_093/metrics.json')
    original=load(BASE/'runs/argoverse_hd_benchmark_v1/fixed2_093/metrics.json')
    plan=load(ROOT/'manifests/frames.json');out={};table=[];perseq=[]
    for name,key in [('dense','A'),('original_fixed2',None),('practical','B'),('high_ap','D')]:
        config=load(ROOT/f'configs/{name}.json')
        if key:
            v=e['methods'][key];accuracy=v['accuracy']['COCO'];point=v['accuracy']['fixed_conf25']
            mean=v['pooled']['decode_inclusive_ms']['mean'];cpu=v['pooled']['decode_inclusive_cpu_ms']['mean']
            entry=dict(COCO=accuracy,fixed_conf25=point,fps=1000/mean,wall_ms_frame=mean,cpu_ms_frame=cpu,
                calls=3416 if key=='A' else 1710,timing_repetitions=5,
                cpu_scope='decode-inclusive measured frame loop',timing_scope='JPEG decode plus pipeline return',
                source='runs/argoverse_hd_real_runtime_E_v4/summary.json',
                per_sequence={seq:dict(COCO=s['COCO'],fixed_conf25=s['fixed']['0.25'],
                    fps=1000/s['scopes']['decode_inclusive_ms']['mean'],
                    cpu_ms_frame=s['scopes']['decode_inclusive_cpu_ms']['mean']) for seq,s in v['per_sequence'].items()})
        else:
            with (BASE/'runs/argoverse_hd_benchmark_v1/fixed2_093/frames.csv').open() as f:rows=list(csv.DictReader(f))
            inclusive=[float(r['t_total_ms'])+float(r['decode_ms']) for r in rows]
            entry=dict(COCO=o['official_COCO_max100'],fixed_conf25=o['supplementary_fixed_conf25_IoU50'],
                fps=1000/np.mean(inclusive),wall_ms_frame=float(np.mean(inclusive)),
                cpu_ms_frame=original['efficiency']['cpu_ms_per_frame'],calls=1710,timing_repetitions=1,
                cpu_scope='historical whole evaluation process; broader than E; not paired/comparable',
                timing_scope='historical decode plus internal pipeline; different campaign/instrumentation',
                source='runs/argoverse_hd_coco_phase1_v1/fixed2_093/metrics.json',
                per_sequence={seq:dict(COCO=s['metrics'],fixed_conf25=None,
                    fps=1000/np.mean([float(r['t_total_ms'])+float(r['decode_ms']) for r in rows if r['video']==seq]),
                    cpu_ms_frame=None) for seq,s in o['per_sequence'].items()})
        entry.update(config=config,config_sha256=sha(ROOT/f'configs/{name}.json'),frames=3416,reused=True)
        out[name]=entry
        table.append(dict(method=name,frames=3416,calls=entry['calls'],fps=entry['fps'],
            wall_ms_frame=entry['wall_ms_frame'],cpu_ms_frame=entry['cpu_ms_frame'],cpu_scope=entry['cpu_scope'],
            timing_repetitions=entry['timing_repetitions'],**{k:entry['COCO'][k] for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large']},
            precision=entry['fixed_conf25']['micro_precision'],recall=entry['fixed_conf25']['micro_recall'],F1=entry['fixed_conf25']['micro_F1']))
        for seq,s in entry['per_sequence'].items():
            perseq.append(dict(method=name,sequence=seq,frames=sum(r['video']==seq for r in plan),
                fps=s['fps'],cpu_ms_frame=s['cpu_ms_frame'],**{k:s['COCO'][k] for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large']}))
    dump(ROOT/'analysis/offline.json',out)
    csv_out(ROOT/'analysis/offline.csv',table);csv_out(ROOT/'analysis/offline_per_sequence.csv',perseq)
    dump(ROOT/'analysis/offline_reuse_evidence.json',dict(no_new_inference=True,population_frames=3416,
        methods=list(out),original_cpu_not_directly_comparable=True,
        provenance_note='Original Fixed-2 scores are official Phase1 rescoring of its saved whole-dataset predictions. E reuses exact prediction/tensor parity. No rejected method included.'))


def summary_rows():
    """Available while execution runs: read-only inspection; does not score."""
    plan=load(ROOT/'execution_plan.json');rows=[]
    for item in plan['matrix']:
        out=Path(item['reuse']) if item['reuse'] else ROOT/'passes'/item['name']
        status='complete' if (out/'COMPLETE.json').exists() else 'incomplete' if (out/'INCOMPLETE.json').exists() else 'not_started'
        rows.append(dict(item,path=str(out),status=status))
    return rows


def scoring():
    assert (ROOT/'EXECUTION_COMPLETE.json').exists(),'Never overlap official scoring with timed replay'
    import shutil
    sources=[BASE/p for p in ['analysis/branch_b_expanded_v1/analyze_local.py',
        'analysis/branch_b_expanded_v1/cases_local.py','analysis/latest_frame_fixed2_validation.py',
        'analysis/latest_frame_validation.py','analysis/argoverse_hd_phase2/common.py',
        'analysis/argoverse_hd_phase2/scoring.py','analysis/argoverse_hd_runtime_E/run.py']]
    dump(ROOT/'analysis/scoring_sources.json',{str(p.relative_to(BASE)):sha(p) for p in sources})
    snapshot=ROOT/'analysis/source_snapshot';snapshot.mkdir(exist_ok=False)
    for p in sources:
        target=snapshot/p.relative_to(BASE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    start=time.perf_counter();manifest={seq:load(ROOT/'manifests'/f'{seq}.json') for seq in [f'sequence_{i:02d}' for i in range(5)]}
    fullgt=load(ROOT/'manifests/ground_truth.json');gts={}
    (ROOT/'analysis/scores').mkdir(exist_ok=True)
    for seq,frames in manifest.items():
        ids={r['image_id'] for r in frames}
        gts[seq]=dict(fullgt,images=[im for im in fullgt['images'] if im['id'] in ids],
            annotations=[a for a in fullgt['annotations'] if a['image_id'] in ids])
    records=summary_rows();completed={};data={};preds={};assignments={};verification={}
    ref=BASE/'runs/argoverse_hd_benchmark_v1/baseline'
    for item in records:
        if item['status']!='complete':continue
        out=Path(item['path']);name=item['name'];seq=item['sequence']
        verification[name]=verify_pass(out,manifest[seq],ref/'preds.npz',ref/'prediction_details.npz')
        s=load(out/'COMPLETE.json');rows=read_rows(out);events=load(out/'capture_events.json')
        with np.load(out/'preds.npz',allow_pickle=False) as p:arrays={k:p[k] for k in p.files}
        held,assign=held_outputs(rows,events,arrays)
        assignments[name]=assign;data[name]=rows;preds[name]=arrays
        span=s['source_duration_s'];service=sum(float(r['service_ms']) for r in rows)/1000
        s.update(item,cpu_ms_per_source=1000*s['loop_cpu_s']/s['source_frames'],
            cpu_ms_per_consumed=1000*s['loop_cpu_s']/s['processed'],
            service_throughput_fps=len(rows)/service,consumed_fraction=s['processed']/s['source_frames'],
            overwritten=s['buffer']['replaced'],received=s['buffer']['captured'],
            flow_skipped_yolo=s['accepted_flow_steps'],drain_output_count=s['processed']-s['active_output_count'],
            source_gap_counts=dict(collections.Counter(int(r.get('source_gap',1)) for r in rows)),
            visible_age_ms=distribution([a['visible_age_ms'] for a in assign if a['visible_age_ms'] is not None]),
            no_result_queries=sum(a['result_source_frame_id'] is None for a in assign))
        completed[name]=s
    # Cache exact predictions+GT+population: metric reuse requires identical bytes.
    def evaluate(seq,arrays,ids):
        h=hashlib.sha256(json.dumps(gts[seq],sort_keys=True).encode())
        for i in ids:
            a=arrays[i];h.update(str((i,a.dtype.str,a.shape)).encode());h.update(a.tobytes())
        digest=h.hexdigest();out=ROOT/'analysis/scores'/digest
        if not (out/'metrics.json').exists():
            metrics,_=score_arrays(arrays,gts[seq],ids,out)
            dump(out/'population.json',dict(sequence=seq,frame_ids=ids,digest=digest))
        return dict(metrics=load(out/'metrics.json'),score_directory=str(out),population_count=len(ids))
    aligned={};common={}
    for idx,(name,s) in enumerate(completed.items(),1):
        held,assign=held_outputs(data[name],load(Path(s['path'])/'capture_events.json'),preds[name])
        aligned[name]=evaluate(s['sequence'],held,list(range(s['source_frames'])))
        dump(ROOT/'analysis'/(name+'_assignments.json'),assign)
        print(f'Aligned scoring {idx}/{len(completed)} {name}',flush=True)
    groups=collections.defaultdict(list)
    for name,s in completed.items():groups[(s['sequence'],s['fps'],s['trial'])].append(name)
    for group,names in groups.items():
        seq,fps,trial=group;ids=sorted(set.intersection(*[{r['frame_id'] for r in data[n]} for n in names]))
        cell={}
        for name in names:
            byid={r['frame_id']:preds[name][r['key']] for r in data[name]}
            cell[name]=evaluate(seq,{i:byid[i] for i in ids},ids)
        common[f'{seq}_{fps}_r{trial}']=dict(frame_ids=ids,complete_arms=[completed[n]['mode'] for n in names],metrics=cell,
            population='Intersection of consumed frames across complete arms in this cell; conditional, not full-stream AP')
    table=[]
    for name,s in completed.items():
        key=f'{s["sequence"]}_{s["fps"]}_r{s["trial"]}';c=common[key]['metrics'][name]['metrics'];a=aligned[name]['metrics']
        table.append(dict(name=name,dataset='local Argoverse-HD development',sequence=s['sequence'],trial=s['trial'],rate=s['fps'],mode=s['mode'],
            reused=bool(s['reuse']),pacing_valid=s['valid'],received=s['received'],consumed=s['processed'],processed=s['processed'],
            flow_skipped_yolo=s['flow_skipped_yolo'],overwritten=s['overwritten'],calls=s['measured_yolo_calls'],
            active_output_fps=s['active_output_fps'],drain_output_fps=s['output_fps_through_drain'],service_throughput_fps=s['service_throughput_fps'],
            loop_cpu_s=s['loop_cpu_s'],cpu_ms_source=s['cpu_ms_per_source'],peak_rss_mib=s['peak_rss_mib'],peak_pending=s['buffer']['peak_pending'],
            latency_p50_ms=s['capture_to_result_ms']['p50'],latency_p95_ms=s['capture_to_result_ms']['p95'],latency_max_ms=s['capture_to_result_ms']['max'],
            post_eof_drain_ms=s['post_eof_drain_ms'],frame_coverage=s['consumed_fraction'],
            common_frames=len(common[key]['frame_ids']),common_AP=c['COCO']['AP'],common_AP50=c['COCO']['AP50'],common_AP75=c['COCO']['AP75'],
            common_precision=c['fixed_conf25']['micro_precision'],common_recall=c['fixed_conf25']['micro_recall'],common_F1=c['fixed_conf25']['micro_F1'],
            aligned_AP=a['COCO']['AP'],aligned_AP50=a['COCO']['AP50'],aligned_AP75=a['COCO']['AP75'],
            aligned_precision=a['fixed_conf25']['micro_precision'],aligned_recall=a['fixed_conf25']['micro_recall'],aligned_F1=a['fixed_conf25']['micro_F1'],
            aligned_TP=a['fixed_conf25']['counts']['TP'],aligned_FP=a['fixed_conf25']['counts']['FP'],
            forced_full=s['forced_full'],gap_triggered_full=s['gap_triggered_full'],no_result_queries=s['no_result_queries']))
    csv_out(ROOT/'analysis/streaming.csv',table)
    # Joint AP over independent complete streams is a separate population from
    # descriptive per-scene means. Preserve every cohort's pacing-valid flag.
    pooled=[]
    pooled_cache=ROOT/'analysis/pooled_scores';pooled_cache.mkdir(exist_ok=False)
    from adaptive_pipeline.evaluation.coco import prediction_records,make_coco,run_coco,coco_metrics,fixed_operating_point
    for mode in ['fifo','latest','merged']:
        for fps in [15,30,60,90]:
            for trial in [1,2,3]:
                names=[n for n,s in completed.items() if s['mode']==mode and s['fps']==fps and s['trial']==trial]
                if len(names)!=5:continue
                global_arrays={}
                for name in names:
                    s=completed[name];seq=s['sequence']
                    held,_=held_outputs(data[name],load(Path(s['path'])/'capture_events.json'),preds[name])
                    global_arrays.update({r['image_id']:held[r['frame_id']] for r in manifest[seq]})
                assert len(global_arrays)==3416
                digest=hashlib.sha256(json.dumps(fullgt,sort_keys=True).encode())
                for image_id in sorted(global_arrays):
                    a=global_arrays[image_id];digest.update(str((image_id,a.dtype.str,a.shape)).encode());digest.update(a.tobytes())
                out=pooled_cache/digest.hexdigest()
                if not (out/'metrics.json').exists():
                    records_coco,_=prediction_records(global_arrays,{i:i for i in range(8)})
                    cg,cd=make_coco(fullgt,records_coco);ev,log=run_coco(cg,cd)
                    out.mkdir(exist_ok=False)
                    metric=dict(COCO=coco_metrics(ev),fixed_conf25=fixed_operating_point(ev))
                    dump(out/'metrics.json',metric);(out/'coco_stdout.txt').write_text(log)
                    np.savez_compressed(out/'official_tensors.npz',precision=ev.eval['precision'],recall=ev.eval['recall'],scores=ev.eval['scores'])
                    del ev,cg,cd,records_coco
                metric=load(out/'metrics.json')
                dump(ROOT/'analysis'/f'pooled_{mode}_{fps}_r{trial}.json',dict(score_directory=str(out),
                    population_frames=3416,source_runs=names,exact_prediction_and_ground_truth_digest=digest.hexdigest()))
                pooled.append(dict(mode=mode,rate=fps,trial=trial,frames=3416,sequences=5,
                    pacing_valid=all(completed[n]['valid'] for n in names),AP=metric['COCO']['AP'],
                    AP50=metric['COCO']['AP50'],AP75=metric['COCO']['AP75'],
                    precision=metric['fixed_conf25']['micro_precision'],recall=metric['fixed_conf25']['micro_recall'],
                    F1=metric['fixed_conf25']['micro_F1'],score_directory=str(out),
                    population='Pooled held-output queries from five independently reset complete sequences'))
                print(f'Pooled aligned scoring {mode} {fps} trial {trial}',flush=True)
                del global_arrays
    csv_out(ROOT/'analysis/pooled_aligned.csv',pooled)
    partial=[]
    for item in records:
        if item['status']!='incomplete':continue
        error=load(Path(item['path'])/'INCOMPLETE.json');buffer=error.get('buffer') or {}
        partial.append(dict(name=item['name'],sequence=item['sequence'],rate=item['fps'],trial=item['trial'],mode=item['mode'],
            error=error.get('error'),received=buffer.get('captured'),dequeued_not_verified_completed=buffer.get('consumed'),
            overwritten=buffer.get('replaced'),remaining_pending=buffer.get('pending'),peak_pending=buffer.get('peak_pending'),
            peak_rss_mib=error.get('peak_rss_mib'),forward_hooks_including_warmups=error.get('total_forwards'),
            accuracy_available=False,complete=False))
    if partial:csv_out(ROOT/'analysis/incomplete.csv',partial)
    csv_out(ROOT/'analysis/execution_matrix.csv',[{k:r[k] for k in ['dataset','sequence','trial','fps','mode','name','path','status']} for r in records])
    dump(ROOT/'analysis/results.json',dict(passes=completed,aligned=aligned,common=common,records=records,scoring_wall_s=time.perf_counter()-start,
        scoring_wall_scope='Post-run validation, query assignment, scoring and serialization; not COCO-only time'))
    dump(ROOT/'analysis/PASS_VERIFICATION.json',verification)
    print('Saved streaming results',len(table),'completed passes',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['offline','scoring','status']);a=p.parse_args()
    if a.stage=='status':print(collections.Counter(r['status'] for r in summary_rows()))
    else:globals()[a.stage]()
