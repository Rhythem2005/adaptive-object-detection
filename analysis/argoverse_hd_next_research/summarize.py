"""Read-only analysis of scored outputs; produces a fresh reproducible report."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from analysis.argoverse_hd_phase2.common import archive, dump, load, sha
from adaptive_pipeline.shared.boxes import iou_matrix
from .run import BASE, ROOT, P3, verify


def points_from_arrays(path):
    m=archive(path);out={}
    for c in [.001,.1,.25]:
        selected=m['considered']&(m['conf']>=c)
        tp=int((selected&m['tp'][:,0]).sum());fp=int((selected&~m['tp'][:,0]&~m['ignored'][:,0]).sum())
        p=tp/max(tp+fp,1);r=tp/55412
        out[str(c)]={'TP':tp,'FP':fp,'micro_precision':p,'micro_recall':r,
            'micro_F1':2*p*r/max(p+r,1e-16)}
    return out


def arm_summary(path):
    m=load(path/'scoring/metrics.json');p=load(path/'operating_points.json')
    cost=load(path/'replay_summary.json')
    with np.load(path/'scoring/official_tensors.npz') as x:
        # COCO macro precision at predeclared recall-grid points, IoU50/all area/max100.
        curve={str(r):float(x['precision'][0,int(round(r*100)),:,0,2].mean()) for r in [.1,.2,.3,.4,.5]}
    return {'path':str(path),'COCO':m['COCO'],'fixed_conf25':m['fixed_conf25'],
        'per_sequence':m['per_sequence'],'operating_points':p,'cost':cost,
        'COCO_macro_P50_at_recall_grid':curve}


def paired(reference, candidate):
    a=archive(reference/'scoring/official_matches.npz');b=archive(candidate/'scoring/official_matches.npz')
    with (candidate/'frames.csv').open() as stream:decisions=[r['decision'] for r in csv.DictReader(stream)]
    offa=np.r_[0,np.cumsum(a['n_pred'])];offb=np.r_[0,np.cumsum(b['n_pred'])]
    output={}
    for conf in [.001,.1,.25]:
        for j,threshold in [(0,50),(5,75)]:
            cells={k:Counter() for k in ['all','FULL','SKIP']}
            for i,decision in enumerate(decisions):
                def ids(m,off):
                    sl=slice(off[i],off[i+1]);valid=m['tp'][sl,j]&m['considered'][sl]&(m['conf'][sl]>=conf)
                    return set(m['matched_gt'][sl,j][valid].tolist())
                old,new=ids(a,offa),ids(b,offb)
                values={'both':len(old&new),'gained':len(new-old),'lost':len(old-new)}
                for group in ['all',decision]:cells[group].update(values)
            output[f'conf{conf}_IoU{threshold}']={k:dict(v) for k,v in cells.items()}
    return output


def source_events(path,frames):
    dense=archive(BASE/'runs/argoverse_hd_coco_phase1_v1/baseline/official_matches100.npz')
    offsets=np.r_[0,np.cumsum(dense['n_pred'])];indices={r['key']:i for i,r in enumerate(frames)}
    counts=Counter();gate_labels=Counter();sizes=[]
    source_cache=archive(BASE/'runs/argoverse_hd_benchmark_v1/baseline/preds.npz')
    with gzip.open(path/'events.jsonl.gz','rt') as stream:
        for line in stream:
            frame=json.loads(line);i=indices[frame['key']]
            for e in frame['events']:
                if e['type'] in ['spawn','confirm']:
                    kind='low_birth' if e['type']=='spawn' and e['detection'][4]<.25 else 'high_birth' if e['type']=='spawn' else 'low_confirm' if e.get('low') else 'high_confirm'
                    rows=e['source_rows']
                    if len(rows)!=1:label='ambiguous'
                    else:
                        k=offsets[i]+rows[0]
                        label='capped' if not dense['considered'][k] else 'ignored' if dense['ignored'][k,0] else 'TP50' if dense['tp'][k,0] else 'FP50'
                    counts[kind+'_'+label]+=1
                if e['type']=='weak_birth_gate':
                    previous=np.asarray(e['previous']).reshape(-1,6);sizes.append(len(previous))
                    raw=source_cache[frame['key']]
                    for candidate,accepted in zip(e['candidates'],e['accepted']):
                        rows=np.flatnonzero(np.all(raw==np.asarray(candidate,dtype=raw.dtype),axis=1))
                        if len(rows)!=1:label='ambiguous'
                        else:
                            k=offsets[i]+int(rows[0])
                            label='capped' if not dense['considered'][k] else 'ignored' if dense['ignored'][k,0] else 'TP50' if dense['tp'][k,0] else 'FP50'
                        gate_labels[('retained_' if accepted else 'rejected_')+label]+=1
    return {'source_event_counts':dict(counts),'actual_gate_source_labels':dict(gate_labels),
        'mean_previous_tentative_pool':float(np.mean(sizes)) if sizes else None,
        'max_previous_tentative_pool':max(sizes) if sizes else None,
        'note':'Source Dense TP50/FP50 update events, not unique identities or an additive AP attribution. Output-filter FULL rows still use raw source indices.'}


def weak_fp_overlap(frames):
    """Descriptive overlap candidates; never reruns matching or changes labels."""
    ground=load(ROOT/'ground_truth.json');byimage={}
    for a in ground['annotations']:
        if a.get('iscrowd') or a.get('ignore'):continue
        byimage.setdefault(a['image_id'],[]).append(a)
    boxes={}
    for image,anns in byimage.items():
        a=np.array([x['bbox'] for x in anns],float);a[:,2:]+=a[:,:2]
        boxes[image]=(a,np.array([x['category_id'] for x in anns]))
    info={f['key']:f for f in frames};cache=archive(BASE/'runs/argoverse_hd_benchmark_v1/baseline/preds.npz')
    counts=Counter();classes=Counter()
    with gzip.open(ROOT/'weak_evidence_rows.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['group']!='birth' or row['label']!='FP':continue
            f=info[row['key']];d=cache[row['key']][row['source_rows'][0]]
            gt,cls=boxes.get(f['image_id'],(np.zeros((0,4)),np.zeros(0)))
            overlap=iou_matrix(d[None,:4],gt)[0] if len(gt) else np.zeros(0)
            same=overlap[cls==d[5]]
            if len(same) and same.max()>=.5:label='same_class_overlap_ge50_duplicate_or_matching_conflict'
            elif len(overlap) and overlap.max()>=.5:label='wrong_class_overlap_ge50_candidate'
            elif len(overlap) and overlap.max()>=.1:label='coarse_overlap_10_to_50_candidate'
            else:label='background_or_very_poor_localisation_or_unannotated'
            group='retained' if row['retained'] else 'rejected'
            counts[group+'__'+label]+=1;classes[group+'__class'+str(row['class'])]+=1
    return {'counts':dict(counts),'per_class':dict(classes),
        'note':'Mutually exclusive descriptive overlaps against regular GT for officially FP50 unmatched weak source observations. These labels do not establish identity, true background, annotation error, or unique physical objects. No rescoring or relabeling.'}


def main():
    verify();assert (ROOT/'TIMING_COMPLETE.json').exists()
    if (ROOT/'ANALYSIS_COMPLETE.json').exists():raise FileExistsError('Analysis complete')
    frames=load(ROOT/'frameplan.json');arms={p.name:arm_summary(p) for p in sorted((ROOT/'arms').iterdir()) if (p/'scoring/SCORE_COMPLETE.json').exists()}
    old={n:arm_summary(P3/'arms'/n) for n in ['R_T1_anchor','R_T3_recovery','R_T4_shared10','V1_separate_birth10','V3_low_iou50',
        'R_T3_recovery__grid3','R_T3_recovery__fixed2_phase1','R_T3_recovery__fixed3_phase0','R_T3_recovery__fixed3_phase1']}
    dense_path=BASE/'runs/argoverse_hd_coco_phase1_v1/baseline'
    dm=load(dense_path/'metrics.json')
    dense={'COCO':dm['official_COCO_max100'],'fixed_conf25':dm['supplementary_fixed_conf25_IoU50'],
        'operating_points':points_from_arrays(dense_path/'official_matches100.npz'),'per_sequence':{k:v['metrics'] for k,v in dm['per_sequence'].items()},
        'detector_requests':3416,'runtime':'Historical inference not remeasured; no combining with cache costs'}
    comparisons={n:paired(ROOT/'arms/R_grid3',ROOT/'arms'/n) for n in arms if n!='R_grid3'}
    events={n:source_events(ROOT/'arms'/n,frames) for n in arms}
    gate=load(ROOT/'weak_evidence_gate.json');selection=load(ROOT/'selection.json');timing=load(ROOT/'timing_summary.json')
    selected=selection['selected'];before=timing[selected];after=timing['F_fastpath']
    change={k:100*(after[k]['mean']/before[k]['mean']-1) for k in ['step_wall_ms','step_cpu_ms','signal_ms']}
    change['round_wall_percent']=[100*(b/a-1) for a,b in zip(before['round_wall_means_ms'],after['round_wall_means_ms'])]
    # Multivariate descriptive frontier; CPU unavailable for most scored arms,
    # so do not present flow points as measured latency in this frontier.
    vector={n:[-a['COCO']['AP'],-a['COCO']['AR100'],-a['operating_points']['all']['thresholds']['0.25']['50']['TP'],
        a['operating_points']['all']['thresholds']['0.25']['50']['FP'],a['operating_points']['all']['thresholds']['0.1']['50']['FP'],
        a['cost']['simulated_FULL_requests'],a['cost']['total_flow_point_evaluations_before_FB']] for n,a in arms.items()}
    frontier=[n for n,v in vector.items() if not any(all(x<=y for x,y in zip(w,v)) and any(x<y for x,y in zip(w,v)) for m,w in vector.items() if m!=n)]
    output={'new_arms':arms,'prior_phase3':old,'Dense':dense,'paired_vs_grid3':comparisons,'source_events':events,
        'weak_FP_overlap_candidates':weak_fp_overlap(frames),
        'timing_change_percent':change,'frontier_with_flow_point_proxy_not_latency':frontier,'selection':selection,'gate':gate}
    dump(ROOT/'summary.json',output)
    with (ROOT/'summary.csv').open('x',newline='') as stream:
        fields=['method','AP_percent','AP50_percent','AP75_percent','AR100_percent','APsmall_percent','APmedium_percent','APlarge_percent',
            'TP10','FP10','P10_percent','R10_percent','F110_percent','TP25','FP25','P25_percent','R25_percent','F125_percent',
            'macro_P25_percent','macro_R25_percent','macro_F125_percent','requests','call_savings_percent','mean_tracks','flow_points']
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        entries={'Dense':dense,**{'prior_'+k:v for k,v in old.items()},**arms}
        for n,a in entries.items():
            p=a['operating_points'] if n=='Dense' else a['operating_points']['all']['thresholds']
            p10=p['0.1'] if n=='Dense' else p['0.1']['50'];p25=p['0.25'] if n=='Dense' else p['0.25']['50']
            c=a.get('cost',{});calls=c.get('simulated_FULL_requests',3416);m=a['COCO'];f=a['fixed_conf25']
            row={'method':n,**{k+'_percent':100*m[k] for k in ['AP','AP50','AP75','AR100']},
                **{'AP'+k+'_percent':100*m['AP_'+k] for k in ['small','medium','large']},
                'requests':calls,'call_savings_percent':100*(1-calls/3416),'mean_tracks':c.get('mean_post_frame_tracks'),
                'flow_points':c.get('total_flow_point_evaluations_before_FB')}
            for label,point in [('10',p10),('25',p25)]:
                row.update({'TP'+label:point['TP'],'FP'+label:point['FP'],
                    'P'+label+'_percent':100*point['micro_precision'],'R'+label+'_percent':100*point['micro_recall'],
                    'F1'+label+'_percent':100*point['micro_F1']})
            for label,key in [('P','precision'),('R','recall'),('F1','F1')]:row['macro_'+label+'25_percent']=100*f['macro_'+key]
            w.writerow(row)
    report=['# Evidence-driven optimization: measured results','',
        'This stage uses saved stock YOLOv8n detections only. All five sequences / 3,416 frames / 55,412 regular targets and 2,299 crowd annotations are retained. Official COCO evaluation is unchanged. No inference, training, D/E, Phase 4, commit or push occurred.','',
        '## Controls and diagnostic gates','',
        'All three controls reproduce predictions, details, tracker-state hashes, official precision/recall/score tensors and aggregate/class/sequence metrics exactly.','',
        '| Gate | Original TP/FP | Retained TP/FP | Original precision % | Retained precision % | TP retention % | Passed |',
        '|---|---:|---:|---:|---:|---:|---|']
    for g in ['birth','output']:
        a=gate['counts'][g+'_all'];b=gate['counts'][g+'_retained']
        report.append(f"| {g} | {a['TP']}/{a['FP']} | {b['TP']}/{b['FP']} | {100*a['precision']:.3f} | {100*b['precision']:.3f} | {100*gate['gates'][g]['TP_retention']:.3f} | {gate['gates'][g]['passed']} |")
    report+=['','The gate requires >=50% TP retention, >=5 percentage-point precision gain and >=50 source TP events. These predeclared engineering thresholds are not significance tests. Diagnostics use recovery+grid3 unmatched weak observations, with official Dense source labels; intervention changes future candidate membership. protocol_amendment.json explains why immediate-birth events would have introduced selection bias. No candidate replay preceded this correction. Stable background can persist and real objects can fail unwarped IoU confirmation. Ground-truth labels are never consumed by a runtime candidate.','',
        'Skipped arms: '+json.dumps(load(ROOT/'variants_skipped.json')),'',
        '## Official metrics and operating points','',
        'AP/AR and precision cells below are percentages. P25 is micro precision at exact confidence 0.25 and IoU50. FP10 is all-frame FP at exact confidence 0.10. These are different estimands from COCO AP.','',
        '| Method | AP | AP50 | AP75 | AR100 | TP25 | FP25 | P25 | TP10 | FP10 | Requests | Mean tracks | Flow points |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    p=dense['operating_points'];m=dense['COCO']
    report.append(f"| Dense existing | {100*m['AP']:.6f} | {100*m['AP50']:.6f} | {100*m['AP75']:.6f} | {100*m['AR100']:.6f} | {p['0.25']['TP']} | {p['0.25']['FP']} | {100*p['0.25']['micro_precision']:.4f} | {p['0.1']['TP']} | {p['0.1']['FP']} | 3416 | — | — |")
    for n,a in arms.items():
        m=a['COCO'];c=a['cost'];p=a['operating_points']['all']['thresholds'];p25=p['0.25']['50'];p10=p['0.1']['50']
        report.append(f"| {n} | {100*m['AP']:.6f} | {100*m['AP50']:.6f} | {100*m['AP75']:.6f} | {100*m['AR100']:.6f} | {p25['TP']} | {p25['FP']} | {100*p25['micro_precision']:.4f} | {p10['TP']} | {p10['FP']} | {c['simulated_FULL_requests']} | {c['mean_post_frame_tracks']:.5f} | {c['total_flow_point_evaluations_before_FB']} |")
    report+=['','Full macro/micro P/R/F1 at exact thresholds, size/class AP, all prior Phase 3 controls, and source-event/paired-match diagnostics are in summary.json and summary.csv. AP is pooled, not an average of sequence scores.','',
        '## Per-sequence COCO AP (%)','',
        '| Method | sequence_00 | sequence_01 | sequence_02 | sequence_03 | sequence_04 |','|---|---:|---:|---:|---:|---:|']
    for n,a in {'Dense':dense,**arms}.items():report.append('| '+n+' | '+' | '.join(f"{100*a['per_sequence'][f'sequence_{i:02d}']['AP']:.5f}" for i in range(5))+' |')
    report+=['','Sequence class support differs; sequence AP must not be averaged to recover pooled AP. The five sequences have already influenced development. No untouched test set or generalization claim is implied.','',
        '## Cached processing cost and parity','',
        f"Selected timed reference: {selected}. Fastpath changes only unused thumbnail/LAB/illumination preparation; required 640-wide flow preparation still executes on FULL and SKIP. The raw model input, detector schedule, output arrays, details, tracker arrays and official tensors are identical. Inactive luminance/controller-thumbnail/illumination telemetry is intentionally absent or no longer updated; no enabled decision or enhancement consumes it.",'',
        '| Arm | Mean wall ms | Median | P95 | P99 | Mean CPU ms | Mean signal ms | Required flow preparation ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n in [selected,'F_fastpath']:
        t=timing[n]
        report.append(f"| {n} | {t['step_wall_ms']['mean']:.6f} | {t['step_wall_ms']['median']:.6f} | {t['step_wall_ms']['p95']:.6f} | {t['step_wall_ms']['p99']:.6f} | {t['step_cpu_ms']['mean']:.6f} | {t['signal_ms']['mean']:.6f} | {t['required_flow_prepare_ms']['mean']:.6f} |")
    report+=['',f"Wall change: {change['step_wall_ms']:.3f}%; CPU change: {change['step_cpu_ms']:.3f}%. Paired round wall changes: {change['round_wall_percent']}. Three AB/BA/AB repetitions include identical lightweight component timer wrappers; their overhead is included. No heavy event observers execute. Decode/hash/output validation/serialization and model execution are outside the timed step. CPU time includes worker threads. These are not end-to-end FPS, energy measurements or streaming AP.",'',
        '## Protocol correction and development failures','', 'Before the first gate or new variant, the candidate cohort was corrected to avoid conditioning on immediate weak births. Only the unused gate function in the driver changed; controls and all algorithm source bytes remained identical, and original source/manifest versions are retained. Two diagnostic helper-import errors were corrected before a gate result was written; they caused no pipeline replay or scoring. The initial pytest command was unavailable; the existing unittest runner was used without installing dependencies. See protocol_amendment.json, execution_source_revision.json and development_failures.json.','', '## Reproduction and boundaries','',
        'Commands (repository root; every completed stage refuses overwrite):','',
        '```text']
    for stage in ['prepare','controls','gate','variants','finalists','timing']:
        report.append('PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.run --stage '+stage)
    report+=['PYTHONPATH=extracted:. .venv/bin/python -m analysis.argoverse_hd_next_research.summarize','```','',
        'Exact configs, conditional gates, forced schedules, source snapshots/hashes and pre-existing dirty files are archived here. plan.json is the predeclared scope; finalist_specs.json contains the selected full configurations. Historical results and dataset files were not overwritten. The fastpath defaults off; causal birth/output adapters remain isolated in analysis, rather than silently replacing production algorithms.','',
        'This report is descriptive. Summary JSON retains paired gained/lost official GT matches at multiple confidence/IoU thresholds. Those are not additive AP contributions. Flow-point counts are work proxies, not latency measurements. Candidate selection uses the declared TP/FP gate followed by AP and point work; the measured frontier has no repeated CPU price for every arm. D/E remain deferred.']
    (ROOT/'report.md').write_text('\n'.join(report)+'\n')
    dump(ROOT/'ANALYSIS_COMPLETE.json',{'arms':list(arms),'new_YOLO_inferences':0})
    print('Analysis complete:',ROOT/'report.md')


if __name__=='__main__':main()
