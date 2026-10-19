"""Summarize approved replay outputs; no new pipeline execution or inference."""
from collections import Counter, defaultdict
import argparse
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
from adaptive_pipeline.evaluation.coco import prediction_records, make_coco, run_coco
from .common import archive, dump, load
from .paired import annotation_index, overlap, size_name
from .scoring import match_arrays, frame_counts


def quantiles(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return {'n':len(a),'median':float(np.median(a)) if len(a) else None,
            'p90':float(np.percentile(a,90)) if len(a) else None,
            'mean':float(a.mean()) if len(a) else None}


def catalog(out):
    result={}
    for stage in ['R_reference_parity','B_tracking_factors','C_schedule_factors']:
        for p in sorted((out/stage).iterdir()):
            if p.is_dir() and (p/'scoring/SCORE_COMPLETE.json').exists():
                result[p.name]={'path':p,'stage':stage,
                    'metrics':load(p/'scoring/metrics.json'), 'cost':load(p/'replay_summary.json')}
    return result


def operating_points(gtdata, plan, directory):
    byimage,_=annotation_index(gtdata)
    with (directory/'frames.csv').open() as f:rows=list(csv.DictReader(f))
    matches=archive(directory/'scoring/official_matches.npz')
    groups={'all':np.ones(len(rows),bool)}
    for d in ['FULL','SKIP']:groups[d]=np.array([r['decision']==d for r in rows])
    for age in sorted({int(r['since_full']) for r in rows if r['decision']=='SKIP'}):
        groups[f'SKIP_age{age}']=np.array([r['decision']=='SKIP' and int(r['since_full'])==age for r in rows])
    result={}
    for label,mask in groups.items():
        n=sum(len(byimage[p['image_id']]) for p,m in zip(plan,mask) if m)
        result[label]={'frames':int(mask.sum()),'regular_targets':n,'thresholds':{}}
        for conf in ['0.001','0.1','0.25']:
            counts=matches['counts_conf'+conf][mask].sum(0)
            result[label]['thresholds'][conf]={}
            for j,iou in [(0,50),(1,75)]:
                tp,fp=map(int,counts[j]);precision=tp/max(tp+fp,1);recall=tp/max(n,1)
                result[label]['thresholds'][conf][str(iou)]={'TP':tp,'FP':fp,'micro_precision':precision,
                    'micro_recall':recall,'micro_F1':2*precision*recall/max(precision+recall,1e-16)}
    return result


def runtime_geometry(gtdata, plan, dense_matches, directory):
    byimage,identity=annotation_index(gtdata);byid={a['id']:a for aa in byimage.values() for a in aa}
    bykey={r['key']:r for r in plan}
    offsets=np.r_[0,np.cumsum(dense_matches['n_pred'])];index={r['key']:i for i,r in enumerate(plan)}
    states=archive(directory/'track_states.npz');preds=archive(directory/'preds.npz');details=archive(directory/'prediction_details.npz')
    cfg=load(directory/'config.json');events=Counter();links={};previous_video=None;rows=[];visibility=Counter();flowstats=defaultdict(list)
    with gzip.open(directory/'events.jsonl.gz','rt') as stream:
        for line in stream:
            frame=json.loads(line);key=frame['key'];info=bykey[key];i=index[key]
            if previous_video!=info['video']:links={};previous_video=info['video']
            for e in frame['events']:
                if e['type'] in ['spawn','confirm']:
                    events[e['type']]+=1
                    if e['type']=='confirm' and e['low']:events['low_score_confirmations']+=1
                    if e['type']=='confirm' and e['previous_missed']>0:events['hidden_track_recoveries']+=1
                    if e['type']=='spawn' and e['detection'][4]<.25:events['births_below_25']+=1
                    candidates=[r for r in e['source_rows'] if dense_matches['tp'][offsets[i]+r,0]]
                    if len(candidates)==1:
                        row=offsets[i]+candidates[0];a=byid[int(dense_matches['matched_gt'][row,0])]
                        links[e['track_id']]={'identity':(a['category_id'],a['track']),
                            'seed_conf':e['detection'][4], 'seed_TP75':bool(dense_matches['tp'][row,5])}
                    else:
                        links.pop(e['track_id'],None)
                        events['confirm_or_birth_without_unique_correct_Dense_TP50']+=1
                if e['type']=='post_FULL':
                    events['post_FULL_hidden_track_checks']+=len(e['hidden_ids'])
                    events['deleted_tracks']+=len(e['deleted_ids'])
                    for tid in e['deleted_ids']:links.pop(tid,None)
                if e['type']=='flow':
                    for k,v in e['diagnostics'].items():
                        flowstats[k].extend(x if x is not None else np.nan for x in v)
                    events['flow_track_updates']+=len(e['ok'])
                    events['flow_rejected_updates']+=sum(not x for x in e['ok'])
            if frame['decision']!='SKIP':continue
            emitted={int(t[0]):n for n,t in enumerate(details[key])}
            for t in states[key]:
                tid=int(t[0]);visibility['SKIP_internal_track_frames']+=1
                if tid not in links:visibility['no_unique_correct_seed_identity']+=1;continue
                a=identity.get((info['image_id'],*links[tid]['identity']))
                if a is None:visibility['linked_identity_not_annotated_current_frame']+=1;continue
                n=emitted.get(tid)
                reason='emitted' if n is not None else 'missed_FULL' if t[4]>0 else 'flow_coast_exceeded' if t[5]>cfg['coast_max'] else 'output_NMS_suppressed'
                visibility[reason]+=1
                x,y,w,h=a['bbox'];g=np.array([x,y,x+w,y+h]);box=preds[key][n,:4] if n is not None else t[6:10]
                bw,bh=np.maximum(box[2:]-box[:2],1e-9)
                centre=float(np.linalg.norm((box[:2]+box[2:])/2-(g[:2]+g[2:])/2)/max(np.hypot(w,h),1e-9))
                rows.append([info['image_id'],tid,a['id'],a['category_id'],a['area'],links[tid]['seed_conf'],
                    int(links[tid]['seed_TP75']),t[3],int(n is not None),float(overlap([g],[box])[0,0]),centre,
                    abs(float(np.log(bw/w))),abs(float(np.log(bh/h))),t[10],t[11]])
    arr=np.asarray(rows,float).reshape(-1,15)
    summary={'event_counts':dict(events),'SKIP_visibility':dict(visibility),
        'flow_diagnostic_all_internal_updates':{k:quantiles(v) for k,v in flowstats.items()},
        'geometry_note':'Linked only from unique official Dense TP50 source rows at actual confirmations/births; GT track IDs may fragment. Includes hidden states separately from emitted predictions.',
        'emitted_geometry':{},'by_age':{},'by_size':{},'quality_bins':{}}
    if len(arr):
        mask=arr[:,8]>0
        def describe(m):
            return {'eligible_track_frames':int(m.sum()),'IoU50_rate':float((arr[m,9]>=.5).mean()) if m.any() else None,
                'IoU75_rate':float((arr[m,9]>=.75).mean()) if m.any() else None,
                'centre_error_GT_diagonal':quantiles(arr[m,10]),'absolute_log_width_ratio':quantiles(arr[m,11]),
                'absolute_log_height_ratio':quantiles(arr[m,12])}
        summary['emitted_geometry']=describe(mask)
        summary['all_linked_internal_geometry']=describe(np.ones(len(arr),bool))
        for age in sorted(set(arr[:,7])):summary['by_age'][str(int(age))]=describe(mask&(arr[:,7]==age))
        for name,m in [('small',arr[:,4]<1024),('medium',(arr[:,4]>=1024)&(arr[:,4]<9216)),('large',arr[:,4]>=9216)]:summary['by_size'][name]=describe(mask&m)
        for lo,hi in [(0,.5),(.5,.95),(.95,1.01)]:summary['quality_bins'][f'{lo}:{hi}']=describe(mask&(arr[:,13]>=lo)&(arr[:,13]<hi))
    return summary,arr


def strong_seed_cohort(gtdata, plan, dense_matches, directory):
    byimage,identity=annotation_index(gtdata);matches=archive(directory/'scoring/official_matches.npz')
    with (directory/'frames.csv').open() as f:fr=list(csv.DictReader(f))
    last={};counts=Counter();ids=[]
    for info,row in zip(plan,fr):
        seq=info['video']
        if row['decision']=='FULL':last[seq]=info['image_id'];continue
        for a in byimage[info['image_id']]:
            prev=identity.get((last[seq],a['category_id'],a['track']))
            if prev and dense_matches['gt_scores'][prev['id'],5]>=.25 and dense_matches['gt_scores'][a['id'],5]>=.25:
                counts['GT_instances']+=1;ids.append(a['id'])
                for j in [0,5]:counts[f'adaptive_TP_IoU{50+5*j}']+=int(matches['gt_scores'][a['id'],j]>=.001)
    return dict(counts),ids


def schedule_utility(gtdata, plan, dense_matches, directory, destination):
    shadows=archive(directory/'pre_full_shadow.npz');preds=archive(directory/'preds.npz')
    with (directory/'frames.csv').open() as f:fr=list(csv.DictReader(f))
    subset=[p for p in plan if p['key'] in shadows]
    records,_=prediction_records({p['image_id']:shadows[p['key']] for p in subset},{i:i for i in range(8)})
    gt,dt=make_coco(gtdata,records);ev,log=run_coco(gt,dt,[p['image_id'] for p in subset])
    sm=match_arrays(ev,records);sc=frame_counts(sm,subset,shadows);del ev,gt,dt
    shadowindex={p['key']:i for i,p in enumerate(subset)}
    main=archive(directory/'scoring/official_matches.npz');rows=[]
    # FULL-frame post-update track counts contain detector information. Use the
    # captured pre-decision state for all signal/benefit correlations instead.
    pre_signals={}
    with gzip.open(directory/'events.jsonl.gz','rt') as stream:
        for line in stream:
            frame=json.loads(line)
            event=next((e for e in frame['events'] if e['type']=='pre_decision'),None)
            if event is None:continue
            tracks=np.asarray(event['tracks'],float).reshape(-1,15)
            visible=tracks[:,4]==0
            pre_signals[frame['key']]={'n_tracks':len(tracks),
                'flow_quality_mean':float(tracks[visible,10].mean()) if visible.any() else np.nan,
                'uncertainty_mean':float(tracks[visible,12].mean()) if visible.any() else np.nan}
    for i,(info,row) in enumerate(zip(plan,fr)):
        key=info['key']
        if row['decision']=='FULL' and key not in shadows:continue
        rec={'key':key,'decision':row['decision'],'since_full':int(row['since_full']),
             'novelty':float(row['novelty']),'fail_frac':float(row['fail_frac']),
             **pre_signals[key]}
        for conf in ['0.001','0.25']:
            prop=sc[conf][shadowindex[key]] if row['decision']=='FULL' else main['counts_conf'+conf][i]
            delta=dense_matches['counts_conf'+conf][i]-prop
            rec.update({f'Dense_minus_propagation_TP50_conf{conf}':int(delta[0,0]),
                f'Dense_minus_propagation_FP50_conf{conf}':int(delta[0,1]),
                f'Dense_minus_propagation_TP75_conf{conf}':int(delta[1,0]),
                f'Dense_minus_propagation_FP75_conf{conf}':int(delta[1,1])})
        rows.append(rec)
    destination.mkdir(parents=True,exist_ok=False)
    with (destination/'per_frame.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'eligible_frames':len(rows),'note':'Same-history immediate Dense minus propagated counts; excludes initial FULL; not downstream causal benefit, frame AP, or additive AP decomposition. Both false positives and true positives matter.', 'by_decision':{},'signal_correlations':{}}
    for decision in ['FULL','SKIP']:
        rr=[r for r in rows if r['decision']==decision]
        summary['by_decision'][decision]={'frames':len(rr),**{k:quantiles([r[k] for r in rr]) for k in rows[0] if k.startswith('Dense_minus')}}
    for signal in ['since_full','novelty','fail_frac','flow_quality_mean','uncertainty_mean','n_tracks']:
        x=np.array([r[signal] for r in rows]);y=np.array([r['Dense_minus_propagation_TP50_conf0.25'] for r in rows]);m=np.isfinite(x)&np.isfinite(y)
        rho=float(spearmanr(x[m],y[m]).statistic) if m.sum()>2 and len(np.unique(x[m]))>1 else None
        summary['signal_correlations'][signal]={'eligible_frames':int(m.sum()),'Spearman_rho_with_TP50_conf25_benefit':rho}
    dump(destination/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out).resolve()
    for stage in ['R','A','B','C']:
        if not (out/(stage+'_COMPLETE.json')).exists():raise RuntimeError('All approved stages must finish first')
    dest=out/'combined_diagnostics';dest.mkdir(exist_ok=False)
    plan=load(out/'frameplan.json')
    import gzip
    with gzip.open(Path(load(out/'execution_plan.json')['approved_plan']['verified_scoring'])/'annotations.coco.json.gz','rt') as f:gtdata=json.load(f)
    dense=archive(out/'A_saved_pairing/Dense_scoring/official_matches.npz');arms=catalog(out)
    table=[];cohorts={};cohort_ids=None;all_diagnostics={};points={}
    for name,arm in arms.items():
        print('Analysing runtime events:',name,flush=True)
        diag,geom=runtime_geometry(gtdata,plan,dense,arm['path'])
        np.savez_compressed(dest/(name+'_geometry.npz'),rows=geom,
            columns=['image_id','track_id','GT_id','class','GT_area','seed_conf','seed_TP75','age','emitted','IoU','centre_error','log_width_error','log_height_error','flow_quality','flow_innovation'])
        all_diagnostics[name]=diag;dump(dest/(name+'_runtime.json'),diag)
        points[name]=operating_points(gtdata,plan,arm['path'])
        metrics=arm['metrics']['COCO'];cost=arm['cost']
        table.append({'arm':name,'stage':arm['stage'],'AP':metrics['AP'],'AP50':metrics['AP50'],'AP75':metrics['AP75'],
            'AR100':metrics['AR100'],'AP_small':metrics['AP_small'],'AP_medium':metrics['AP_medium'],'AP_large':metrics['AP_large'],
            'simulated_FULL_requests':cost['simulated_FULL_requests'],'call_savings':1-cost['simulated_FULL_requests']/len(plan),
            'mean_tracks':cost['mean_post_frame_tracks'],'flow_points':cost['total_flow_point_evaluations_before_FB'],
            'corrected_replay_step_wall_ms':cost['mean_corrected_step_wall_ms'],'corrected_replay_step_cpu_ms':cost['mean_corrected_step_cpu_ms'],
            'diagnostic_wall_ms':cost['mean_diagnostic_wall_ms'],'decode_ms':cost['mean_decode_ms'],
            'replay_stage_wall_seconds':cost['stage_wall_seconds_including_decode_hash_diagnostics_IO'],
            'scoring_wall_seconds':arm['metrics']['scoring_wall_seconds']})
        if arm['stage']=='B_tracking_factors' or name=='fixed2_093':
            cohort,ids=strong_seed_cohort(gtdata,plan,dense,arm['path']);cohorts[name]=cohort
            if cohort_ids is None:cohort_ids=ids
            if ids!=cohort_ids:raise AssertionError('Fixed schedule common strong-seed cohort differs')
    utilities={}
    for name,arm in arms.items():
        if arm['stage']=='C_schedule_factors':
            print('Analysing same-history refresh benefit:',name,flush=True)
            utilities[name]=schedule_utility(gtdata,plan,dense,arm['path'],dest/(name+'_utility'))
    with (dest/'summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    dump(dest/'fixed_schedule_strong_seed_cohort.json',{'definition':'Previous FULL and current Dense are official TP75 at confidence>=.25, using GT track identity; same selected GT set in every fixed-schedule arm.','results':cohorts})
    dump(dest/'operating_points_by_decision_age.json',points)
    dump(dest/'summary.json',{'arms':table,'strong_seed_cohort':cohorts,'schedule_utilities':utilities,
        'processed_frame_steps':sum(a['cost']['frames'] for a in arms.values()),
        'cache_FULL_lookups':sum(a['cost']['simulated_FULL_requests'] for a in arms.values()),
        'new_YOLO_inferences':0})
    dump(out/'ANALYSIS_COMPLETE.json',{'arms':len(arms),'new_YOLO_inferences':0})


if __name__=='__main__':main()
