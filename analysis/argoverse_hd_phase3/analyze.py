"""Post-score paired diagnostics; no detector, pipeline replay or metric changes."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from analysis.argoverse_hd_phase2.common import archive, dump, load
from analysis.argoverse_hd_phase2.analyze import runtime_geometry, strong_seed_cohort
from .run import ROOT, P2, conservative_gate, result, verify_sources


def event_summary(directory, frames, dense):
    offsets=np.r_[0,np.cumsum(dense['n_pred'])]
    indices={f['key']:i for i,f in enumerate(frames)}
    counts=Counter();association=defaultdict(Counter)
    with gzip.open(directory/'events.jsonl.gz','rt') as stream:
        for line in stream:
            frame=json.loads(line);i=indices[frame['key']]
            for e in frame['events']:
                if e['type']=='association_comparison':
                    c=association[e['stage']];c['matrices']+=1
                    c['different_pairs']+=int(e['different_pairs'])
                    c['gated_more_valid_pairs_matrices']+=int(e['gated_matches']>e['legacy_matches'])
                    c['gated_extra_valid_pairs_total']+=e['gated_matches']-e['legacy_matches']
                    c['legacy_matches_total']+=e['legacy_matches'];c['gated_matches_total']+=e['gated_matches']
                    c['gated_lower_summed_IoU_matrices']+=int(e['gated_iou_sum']+1e-8<e['legacy_iou_sum'])
                if e['type'] not in ['spawn','confirm']:continue
                label='birth_low' if e['type']=='spawn' and e['detection'][4]<.25 else 'birth_high' if e['type']=='spawn' else 'confirm_low' if e['low'] else 'confirm_high'
                ids=e['source_rows']
                if len(ids)!=1:quality='ambiguous_duplicate_source_rows'
                else:
                    n=offsets[i]+ids[0]
                    quality='official_TP50' if dense['tp'][n,0] else 'official_ignored50' if dense['ignored'][n,0] else 'beyond_cap' if not dense['considered'][n] else 'official_FP50'
                counts[label+'__'+quality]+=1
                if e['type']=='confirm' and e['low'] and e['previous_missed']>0:
                    counts['hidden_low_confirm__'+quality]+=1
    return {'source_observation_events':dict(counts),'same_input_assignment_comparisons':{k:dict(v) for k,v in association.items()},
        'note':'Event counts, not unique objects. Official source TP50 establishes current detector correctness, not physical track-identity continuity. Assignment alternatives share each logged matrix/history; they are not whole-run causal AP estimates.'}


def paired_targets(gt, frames, candidate, reference):
    a=archive(candidate/'scoring/official_matches.npz')['gt_scores']
    b=archive(reference/'scoring/official_matches.npz')['gt_scores']
    with (candidate/'frames.csv').open() as f:arows=list(csv.DictReader(f))
    with (reference/'frames.csv').open() as f:brows=list(csv.DictReader(f))
    image_groups={r['image_id']:(ar['decision'],br['decision']) for r,ar,br in zip(frames,arows,brows)}
    groups=defaultdict(list)
    for ann in gt['annotations']:
        if ann['iscrowd']:continue
        groups['all'].append(ann['id'])
        ad,bd=image_groups[ann['image_id']]
        groups[f'{ad}_candidate__{bd}_reference'].append(ann['id'])
    answer={}
    for label,ids in groups.items():
        ids=np.asarray(ids,int);answer[label]={'regular_GT_instances':len(ids),'matches':{}}
        for j,iou in [(0,50),(5,75)]:
            for conf in [.001,.1,.25]:
                am=a[ids,j]>=conf;bm=b[ids,j]>=conf
                answer[label]['matches'][f'IoU{iou}_conf{conf}']={
                    'both':int((am&bm).sum()),'neither':int((~am&~bm).sum()),
                    'candidate_only':int((am&~bm).sum()),'reference_only':int((~am&bm).sum())}
    return answer


def main():
    verify_sources()
    if not (ROOT/'TIMING_COMPLETE.json').exists():raise RuntimeError('Approved scoring/timing incomplete')
    destination=ROOT/'diagnostics';destination.mkdir(exist_ok=False)
    gt=load(ROOT/'ground_truth.json');frames=load(ROOT/'frameplan.json')
    dense=archive(P2/'A_saved_pairing/Dense_scoring/official_matches.npz')
    base=ROOT/'arms/R_T3_recovery';rows=[];details={}
    for directory in sorted((ROOT/'arms').iterdir()):
        if not (directory/'scoring/SCORE_COMPLETE.json').exists():raise RuntimeError(f'Incomplete arm:{directory}')
        name=directory.name;r=result(name);m=r['metrics'];c=r['cost'];p=r['points']['all']['thresholds'];sp=r['points']['SKIP']['thresholds']
        row={'arm':name,**{k:m['COCO'][k] for k in ['AP','AP50','AP75','AR100','AP_small','AP_medium','AP_large']},
            'macro_P25':m['fixed_conf25']['macro_precision'],'macro_R25':m['fixed_conf25']['macro_recall'],
            'macro_F1_25':m['fixed_conf25']['macro_F1'],'simulated_FULL_requests':c['simulated_FULL_requests'],
            'call_savings':1-c['simulated_FULL_requests']/len(frames),'mean_tracks':c['mean_post_frame_tracks'],
            'flow_points':c['total_flow_point_evaluations_before_FB'],
            'corrected_replay_wall_ms':c['mean_corrected_step_wall_ms'],
            'corrected_replay_cpu_ms':c['mean_corrected_step_cpu_ms']}
        for conf in ['0.001','0.1','0.25']:
            for k in ['TP','FP','micro_precision','micro_recall','micro_F1']:
                row[f'{k}_conf{conf}']=p[conf]['50'][k]
            row[f'SKIP_TP_conf{conf}']=sp[conf]['50']['TP'];row[f'SKIP_FP_conf{conf}']=sp[conf]['50']['FP']
        rows.append(row)
        runtime,geometry=runtime_geometry(gt,frames,dense,directory)
        dump(destination/(name+'_runtime.json'),runtime)
        np.savez_compressed(destination/(name+'_linked_geometry.npz'),rows=geometry)
        cohort,ids=strong_seed_cohort(gt,frames,dense,directory)
        details[name]={'paired_vs_T3':paired_targets(gt,frames,directory,base),
            'event_summary':event_summary(directory,frames,dense),'strong_seed_cohort':cohort,
            'strong_seed_GT_ids':ids,'per_sequence_AP':{seq:value['AP'] for seq,value in m['per_sequence'].items()},
            'conservative_T3_screen':conservative_gate(r,result('R_T3_recovery'))}
        print('Analysed:',name,flush=True)
    with (destination/'summary.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    a=archive(ROOT/'arms/V1_separate_birth10/preds.npz')
    b=archive(ROOT/'arms/R_T4_shared10/preds.npz')
    unequal=[k for k in a if not np.array_equal(a[k],b[k])]
    equivalence={'V1_vs_T4_prediction_arrays_differing_frames':unequal,
        'V1_vs_T4_official_aggregate_metrics_exact':result('V1_separate_birth10')['metrics']['COCO']==result('R_T4_shared10')['metrics']['COCO'],
        'note':'Equal aggregate metrics do not imply bitwise equal predictions or universal equivalence of association policies.'}
    dump(destination/'summary.json',{'arms':rows,'details':details,'equivalence_checks':equivalence,
        'scored_replays':len(rows),'timing_replays':6,'total_frame_steps':(len(rows)+6)*len(frames),
        'new_YOLO_inferences':0,'D_E_Phase4_executed':False})
    dump(ROOT/'ANALYSIS_COMPLETE.json',{'scored_replays':len(rows),'new_YOLO_inferences':0})
    print('ANALYSIS COMPLETE',flush=True)


if __name__=='__main__':main()
