"""A: pair official GT matches, keeping overlap-only explanations separate."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from .common import archive, dump, load
from .scoring import score


def overlap(g, p):
    g,p=np.asarray(g,float).reshape(-1,4),np.asarray(p,float).reshape(-1,4)
    lo=np.maximum(g[:,None,:2],p[None,:,:2]);hi=np.minimum(g[:,None,2:],p[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(2)
    return inter/np.maximum(np.maximum(g[:,2:]-g[:,:2],0).prod(1)[:,None]+np.maximum(p[:,2:]-p[:,:2],0).prod(1)[None,:]-inter,1e-12)


def annotation_index(gtdata):
    byimage=defaultdict(list);identity={}
    for a in gtdata['annotations']:
        if not a.get('iscrowd'):
            byimage[a['image_id']].append(a)
            key=(a['image_id'],a['category_id'],a['track'])
            if key in identity:
                raise ValueError(f'Duplicate regular annotation identity: {key}')
            identity[key]=a
    return byimage,identity


def size_name(area):
    # Descriptive bins are disjoint. Official AP area boundaries remain unchanged.
    return 'small' if area<1024 else 'medium' if area<9216 else 'large'


def saved_pairing(gtdata, plan, dense, benchmark, phase1, out):
    target=out/'A_saved_pairing';target.mkdir(exist_ok=False)
    score(dense,gtdata,plan,target/'Dense_scoring',expected=load(phase1/'baseline/metrics.json')['official_COCO_max100'],per_sequence=False)
    dm=archive(target/'Dense_scoring/official_matches.npz')
    byimage,identity=annotation_index(gtdata)
    offsets=np.r_[0,np.cumsum(dm['n_pred'])];framebykey={r['key']:r for r in plan}
    regular=[a for a in gtdata['annotations'] if not a.get('iscrowd')]
    methods=['fixed2_093','C_fullfallback','adaptive_observed','adaptive_quality']
    results={};candidates=[]
    for method in methods:
        directory=out/'R_reference_parity'/method
        am=archive(directory/'scoring/official_matches.npz');pred=archive(directory/'preds.npz')
        with (directory/'frames.csv').open() as f:rows=list(csv.DictReader(f))
        masks={r['image_id']:row['decision']=='SKIP' for r,row in zip(plan,rows)}
        part=[a for a in regular if masks[a['image_id']]]
        result={'SKIP_frames':sum(masks.values()),'regular_targets_on_SKIP':len(part),'paired_gt_matches':{},'class_size_confidence_cohorts':{},'previous_FULL_candidate_evidence':{}}
        for conf in [.001,.1,.25]:
            for j in [0,5]:
                name=f'IoU{50+5*j}_conf{conf}'
                counts=Counter();cohorts=defaultdict(Counter)
                for a in part:
                    db=dm['gt_scores'][a['id'],j]>=conf;ab=am['gt_scores'][a['id'],j]>=conf
                    state='both' if db and ab else 'Dense_only' if db else 'adaptive_only' if ab else 'neither'
                    counts[state]+=1
                    cohorts[f"class{a['category_id']}/{size_name(a['area'])}"][state]+=1
                result['paired_gt_matches'][name]=dict(counts)
                result['class_size_confidence_cohorts'][name]={k:dict(v) for k,v in cohorts.items()}
        # Join detector-row observations to internal track IDs on the previous FULL.
        admitted={};last_full={};reasons=Counter();seed_runtime=Counter()
        states=archive(directory/'track_states.npz')
        events={}
        with gzip.open(directory/'events.jsonl.gz','rt') as f:
            for line in f:
                row=json.loads(line);events[row['key']]=row['events']
        for i,(frame,row) in enumerate(zip(plan,rows)):
            seq=frame['video'];anns=byimage[frame['image_id']]
            if row['decision']=='FULL':
                known={}
                local_matches=dm['matched_gt'][offsets[i]:offsets[i+1],0]
                local_tp=dm['tp'][offsets[i]:offsets[i+1],0]
                linked={}
                for event in events[frame['key']]:
                    if event['type'] in ['spawn','confirm']:
                        for source_row in event['source_rows']:
                            linked[source_row]=event['track_id']
                for a in anns:
                    candidates_rows=np.flatnonzero(local_tp & (local_matches==a['id']))
                    source_row=int(candidates_rows[0]) if len(candidates_rows) else None
                    known[(a['category_id'],a['track'])]={'score':float(dm['gt_scores'][a['id'],0]),
                        'runtime_id':linked.get(source_row),'annotation_id':a['id']}
                last_full[seq]=known
                continue
            current_ids={int(t[0]):t for t in states[frame['key']]}
            for a in anns:
                if dm['gt_scores'][a['id'],0]<.001 or am['gt_scores'][a['id'],0]>=.001:
                    continue
                prev=last_full.get(seq,{}).get((a['category_id'],a['track']))
                if prev is None:reason='annotation_identity_absent_previous_FULL'
                elif prev['score']<.001:reason='no_correct_previous_Dense_match'
                elif prev['score']<.1:reason='previous_correct_score_001_to_10'
                elif prev['score']<.25:reason='previous_correct_score_10_to_25'
                else:reason='previous_correct_score_at_least_25'
                reasons[reason]+=1
                runtime='no_correct_previous_detector_row_linked'
                if prev and prev['runtime_id'] is not None:
                    tr=current_ids.get(prev['runtime_id'])
                    runtime='linked_track_deleted' if tr is None else 'linked_track_missed_FULL' if tr[4]>0 else 'linked_track_exceeds_flow_coast' if tr[5]>1e-9+load(directory/'config.json')['coast_max'] else 'linked_track_present_but_no_correct_official_output'
                seed_runtime[runtime]+=1
                if method=='adaptive_quality':
                    casecat='low_score_candidate_loss' if reason.startswith('previous_correct_score_0') or reason=='previous_correct_score_10_to_25' else 'strong_seed_propagation_loss' if reason=='previous_correct_score_at_least_25' else None
                    if casecat:candidates.append({'category':casecat,'sequence':seq,'image_id':frame['image_id'],'annotation_id':a['id'],'GT_bbox':a['bbox'],'class':a['category_id'],'history_reason':reason,'runtime_status':runtime})
        result['previous_FULL_candidate_evidence']=dict(reasons)
        result['runtime_link_status_of_Dense_only_SKIP_targets']=dict(seed_runtime)
        results[method]=result
        dump(target/(method+'_paired.json'),result)
    # FP candidates use official ignore/TP flags; overlap labels are explanatory only.
    fp_results={}
    for method in ['baseline','adaptive_quality']:
        matches=dm if method=='baseline' else archive(out/'R_reference_parity'/method/'scoring/official_matches.npz')
        preds=dense if method=='baseline' else archive(out/'R_reference_parity'/method/'preds.npz')
        off=np.r_[0,np.cumsum(matches['n_pred'])];counts=Counter()
        for i,frame in enumerate(plan):
            aa=byimage[frame['image_id']];p=preds[frame['key']]
            boxes=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in aa]).reshape(-1,4)
            ov=overlap(boxes,p[:,:4]);classes=np.array([a['category_id'] for a in aa])
            same=ov*(classes[:,None]==p[None,:,5]);sl=slice(off[i],off[i+1])
            fp=matches['considered'][sl]&~matches['tp'][sl,0]&~matches['ignored'][sl,0]
            for k in np.flatnonzero(fp):
                best=float(same[:,k].max(initial=0));anybest=float(ov[:,k].max(initial=0))
                label='duplicate_or_matching_conflict' if best>=.5 else 'wrong_class_overlap' if anybest>=.5 else 'coarse_localisation' if best>=.1 else 'background_stale_or_very_poor_localisation'
                counts[label]+=1
                if p[k,4]>=.25:counts[label+'_conf25']+=1
                if method=='adaptive_quality' and label in ['wrong_class_overlap','duplicate_or_matching_conflict','background_stale_or_very_poor_localisation']:
                    candidates.append({'category':'wrong_class_overlap' if label=='wrong_class_overlap' else 'duplicate_stale_background',
                        'sequence':frame['video'],'image_id':frame['image_id'],'prediction_row':int(k),
                        'prediction':p[k].tolist(),'candidate_label':label})
            if method=='baseline':
                for a in aa:
                    if dm['gt_scores'][a['id'],0]<.001:
                        candidates.append({'category':'Dense_miss','sequence':frame['video'],'image_id':frame['image_id'],
                            'annotation_id':a['id'],'GT_bbox':a['bbox'],'class':a['category_id']})
        fp_results[method]=dict(counts)
    categories=['Dense_miss','low_score_candidate_loss','strong_seed_propagation_loss','wrong_class_overlap','duplicate_stale_background']
    selected=[]
    for seq in sorted({r['video'] for r in plan}):
        for cat in categories:
            subset=[c for c in candidates if c['sequence']==seq and c['category']==cat]
            subset.sort(key=lambda c:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest())
            selected.extend(subset[:4])
    dump(target/'manual_review_candidates.json',{'selection':'up to four per category and sequence, SHA256 ordered','cases':selected,'status':'candidate labels; requires visual review'})
    dump(target/'summary.json',{'methods':results,'FP_overlap_candidates_not_formal_TIDE':fp_results,
        'caveat':'Paired official target match sets are not an additive AP decomposition; history joins use potentially noisy annotation track IDs.'})
    dump(out/'A_COMPLETE.json',{'paired_analysis_complete':True,'manual_review_candidates':len(selected),'new_YOLO_inferences':0})
