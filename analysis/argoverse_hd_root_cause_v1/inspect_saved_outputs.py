"""Read-only summaries of existing annotations, predictions and matching flags.

No model, pipeline, evaluator, AP routine, image decoder or tracker is invoked.
Box-coverage diagnostics are many-to-one overlap checks, NOT official recall/AP.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RUN = REPO / 'runs/argoverse_hd_benchmark_v1'
METHODS = ['baseline', 'fixed2_093', 'C_fullfallback', 'adaptive_observed', 'adaptive_quality']


def load_stats(method):
    with np.load(RUN / method / 'eval_stats.npz') as z:
        return {k: z[k] for k in z.files}


def iou(a, b):
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.maximum(hi-lo, 0).prod(2)
    aa = np.maximum(a[:, 2:4]-a[:, :2], 0).prod(1)
    ba = np.maximum(b[:, 2:4]-b[:, :2], 0).prod(1)
    return inter / np.maximum(aa[:, None]+ba[None, :]-inter, 1e-12)


def describe(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return {'n': len(x), 'mean': float(x.mean()), 'p10': float(np.percentile(x,10)),
            'median': float(np.median(x)), 'p90': float(np.percentile(x,90))} if len(x) else {'n': 0}


def coverage(gt, pred):
    ov = iou(gt[:,1:], pred[:,:4])
    same = ov * (gt[:,None,0] == pred[None,:,5])
    best = same.max(1, initial=0)
    any_best = ov.max(1, initial=0)
    conf50 = np.where(same >= .5, pred[None,:,4], 0).max(1, initial=0)
    conf75 = np.where(same >= .75, pred[None,:,4], 0).max(1, initial=0)
    return best, any_best, conf50, conf75


data = json.loads((REPO/'data/argoverse_hd/annotations.json').read_text())
manifest = json.loads((REPO/'data/argoverse_hd/manifest.json').read_text())
images = {x['id']: x for x in data['images']}
anns = defaultdict(list)
for x in data['annotations']:
    if not x.get('iscrowd') and not x.get('ignore'):
        anns[x['image_id']].append(x)
ordered = sorted(manifest, key=lambda r:(images[r['image_id']]['sid'],r['frame_index']))
keys = [f"sequence_{images[r['image_id']]['sid']:02d}/{r['frame_index']}" for r in ordered]
gts=[]; tracks=[]; areas=[]; sizes=[]; classes=[]
track_classes=defaultdict(set)
for row in ordered:
    selected=anns[row['image_id']]
    gt=np.array([[a['category_id'],a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]]
                 for a in selected],float).reshape(-1,5)
    gts.append(gt); tracks.append([a['track'] for a in selected])
    for a in selected:
        areas.append(a['area']);sizes.append(a['bbox'][2:]);classes.append(a['category_id'])
        track_classes[(images[row['image_id']]['sid'],a['track'])].add(a['category_id'])
areas=np.array(areas);sizes=np.array(sizes);classes=np.array(classes)
go=np.r_[0,np.cumsum([len(x) for x in gts])]
metrics={m:json.loads((RUN/m/'metrics.json').read_text()) for m in METHODS}
rows={}
for m in METHODS:
    with (RUN/m/'frames.csv').open() as stream:rows[m]=list(csv.DictReader(stream))
stats={m:load_stats(m) for m in METHODS}
preds={}
for m in METHODS:
    with np.load(RUN/m/'preds.npz') as z:preds[m]=[z[k] for k in keys]
for m in METHODS:
    assert np.array_equal(stats[m]['n_gt'],np.diff(go))
    assert np.array_equal(stats[m]['tcls'],classes)
    assert np.array_equal(stats[m]['n_pred'],[len(p) for p in preds[m]])
    assert [(r['video']+'/'+r['frame']) for r in rows[m]]==keys

report={'scope':'read-only archive diagnostics; no inference, benchmark, evaluator or AP routines executed',
        'frames':len(keys),'regular_instances':len(areas),'track_ids_changing_class':sum(len(x)>1 for x in track_classes.values()),
        'definitions':{'coverage':'GT has >=1 same-class saved prediction at stated IoU/confidence; not one-to-one matching; can overcount crowded cases',
                       'saved_match_rates':'TP counts from original eval_stats divided by original target counts, no rematching',
                       'sizes':'native area S<32², M=32²..96², L>=96²; size boundary conventions noted separately from official COCO AP'}}
report['object_sizes']={'width_input640':describe(sizes[:,0]/3),'height_input640':describe(sizes[:,1]/3),
                        'min_side_input640':describe(sizes.min(1)/3),
                        'input_min_side_below_4':int((sizes.min(1)/3<4).sum()),
                        'input_min_side_below_8':int((sizes.min(1)/3<8).sum()),
                        'area_input_below_32_squared':int((areas/9<1024).sum()),
                        'native_small':int((areas<1024).sum()),'native_medium':int(((areas>=1024)&(areas<9216)).sum()),
                        'native_large':int((areas>=9216).sum())}
base=[coverage(gt,p) for gt,p in zip(gts,preds['baseline'])]
bc=np.concatenate([x[2] for x in base]); bi=np.concatenate([x[0] for x in base]); ba=np.concatenate([x[1] for x in base])
report['dense_box_coverage']={}
for name, mask in [('all',np.ones(len(areas),bool)),('native_small',areas<1024),
                   ('native_medium',(areas>=1024)&(areas<9216)),('native_large',areas>=9216)]+[
                   (c['name'],classes==c['id']) for c in data['categories']]:
    n=int(mask.sum())
    report['dense_box_coverage'][name]={'instances':n,'coverage_iou50_conf001':int((bc[mask]>=.001).sum()),
        'coverage_iou50_conf10':int((bc[mask]>=.1).sum()),'coverage_iou50_conf25':int((bc[mask]>=.25).sum()),
        'coverage_iou75_conf001':int((np.concatenate([x[3] for x in base])[mask]>=.001).sum()),
        'same_class_best_iou_10_to_50':int(((bi[mask]>=.1)&(bi[mask]<.5)).sum()),
        'wrong_class_overlap_iou50_only':int(((bi[mask]<.5)&(ba[mask]>=.5)).sum()),
        'no_any_class_overlap_iou10':int((ba[mask]<.1).sum())}
report['dense_saved_match_counts_by_class']={}
s=stats['baseline']
for c in data['categories']:
    mask=s['pcls']==c['id'];gt_count=int((s['tcls']==c['id']).sum());out={}
    for cf in [.001,.05,.1,.25,.5]:
        p=mask&(s['conf']>=cf)
        out[str(cf)]={'predictions':int(p.sum()),'TP50':int(s['tp'][p,0].sum()),
            'TP75':int(s['tp'][p,5].sum()),'TP95':int(s['tp'][p,9].sum()),
            'FP50':int((~s['tp'][p,0]&~s['ignored'][p,0]).sum())}
    report['dense_saved_match_counts_by_class'][c['name']]={'targets':gt_count,'confidence_cuts':out}

frame_tp={};frame_tp25={}
for m in METHODS:
    s=stats[m];po=np.r_[0,np.cumsum(s['n_pred'])]
    frame_tp[m]=np.array([s['tp'][po[i]:po[i+1]].sum(0) for i in range(len(keys))])
    frame_tp25[m]=np.array([s['tp'][po[i]:po[i+1]][s['conf'][po[i]:po[i+1]]>=.25].sum(0) for i in range(len(keys))])
report['methods']={}
for m in METHODS:
    r=rows[m];s=stats[m];skip=np.array([x['decision']=='SKIP' for x in r]); n_gt=np.diff(go)
    out={'saved_match_TP_all_thresholds':s['tp'].sum(0).tolist(),
         'prediction_confidence':describe(s['conf']),
         'prediction_count_quantiles':describe(s['n_pred']),
         'frames_with_300_supported_predictions':int((s['n_pred']>=300).sum()),
         'triggers':dict(Counter(reason for x in r for reason in x['reasons'].split('|') if reason)),
         'skip_staleness':dict(Counter(x['since_full'] for x in r if x['decision']=='SKIP')),
         'on_own_SKIP':{'frames':int(skip.sum()),'targets':int(n_gt[skip].sum()),
             'dense_TP_by_IoU':frame_tp['baseline'][skip].sum(0).tolist(),
             'method_TP_by_IoU':frame_tp[m][skip].sum(0).tolist(),
             'dense_TP_conf25_by_IoU':frame_tp25['baseline'][skip].sum(0).tolist(),
             'method_TP_conf25_by_IoU':frame_tp25[m][skip].sum(0).tolist()}}
    out['staleness_breakdown']={}
    for age in sorted({x['since_full'] for x in r if x['decision']=='SKIP'},key=int):
        mask=skip&np.array([x['since_full']==age for x in r]);ng=int(n_gt[mask].sum())
        out['staleness_breakdown'][age]={'frames':int(mask.sum()),'targets':ng,
            'dense_TP50':int(frame_tp['baseline'][mask,0].sum()),'method_TP50':int(frame_tp[m][mask,0].sum()),
            'dense_TP75':int(frame_tp['baseline'][mask,5].sum()),'method_TP75':int(frame_tp[m][mask,5].sum())}
    report['methods'][m]=out

report['scheduler_comparisons']={}
for m in ['C_fullfallback','adaptive_observed','adaptive_quality']:
    buckets={};f=rows['fixed2_093'];a=rows[m]
    for fd in ['FULL','SKIP']:
        for ad in ['FULL','SKIP']:
            mask=np.array([x['decision']==fd and y['decision']==ad for x,y in zip(f,a)])
            buckets[f'fixed_{fd}__adaptive_{ad}']={'frames':int(mask.sum()),'targets':int(np.diff(go)[mask].sum()),
                'fixed_TP50':int(frame_tp['fixed2_093'][mask,0].sum()),'adaptive_TP50':int(frame_tp[m][mask,0].sum()),
                'fixed_TP75':int(frame_tp['fixed2_093'][mask,5].sum()),'adaptive_TP75':int(frame_tp[m][mask,5].sum()),
                'fixed_TP_mean10':float(frame_tp['fixed2_093'][mask].sum(0).mean()),
                'adaptive_TP_mean10':float(frame_tp[m][mask].sum(0).mean())}
    report['scheduler_comparisons'][m]=buckets

for m in METHODS[1:]:
    counts=Counter();reasons=Counter();quality_bins=defaultdict(Counter);geometry=defaultdict(list);motion=defaultdict(list)
    loss_by_size=defaultdict(Counter);prev_full={};prev_gt={};last_seq=None
    with np.load(RUN/m/'prediction_details.npz') as details:
        for i,(gt,p,row,ids) in enumerate(zip(gts,preds[m],rows[m],tracks)):
            seq=row['video']
            if seq!=last_seq:prev_full={};prev_gt={};last_seq=seq
            best,any_best,c50,c75=coverage(gt,p)
            dense_c=base[i][2];a_ok=c50>=.001;d_ok=dense_c>=.001
            current={tid:j for j,tid in enumerate(ids)}
            if row['decision']=='SKIP':
                for j,tid in enumerate(ids):
                    category=('both_cover' if d_ok[j] and a_ok[j] else 'dense_only' if d_ok[j] else 'adaptive_only' if a_ok[j] else 'both_miss')
                    counts[category]+=1
                    size='small' if anns[ordered[i]['image_id']][j]['area']<1024 else 'medium' if anns[ordered[i]['image_id']][j]['area']<9216 else 'large'
                    loss_by_size[size][category]+=1
                    if category=='dense_only':
                        if tid not in prev_full:reasons['not_annotated_at_last_FULL']+=1
                        elif prev_full[tid]['correct_conf50']>=.25:reasons['last_FULL_had_seedable_correct_box']+=1
                        elif prev_full[tid]['correct_conf50']>=.001:reasons['last_FULL_only_low_conf_correct_box']+=1
                        else:reasons['no_correct_box_at_last_FULL']+=1
                    if tid in prev_gt:
                        old=prev_gt[tid];b=gt[j,1:];wh=b[2:]-b[:2];oldwh=old[2:]-old[:2]
                        move=np.linalg.norm((b[:2]+b[2:]-old[:2]-old[2:])/2)/max(np.linalg.norm(wh),1e-9)
                        motion['covered' if a_ok[j] else 'missed'].append(move)
                ov=iou(gt[:,1:],p[:,:4]);same=ov*(gt[:,None,0]==p[None,:,5])
                detail=details[keys[i]];assert len(detail)==len(p)
                for k,pred in enumerate(p):
                    match=int(same[:,k].argmax()) if len(gt) else -1
                    b_iou=float(same[:,k].max(initial=0));q=detail[k,2]
                    qbin='0-.5' if q<.5 else '.5-.8' if q<.8 else '.8-.95' if q<.95 else '.95-1'
                    quality_bins[qbin]['predictions']+=1
                    for thr in [.5,.75,.9]:quality_bins[qbin][f'coverage_iou{thr}']+=int(b_iou>=thr)
                    if b_iou>=.1:
                        b=gt[match,1:];wh=b[2:]-b[:2];pw=pred[2:4]-pred[:2]
                        geometry['iou'].append(b_iou)
                        geometry['relative_center_error'].append(float(np.linalg.norm((pred[:2]+pred[2:4]-b[:2]-b[2:])/2)/np.linalg.norm(wh)))
                        geometry['abs_relative_width_error'].append(float(abs(pw[0]/wh[0]-1)))
                        geometry['abs_relative_height_error'].append(float(abs(pw[1]/wh[1]-1)))
            else:
                prev_full={tid:{'correct_conf50':float(c50[j]),'box':gt[j,1:]} for j,tid in enumerate(ids)}
            prev_gt={tid:gt[j,1:] for j,tid in enumerate(ids)}
    report['methods'][m]['skip_GT_coverage_partition']=dict(counts)
    report['methods'][m]['dense_only_candidate_loss_attribution']=dict(reasons)
    report['methods'][m]['skip_coverage_by_size']={k:dict(v) for k,v in loss_by_size.items()}
    report['methods'][m]['skip_prediction_quality_bins']={k:dict(v) for k,v in quality_bins.items()}
    report['methods'][m]['skip_nearest_same_class_geometry_iou_at_least_10']={k:describe(v) for k,v in geometry.items()}
    report['methods'][m]['previous_frame_GT_motion_normalized_by_diagonal']={k:describe(v) for k,v in motion.items()}

for m in METHODS:
    report['methods'][m]['reported_metrics']=metrics[m]['detection']
with (OUT/'saved_output_diagnostics.json').open('x') as stream:json.dump(report,stream,indent=2)
print('Saved',OUT/'saved_output_diagnostics.json')
print(json.dumps({'object_sizes':report['object_sizes'],'dense_box_coverage':report['dense_box_coverage']['all'],
                  'skip_partitions':{m:report['methods'][m]['skip_GT_coverage_partition'] for m in METHODS[1:]}},indent=2))
