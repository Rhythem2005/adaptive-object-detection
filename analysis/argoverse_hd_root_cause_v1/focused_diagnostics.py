"""Additional read-only overlap/history/call-allocation summaries; no AP or inference."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent; REPO=OUT.parents[1]; RUN=REPO/'runs/argoverse_hd_benchmark_v1'
data=json.loads((REPO/'data/argoverse_hd/annotations.json').read_text())
ims={x['id']:x for x in data['images']}; byid=defaultdict(list)
for a in data['annotations']:
    if not a.get('ignore') and not a.get('iscrowd'):byid[a['image_id']].append(a)
ims=sorted(ims.values(),key=lambda x:(x['sid'],x['fid']))
keys=[f"sequence_{x['sid']:02d}/{x['fid']}" for x in ims]
methods=['baseline','fixed2_093','C_fullfallback','adaptive_observed','adaptive_quality']
pred={}; rows={}
for m in methods:
    with np.load(RUN/m/'preds.npz') as z:pred[m]=[z[k] for k in keys]
    with (RUN/m/'frames.csv').open() as f:rows[m]=list(csv.DictReader(f))

def overlap(g,p):
    lo=np.maximum(g[:,None,:2],p[None,:,:2]);hi=np.minimum(g[:,None,2:],p[None,:,2:4])
    it=np.maximum(hi-lo,0).prod(2)
    return it/np.maximum(np.maximum(g[:,2:]-g[:,:2],0).prod(1)[:,None]+np.maximum(p[:,2:4]-p[:,:2],0).prod(1)[None,:]-it,1e-12)

def conf(ov,p,cls,threshold):
    return np.where((ov>=threshold)&(cls[:,None]==p[None,:,5]),p[None,:,4],0).max(1,initial=0)

result={'note':'Descriptive saved-box overlap diagnostics; no official matching/AP rescoring or tracker replay.',
        'dense_wrong_class_overlap_candidates':{},'methods':{},'common_SKIP':{}}
matrix=np.zeros((8,8),int)
for i,im in enumerate(ims):
    aa=byid[im['id']];cls=np.array([a['category_id'] for a in aa])
    g=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in aa],float)
    p=pred['baseline'][i];ov=overlap(g,p)
    same=ov*(cls[:,None]==p[None,:,5]);best=ov.argmax(1)
    bad=(same.max(1,initial=0)<.5)&(ov.max(1,initial=0)>=.5)
    for j in np.flatnonzero(bad):matrix[cls[j],int(p[best[j],5])]+=1
result['dense_wrong_class_overlap_candidates']={data['categories'][i]['name']:{data['categories'][k]['name']:int(n) for k,n in enumerate(r) if n}
                                                for i,r in enumerate(matrix)}
for method in methods[1:]:
    last={};high_history=set();before_last_high=set();last_seq=-1;c=Counter();loc=Counter();size=Counter();history=Counter()
    for i,im in enumerate(ims):
        if im['sid']!=last_seq:last={};high_history=set();before_last_high=set();last_seq=im['sid']
        aa=byid[im['id']];cls=np.array([a['category_id'] for a in aa]);ids=[a['track'] for a in aa]
        g=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in aa],float)
        p=pred[method][i];bp=pred['baseline'][i];ov=overlap(g,p);bv=overlap(g,bp)
        pc=conf(ov,p,cls,.5);dc=conf(bv,bp,cls,.5); pc75=conf(ov,p,cls,.75);dc75=conf(bv,bp,cls,.75)
        if rows[method][i]['decision']=='SKIP':
            size['emitted_boxes']+=len(p)
            size['emitted_boxes_flow_min_side_below8']+=int(((p[:,2:4]-p[:,:2]).min(1)/3<8).sum())
            for j,tid in enumerate(ids):
                if pc[j]<.001 and dc[j]>=.001 and tid in last and .001<=last[tid]['c50']<.25:
                    if tid in high_history:history['ever_high_conf_correct_at_an_earlier_FULL']+=1
                    else:history['never_high_conf_correct_at_any_prior_FULL']+=1
                    if tid in before_last_high:history['high_conf_correct_at_FULL_immediately_before_last_FULL']+=1
                    if last[tid]['c50']>=.1:history['last_FULL_correct_conf_10_to_25']+=1
                    else:history['last_FULL_correct_conf_001_to_10']+=1
                if tid in last and last[tid]['c75']>=.25 and dc75[j]>=.25:
                    loc['seed_and_current_dense_good_at75_conf25']+=1
                    same=ov[j]*(p[:,5]==cls[j]);best=same.max(initial=0)
                    if best<.5:loc['propagation_no_current_box_at50']+=1
                    elif best<.75:loc['propagation_box_50_to75']+=1
                    else:loc['propagation_box_at75_or_better']+=1
        else:
            before_last_high={tid for tid,v in last.items() if v['c50']>=.25}
            last={tid:{'c50':float(pc[j]),'c75':float(pc75[j])} for j,tid in enumerate(ids)}
            high_history|={tid for tid,v in last.items() if v['c50']>=.25}
    result['methods'][method]={'low_conf_candidate_history':dict(history),'strong_seed_localisation_check':dict(loc),'flow_scale_gate':dict(size)}

for a,b in [('C_fullfallback','adaptive_observed'),('adaptive_observed','adaptive_quality'),('C_fullfallback','adaptive_quality')]:
    mask=np.array([x['decision']=='SKIP' and y['decision']=='SKIP' for x,y in zip(rows[a],rows[b])]);frames=int(mask.sum())
    record={'frames':frames,'bitwise_identical_prediction_arrays':sum(np.array_equal(pred[a][i],pred[b][i]) for i in np.flatnonzero(mask))}
    for m in [a,b]:
        with np.load(RUN/m/'eval_stats.npz') as s:
            po=np.r_[0,np.cumsum(s['n_pred'])];part=np.concatenate([np.arange(po[i],po[i+1]) for i in np.flatnonzero(mask)])
            record[m+'_TP10']=s['tp'][part].sum(0).tolist()
    result['common_SKIP'][a+'__'+b]=record
result['scheduler_decision_signals']={}
for m in methods[2:]:
    summary={}
    for d in ['FULL','SKIP']:
        rr=[r for r in rows[m] if r['decision']==d]
        v={}
        for k in ['novelty','fail_frac','n_tracks','n_need','flow_quality_mean','lum']:
            x=np.array([float(r[k]) for r in rr]);x=x[np.isfinite(x)]
            v[k]={'median':float(np.median(x)),'p90':float(np.percentile(x,90))} if len(x) else None
        summary[d]=v
    result['scheduler_decision_signals'][m]=summary
with (OUT/'focused_diagnostics.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
