"""Descriptive summaries of saved decisions/FP flags; no evaluation/inference."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parent;REPO=OUT.parents[1];RUN=REPO/'runs/argoverse_hd_benchmark_v1'
d=json.loads((REPO/'data/argoverse_hd/annotations.json').read_text());aa=defaultdict(list)
for a in d['annotations']:
    if not a.get('iscrowd') and not a.get('ignore'):aa[a['image_id']].append(a)
ims=sorted(d['images'],key=lambda x:(x['sid'],x['fid'])); motion=[];small=[];prev={};sid=None
for im in ims:
    if sid!=im['sid']:prev={};sid=im['sid']
    curr={a['track']:np.array(a['bbox'],float) for a in aa[im['id']]};moves=[]
    for t,b in curr.items():
        if t in prev:
            old=prev[t];moves.append(np.linalg.norm(b[:2]+b[2:]/2-old[:2]-old[2:]/2)/max(np.linalg.norm(b[2:]),1e-9))
    motion.append(float(np.median(moves)) if moves else np.nan)
    small.append(float(np.mean([a['area']<1024 for a in aa[im['id']]])));prev=curr
motion=np.array(motion);small=np.array(small)
out={}
for m in ['C_fullfallback','adaptive_observed','adaptive_quality']:
    with (RUN/m/'frames.csv').open() as f:rr=list(csv.DictReader(f))
    result={'motion_by_decision_at_age1':{},'skip_FP_candidates':{},'age1_SKIP_signal_correlations':{}}
    for dec in ['FULL','SKIP']:
        mask=np.array([r['decision']==dec and r['since_full']=='1' for r in rr])&np.isfinite(motion)
        result['motion_by_decision_at_age1'][dec]={'frames':int(mask.sum()),'GT_motion_median_normalized_diagonal':float(np.median(motion[mask])),
            'native_small_object_fraction_mean':float(small[mask].mean())}
    with np.load(RUN/m/'eval_stats.npz') as z:
        off=np.r_[0,np.cumsum(z['n_pred'])];tp=np.array([z['tp'][off[i]:off[i+1],0].sum() for i in range(len(ims))]);
        with np.load(RUN/'baseline/eval_stats.npz') as b:
            bo=np.r_[0,np.cumsum(b['n_pred'])];btp=np.array([b['tp'][bo[i]:bo[i+1],0].sum() for i in range(len(ims))])
        loss=(btp-tp)/np.maximum(btp,1)
        mask=np.array([r['decision']=='SKIP' and r['since_full']=='1' for r in rr])&(btp>0)
        for key,signal in [('GT_motion',motion),('novelty',np.array([float(r['novelty']) for r in rr])),
                           ('flow_quality_mean',np.array([float(r['flow_quality_mean']) for r in rr]))]:
            valid=mask&np.isfinite(signal);rho=spearmanr(signal[valid],loss[valid]).statistic
            result['age1_SKIP_signal_correlations'][key]={'frames':int(valid.sum()),'Spearman_rho_with_missing_TP_fraction':float(rho)}
        with np.load(RUN/m/'preds.npz') as p:
            counts=Counter()
            for i,(im,row) in enumerate(zip(ims,rr)):
                if row['decision']!='SKIP':continue
                boxes=p[f"sequence_{im['sid']:02d}/{im['fid']}"]
                gt=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in aa[im['id']]])
                cls=np.array([a['category_id'] for a in aa[im['id']]])
                lo=np.maximum(gt[:,None,:2],boxes[None,:,:2]);hi=np.minimum(gt[:,None,2:],boxes[None,:,2:4]);inter=np.maximum(hi-lo,0).prod(2)
                overlap=inter/np.maximum(np.maximum(gt[:,2:]-gt[:,:2],0).prod(1)[:,None]+np.maximum(boxes[:,2:4]-boxes[:,:2],0).prod(1)[None,:]-inter,1e-12)
                same=overlap*(cls[:,None]==boxes[None,:,5]);ss=slice(off[i],off[i+1]);fp=~z['tp'][ss,0]&~z['ignored'][ss,0]
                for k in np.flatnonzero(fp):
                    best=same[:,k].max(initial=0)
                    category='duplicate_or_matching_conflict' if best>=.5 else 'wrong_class_overlap' if overlap[:,k].max(initial=0)>=.5 else 'coarse_localisation' if best>=.1 else 'background_or_stale_or_very_poor_localisation'
                    counts[category]+=1
            result['skip_FP_candidates']=dict(counts)
    out[m]=result
with (OUT/'scheduler_and_fp_diagnostics.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
