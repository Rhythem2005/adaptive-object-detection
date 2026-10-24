"""Supplement offline per-sequence fixed-threshold points from saved matches."""
from collections import Counter
import numpy as np
from analysis.argoverse_hd_phase2.common import load,dump,sha
from analysis.branch_b_expanded_v1.continue_local import BASE,ROOT


def main():
    frames=load(ROOT/'manifests/frames.json');gt=load(ROOT/'manifests/ground_truth.json')
    byid={r['image_id']:r['video'] for r in frames}
    counts=Counter(byid[a['image_id']] for a in gt['annotations'] if not a.get('iscrowd',0))
    path=BASE/'runs/argoverse_hd_coco_phase1_v1/fixed2_093/official_matches100.npz'
    with np.load(path,allow_pickle=False) as m:
        offsets=np.r_[0,np.cumsum(m['n_pred'])];assert len(offsets)==3417
        selected=(m['conf']>=.25)&m['considered'];tp=m['tp'][:,0]&selected
        fp=~m['tp'][:,0]&~m['ignored'][:,0]&selected
        result={}
        for seq in sorted(counts):
            a=b=0
            for i,r in enumerate(frames):
                if r['video']==seq:
                    a+=int(tp[offsets[i]:offsets[i+1]].sum());b+=int(fp[offsets[i]:offsets[i+1]].sum())
            n=counts[seq];result[seq]=dict(TP=a,FP=b,targets=n,precision=a/(a+b) if a+b else None,
                recall=a/n,F1=2*a/(a+b+n))
        expected=load(ROOT/'analysis/offline.json')['original_fixed2']['fixed_conf25']['counts']
        assert sum(s['TP'] for s in result.values())==expected['TP']
        assert sum(s['FP'] for s in result.values())==expected['FP']
    source=load(BASE/'data/argoverse_hd/annotations.json')
    density={}
    for seq in sorted(counts):
        ids={r['image_id'] for r in frames if r['video']==seq};ann=[a for a in gt['annotations'] if a['image_id'] in ids]
        imagecounts=Counter(a['image_id'] for a in ann if not a.get('iscrowd',0))
        areas=np.array([a['area'] for a in ann if not a.get('iscrowd',0)])
        density[seq]=dict(frames=len(ids),regular_targets=counts[seq],crowds=sum(a.get('iscrowd',0) for a in ann),
            mean_targets_per_frame=counts[seq]/len(ids),max_targets_per_frame=max(imagecounts.values()),
            small_targets=int((areas<32**2).sum()),medium_targets=int(((areas>=32**2)&(areas<96**2)).sum()),
            large_targets=int((areas>=96**2).sum()),
            classes={str(c):sum(a['category_id']==c and not a.get('iscrowd',0) for a in ann) for c in range(8)})
    dump(ROOT/'analysis/offline_supplement.json',dict(original_fixed2_per_sequence_fixed25=result,
        dataset=density,source_match_archive=str(path),source_sha256=sha(path),new_scoring=False))


if __name__=='__main__':main()
