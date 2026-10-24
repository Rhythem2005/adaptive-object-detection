"""Low-confidence offline operating points from previously saved official matches."""
import numpy as np
from analysis.argoverse_hd_phase2.common import load,dump
from analysis.branch_b_expanded_v1.continue_local import BASE,ROOT


def main():
    e=load(BASE/'runs/argoverse_hd_real_runtime_E_v4/summary.json');out={}
    for name,key in [('dense','A'),('practical','B'),('high_ap','D')]:out[name]=e['methods'][key]['fixed_points']['0.1']
    with np.load(BASE/'runs/argoverse_hd_coco_phase1_v1/fixed2_093/official_matches100.npz') as m:
        keep=(m['conf']>=.1)&m['considered']
        tp=int((m['tp'][:,0]&keep).sum());fp=int((~m['tp'][:,0]&~m['ignored'][:,0]&keep).sum())
        out['original_fixed2']=dict(TP=tp,FP=fp,precision=tp/(tp+fp),recall=tp/55412,F1=2*tp/(tp+fp+55412))
    dump(ROOT/'analysis/offline_conf10.json',dict(methods=out,confidence=.1,IoU=.5,frames=3416,
        provenance='Saved official matching only; no inference or new COCO matching'))


if __name__=='__main__':main()
