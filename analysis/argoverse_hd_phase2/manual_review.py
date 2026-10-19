"""Render a fixed, balanced subset of A's cases for visual diagnostic review."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/adaptive_phase2_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import numpy as np
from .common import archive, dump, load
from .paired import overlap


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    parser.add_argument('--name',default='manual_visuals_v2');args=parser.parse_args()
    out=Path(args.out).resolve();dest=out/'A_saved_pairing'/args.name;dest.mkdir(exist_ok=False)
    cases=load(out/'A_saved_pairing/manual_review_candidates.json')['cases']
    chosen={}
    for c in cases:chosen.setdefault((c['sequence'],c['category']),c)
    selected=list(chosen.values())
    plan=load(out/'frameplan.json');frames={r['image_id']:r for r in plan};bykey={r['key']:r for r in plan}
    evidence=load(out/'execution_plan.json')['approved_plan']
    data=load(Path(evidence['dataset'])/'annotations.json');byimage={r['image_id']:[] for r in plan}
    for a in data['annotations']:
        if not a.get('iscrowd') and not a.get('ignore'):byimage[a['image_id']].append(a)
    names={c['id']:c['name'] for c in data['categories']}
    dense=archive(Path(evidence['source_benchmark'])/'baseline/preds.npz')
    adaptive=archive(out/'R_reference_parity/adaptive_quality/preds.npz')
    with (out/'R_reference_parity/adaptive_quality/frames.csv').open() as f:decisions={r['video']+'/'+r['frame']:r['decision'] for r in csv.DictReader(f)}
    for start in range(0,len(selected),4):
        batch=selected[start:start+4];fig,axes=plt.subplots(len(batch),3,figsize=(15,3.7*len(batch)),squeeze=False)
        for row,c in enumerate(batch):
            current=frames[c['image_id']]
            box=np.array(c['GT_bbox'],float) if 'GT_bbox' in c else np.r_[c['prediction'][:2],np.array(c['prediction'][2:4])-c['prediction'][:2]]
            x,y,w,h=box;cx,cy=x+w/2,y+h/2;rw,rh=max(w*1.8,180),max(h*1.8,150)
            x0,y0=max(int(cx-rw/2),0),max(int(cy-rh/2),0);x1,y1=min(int(cx+rw/2),1920),min(int(cy+rh/2),1200)
            target=np.array([x,y,x+w,y+h])
            for col,delta in enumerate([-1,0,1]):
                ax=axes[row,col];key=f"{c['sequence']}/{current['frame_index']+delta}";info=bykey.get(key)
                if info is None:ax.axis('off');continue
                image=cv2.imread(info['image']);ax.imshow(cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),extent=[x0,x1,y1,y0],interpolation='nearest')
                for a in byimage[info['image_id']]:
                    gx,gy,gw,gh=a['bbox']
                    if not (x0<=gx+gw/2<=x1 and y0<=gy+gh/2<=y1):continue
                    color='yellow' if a['id']+1==c.get('annotation_id') else 'lime'
                    ax.add_patch(Rectangle((gx,gy),gw,gh,fill=False,edgecolor=color,linewidth=1.4))
                    ax.text(gx,gy,names[a['category_id']],color=color,fontsize=7,clip_on=True,backgroundcolor=(0,0,0,.5))
                for source,color in [(dense,'red'),(adaptive,'cyan')]:
                    p=source[key];ov=overlap([target],p[:,:4])[0];selected_boxes=np.argsort(-ov,kind='stable')[:1]
                    for k in selected_boxes:
                        if ov[k]<=0:continue
                        bx,by,ex,ey,conf,cls=p[k]
                        ax.add_patch(Rectangle((bx,by),ex-bx,ey-by,fill=False,edgecolor=color,linewidth=1.1,linestyle='--' if color=='cyan' else '-'))
                        ax.text(bx,ey,f'{names[int(cls)]} {conf:.3f}',color=color,fontsize=7,clip_on=True,backgroundcolor=(0,0,0,.6))
                ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.tick_params(labelsize=7)
                ax.set_title(f"case {start+row+1}: {key} {decisions[key]}\n{c['category']}",fontsize=9)
        fig.suptitle('GT: green (target yellow) | nearest-overlap Dense: red | adaptive: cyan dashed; fixed adjacent-frame crops',fontsize=11)
        fig.subplots_adjust(top=.94,bottom=.04,left=.05,right=.98,wspace=.22,hspace=.35)
        fig.savefig(dest/f'page_{start//4+1:02d}.png',dpi=130);plt.close(fig)
    dump(dest/'selected_cases.json',{'selection':'first SHA256-ordered candidate per sequence/category; no backfilling','cases':selected,'images_rendered':len(selected)*3,'status':'rendered for visual review; not relabelled'})
    print('Rendered',len(selected),'cases',flush=True)


if __name__=='__main__':main()
