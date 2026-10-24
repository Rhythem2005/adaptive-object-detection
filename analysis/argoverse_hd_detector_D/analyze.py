"""D results only: saved-box/head diagnostics and official-score interpretation."""
from collections import Counter, defaultdict
import csv
import gzip
import json
import time

import numpy as np
import torch
from ultralytics.utils import ops
from adaptive_pipeline.shared.boxes import iou_matrix
from analysis.argoverse_hd_phase2.common import archive, dump, load
from .run import ROOT, verify


CASES=['imgsz640','imgsz960','shadow640_no_global300']


def operating(matches, denominator):
    result={}
    for conf in ['0.001','0.1','0.25']:
        counts=matches['counts_conf'+conf].sum(0)
        result[conf]={}
        for j,iou in [(0,50),(1,75)]:
            tp,fp=map(int,counts[j]);p=tp/max(tp+fp,1);r=tp/denominator
            result[conf][str(iou)]={'TP':tp,'FP':fp,'precision':p,'recall':r,'F1':2*p*r/max(p+r,1e-16)}
    return result


def size_name(area):
    # Disjoint descriptive strata, distinct from COCO's inclusive boundary ranges.
    return 'small_le1024' if area<=1024 else 'medium_le9216' if area<=9216 else 'large_gt9216'


def paired(a,b,gt):
    result={}
    for conf in [.001,.1,.25]:
        for j,iou in [(0,50),(5,75)]:
            counts=defaultdict(Counter)
            for ann in gt:
                old=float(a['gt_scores'][ann['id'],j])>=conf
                new=float(b['gt_scores'][ann['id'],j])>=conf
                label='both' if old and new else 'gained' if new else 'lost' if old else 'neither'
                for group in ['all','class'+str(ann['category_id']),size_name(ann['area'])]:counts[group][label]+=1
            result[f'IoU{iou}_conf{conf}']={k:dict(v) for k,v in counts.items()}
    return result


def raw_evidence(size, frames, gt, official):
    byimage=defaultdict(list)
    for a in gt:byimage[a['image_id']].append(a)
    cfg=load(ROOT/'plan.json')['configs'][str(size)]
    reverse={int(v):int(k) for k,v in cfg['class_map'].items()}
    rows=[];counts=defaultdict(Counter);max_confidence=[]
    for f in frames:
        with np.load(ROOT/f'imgsz{size}/raw'/f"{f['image_id']:06d}.npz") as source:
            raw=source['raw_head'][0];shape=source['input_tensor_shape']
            box=ops.xywh2xyxy(torch.from_numpy(raw[:4].T.copy()))
            box=ops.scale_boxes(shape,box,(1200,1920,3)).numpy()
            scores=raw[4:84];best=scores.max(0);classes=scores.argmax(0)
            gain=min(float(shape[0])/1200,float(shape[1])/1920)
            anns=byimage[f['image_id']]
            if not anns:continue
            targets=np.array([a['bbox'] for a in anns],float);targets[:,2:]+=targets[:,:2]
            overlap=iou_matrix(targets,box)
            for index,ann in enumerate(anns):
                same=classes==reverse[ann['category_id']]
                close=overlap[index]>=.5
                supported=same&close&(best>.001)
                any_class=close&(best>.001)
                correct_conf=float(best[supported].max()) if supported.any() else -1.
                gt_class_conf=float(scores[reverse[ann['category_id']],close].max()) if close.any() else -1.
                matched=float(official['gt_scores'][ann['id'],0])>=.001
                category='official_TP50' if matched else 'same_class_native_preNMS_overlap50' if supported.any() else 'other_class_native_preNMS_overlap50' if any_class.any() else 'no_confident_native_preNMS_overlap50'
                min_side=min(ann['bbox'][2:])*gain
                pixelbin='lt4' if min_side<4 else '4_to8' if min_side<8 else '8_to16' if min_side<16 else 'ge16'
                for group in ['all','class'+str(ann['category_id']),size_name(ann['area']),'resized_min_side_'+pixelbin]:
                    counts[group][category]+=1
                    counts[group]['targets']+=1
                rows.append({'size':size,'key':f['key'],'gt_id':ann['id'],'class':ann['category_id'],'area':ann['area'],
                    'diagnostic_size_bin':size_name(ann['area']),'resized_min_side':min_side,'pixel_bin':pixelbin,
                    'official_TP50':matched,'native_bestclass_correct_overlap50_maxconf':correct_conf,
                    'oracle_GTclass_on_overlap50_maxscore':gt_class_conf,'category':category})
                max_confidence.append(correct_conf)
    confidence={}
    values=np.array(max_confidence)
    for label,mask in [('absent',values<0),('001_to10',(values>=.001)&(values<.1)),('10_to25',(values>=.1)&(values<.25)),('ge25',values>=.25)]:confidence[label]=int(mask.sum())
    return {'counts':{g:dict(c) for g,c in counts.items()},'correct_native_overlap50_confidence':confidence,
        'note':'Raw candidate overlap coverage is not one-to-one detection recall, AP or recoverable objects. Categories for unmatched GT are spatial diagnostic candidates, not assigned causal errors. Oracle GT-class scores use ground truth for analysis only; native 80-class argmax remains unchanged.'},rows


def main():
    verify();assert (ROOT/'SCORING_COMPLETE.json').exists()
    if (ROOT/'ANALYSIS_COMPLETE.json').exists():raise FileExistsError('D analysis complete')
    started=time.perf_counter();frames=load(ROOT/'frames.json');ground=load(ROOT/'ground_truth.json')
    regular=[a for a in ground['annotations'] if not a.get('iscrowd')]
    metrics={name:load(ROOT/'scores'/name/'metrics.json') for name in CASES}
    matches={name:archive(ROOT/'scores'/name/'official_matches.npz') for name in CASES}
    points={name:operating(matches[name],len(regular)) for name in CASES}
    comparisons={'960_vs640':paired(matches['imgsz640'],matches['imgsz960'],regular),
        'cap_shadow_vs640':paired(matches['imgsz640'],matches['shadow640_no_global300'],regular)}
    raw={};rawrows=[];cap={}
    for size in [640,960]:
        raw[str(size)],values=raw_evidence(size,frames,regular,matches['imgsz'+str(size)]);rawrows.extend(values)
        with (ROOT/f'imgsz{size}/frames.csv').open() as stream:r=list(csv.DictReader(stream))
        cap[str(size)]={'frames_with_native_survivors_gt300':sum(int(f['native_survivors'])>300 for f in r),
            'frames_exactly300_native_survivors':sum(int(f['native_survivors'])==300 for f in r),
            'native_survivors_total':sum(int(f['native_survivors']) for f in r),
            'discarded_total80':sum(int(f['discarded_after_global300']) for f in r),
            'discarded_supported8':sum(int(f['discarded_supported_after_global300']) for f in r),
            'max_survivors':max(int(f['native_survivors']) for f in r),
            'all_exact_reconstructions':all(f['exact_all80_reconstruction']=='True' for f in r),
            'NMS_backends':sorted({f['backend'] for f in r}),'actual_tensor_shapes':sorted({f['last_forward'] for f in r})}
    # Primary paired cohort bins use original area; original GT is unchanged.
    heads={size:{group:raw[size]['counts'].get(group,{}) for group in ['all','small_le1024','medium_le9216','large_gt9216']} for size in raw}
    costs=load(ROOT/'INFERENCE_COMPLETE.json')
    events=[json.loads(line) for line in (ROOT/'requests.jsonl').read_text().splitlines() if json.loads(line)['status']=='complete']
    costs['successful_request_prediction_wall_seconds']=sum(r['predict_wall_seconds'] for r in events)
    costs['successful_request_prediction_process_seconds']=sum(r['predict_process_seconds'] for r in events)
    costs['scoring_wall_seconds']=sum(metrics[name]['scoring_wall_seconds'] for name in CASES)
    costs['diagnostic_prediction_wall_by_kind']={}
    for size in [640,960]:
        for kind in ['observed_primary','canonical_validation','excluded_full_frame_warmup']:
            selected=[r for r in events if r['size']==size and r['kind']==kind]
            costs['diagnostic_prediction_wall_by_kind'][f'{size}_{kind}']={'requests':len(selected),
                'mean_ms':1000*np.mean([r['predict_wall_seconds'] for r in selected]),
                'sum_seconds':sum(r['predict_wall_seconds'] for r in selected)}
    result={'metrics':metrics,'operating_points':points,'paired_official_matches':comparisons,'raw_evidence':raw,
        'cap_audit':cap,'execution_costs':costs,'regular_targets':len(regular),'crowds':len(ground['annotations'])-len(regular),
        'subset_only':True,'E_started':False}
    dump(ROOT/'summary.json',result)
    with gzip.open(ROOT/'GT_candidate_diagnostics.jsonl.gz','xt') as stream:
        for row in rawrows:stream.write(json.dumps(row)+'\n')
    with (ROOT/'summary.csv').open('x',newline='') as stream:
        names=['case','AP_percent','AP50_percent','AP75_percent','AR100_percent','APsmall_percent','APmedium_percent','APlarge_percent','TP25','FP25','P25_percent','R25_percent','F125_percent','macro_P25_percent','macro_R25_percent','macro_F125_percent']
        w=csv.DictWriter(stream,fieldnames=names);w.writeheader()
        for name in CASES:
            m=metrics[name]['COCO'];p=points[name]['0.25']['50'];f=metrics[name]['fixed_conf25']
            w.writerow({'case':name,**{k+'_percent':100*m[k] for k in ['AP','AP50','AP75','AR100']},
                **{'AP'+k+'_percent':100*m['AP_'+k] for k in ['small','medium','large']},
                'TP25':p['TP'],'FP25':p['FP'],'P25_percent':100*p['precision'],'R25_percent':100*p['recall'],'F125_percent':100*p['F1'],
                'macro_P25_percent':100*f['macro_precision'],'macro_R25_percent':100*f['macro_recall'],'macro_F125_percent':100*f['macro_F1']})
    dump(ROOT/'ANALYSIS_COMPLETE.json',{'analysis_wall_seconds':time.perf_counter()-started,'new_detector_requests':0,'E_started':False})
    print('D analysis complete',flush=True)


if __name__=='__main__':main()
