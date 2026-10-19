"""Validate and score the existing five-method Argoverse-HD prediction archives.

No images are decoded. No detector, tracker or pipeline module is imported.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path
import shutil
import sys

import numpy as np
import pycocotools._mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .coco import (prepare_ground_truth, prediction_records, make_coco, run_coco, coco_metrics,
                   official_match_arrays, sampled_ap, cap_mask, fixed_operating_point, unique_records)
from . import evaluator as legacy

METHODS = ['baseline','fixed2_093','C_fullfallback','adaptive_observed','adaptive_quality']
PINNED_VERSION = '2.0.11'
STOCK_COCO = {'person':0,'bicycle':1,'car':2,'motorcycle':3,'bus':5,'truck':7,'traffic_light':9,'stop_sign':11}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def dump_gzip(path,value):
    # mtime=0 avoids a timestamp changing the gzip bytes between reproductions.
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as zipped:
            zipped.write(json.dumps(value,separators=(',',':'),allow_nan=False).encode())


def prepare_inputs(dataset, benchmark):
    dataset,benchmark=Path(dataset).resolve(),Path(benchmark).resolve()
    source=json.loads((dataset/'annotations.json').read_text())
    manifest=json.loads((dataset/'manifest.json').read_text())
    audit=json.loads((benchmark/'dataset_audit.json').read_text())
    for name,key in [('annotations.json','annotations_sha256'),('manifest.json','manifest_sha256')]:
        if sha(dataset/name)!=audit[key]:raise ValueError(f'Dataset changed since benchmark: {name}')
    gt,conversion=prepare_ground_truth(source)
    images=unique_records(source['images'],'id');entries=unique_records(manifest,'image_id')
    if set(images)!=set(entries):raise ValueError('Annotation/manifest images differ')
    names=[c['name'] for c in source['categories']]
    if len(set(names))!=len(names) or set(names)!=set(STOCK_COCO):raise ValueError('Unexpected class taxonomy')
    by_name={c['name']:c['id'] for c in source['categories']}
    by_image={i:[] for i in images}
    for a in source['annotations']:by_image[a['image_id']].append(a)
    plan=[]
    for iid,im in images.items():
        row=entries[iid]
        if row['export_path']!=im['file_name'] or row['frame_index']!=im['fid'] or row['sequence']!=source['seq_dirs'][im['sid']]:
            raise ValueError(f'Image/sequence/frame mismatch: {iid}')
        plan.append({'image_id':iid,'key':f"sequence_{im['sid']:02d}/{im['fid']}",
                     'video':f"sequence_{im['sid']:02d}",'frame_index':im['fid'],
                     'image':str((dataset/im['file_name']).resolve())})
    plan.sort(key=lambda r:(r['video'],r['frame_index']))
    if [r['image_id'] for r in plan]!=sorted(images):
        raise ValueError('This benchmark requires original archive order to match increasing image IDs')
    if len({r['key'] for r in plan})!=len(plan):raise ValueError('Duplicate frame key')
    configs={};shared_manifest=None;shared_weights=None;method_info={}
    source_hashes={str(dataset/'annotations.json'):sha(dataset/'annotations.json'),str(dataset/'manifest.json'):sha(dataset/'manifest.json')}
    for method in METHODS:
        directory=benchmark/method
        metrics=json.loads((directory/'metrics.json').read_text())
        configs[method]=metrics['config']
        mapped_names=metrics['dataset']['names']
        if len(set(mapped_names))!=len(mapped_names) or set(mapped_names)!=set(names):raise ValueError('Saved class names disagree')
        expected_map={str(STOCK_COCO[n]):i for i,n in enumerate(mapped_names)}
        if metrics['config']['class_map']!=expected_map:raise ValueError('Saved prediction taxonomy not verified')
        inp=json.loads((directory/'input_manifest.json').read_text())
        digest=hashlib.sha256(json.dumps(inp['frames'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=inp['sha256'] or digest!=metrics['provenance']['input_manifest_sha256']:raise ValueError('Corrupt input manifest')
        if shared_manifest is not None and digest!=shared_manifest:raise ValueError('Different evaluation inputs across methods')
        shared_manifest=digest
        weights=metrics['provenance']['weights_sha256']
        if shared_weights is not None and weights!=shared_weights:raise ValueError('Different source checkpoint hashes')
        shared_weights=weights
        if len(inp['frames'])!=len(plan):raise ValueError('Wrong frame count')
        gt_rows=[]
        for p,r in zip(plan,inp['frames']):
            if any(p[k]!=r[k] for k in ['image_id','video','frame_index','image']):raise ValueError('Saved frame alignment differs')
            targets=[];ignored=[]
            for ann in by_image[p['image_id']]:
                cls=mapped_names.index(next(c['name'] for c in source['categories'] if c['id']==ann['category_id']))
                x,y,w,h=ann['bbox'];row=[cls,x,y,x+w,y+h]
                if ann.get('iscrowd') or ann.get('ignore'):ignored.append(row+[int(bool(ann.get('iscrowd')))])
                else:targets.append(row)
            target=np.array(targets,dtype='<f4').reshape(-1,5);ignore=np.array(ignored,dtype='<f4').reshape(-1,6)
            for value,key in [(target,'targets_sha256'),(ignore,'ignored_sha256')]:
                if hashlib.sha256(value.tobytes()).hexdigest()!=r[key]:raise ValueError(f'Target/ignore hash mismatch: {method}/{p["key"]}')
            gt_rows.append(target)
        with np.load(directory/'eval_stats.npz') as z:stats={k:z[k] for k in z.files}
        if not np.array_equal(stats['n_gt'],[len(x) for x in gt_rows]) or not np.array_equal(stats['tcls'],np.concatenate([x[:,0] for x in gt_rows])):
            raise ValueError('Saved target counts/classes disagree with annotations')
        for name in ['metrics.json','frames.csv','input_manifest.json','eval_stats.npz','preds.npz','prediction_details.npz']:
            source_hashes[str(directory/name)]=sha(directory/name)
        method_info[method]={'names':mapped_names,'category_map':{i:by_name[n] for i,n in enumerate(mapped_names)}}
    for key in ['model_path','device','imgsz','det_conf','det_iou','max_det','half','class_map','use_clahe','keyframe_refine']:
        if any(configs[m][key]!=configs['baseline'][key] for m in METHODS):raise ValueError(f'Unfair source setting: {key}')
    return gt,plan,method_info,{'conversion':conversion,'source_sha256':source_hashes,
        'frames':len(plan),'sequences':sorted({p['video'] for p in plan}),
        'original_input_manifest_sha256':shared_manifest,'source_weights_sha256':shared_weights,
        'source_configurations':configs,'annotations_sha256':audit['annotations_sha256']}


def load_predictions(directory,plan,category_map):
    keys=[r['key'] for r in plan]
    with np.load(directory/'preds.npz') as z:
        if set(z.files)!=set(keys):raise ValueError('Missing/extra prediction frame keys')
        arrays={r['image_id']:z[r['key']] for r in plan}
    records,conversion=prediction_records(arrays,category_map)
    with np.load(directory/'eval_stats.npz') as z:stats={k:z[k] for k in z.files}
    flat=np.concatenate(list(arrays.values()))
    if not np.array_equal(stats['n_pred'],[len(x) for x in arrays.values()]) or \
            not np.array_equal(stats['conf'],flat[:,4].astype(np.float32)) or \
            not np.array_equal(stats['pcls'],flat[:,5].astype(int)):
        raise ValueError('Prediction rows do not align with saved matching statistics')
    return records,conversion,stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True);parser.add_argument('--benchmark',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    out=Path(args.out).resolve();benchmark=Path(args.benchmark).resolve()
    if out.exists():raise FileExistsError(f'Choose a fresh output directory: {out}')
    if out.is_relative_to(benchmark) or out.is_relative_to(Path(args.dataset).resolve()):raise ValueError('Output cannot be inside original data/results')
    if importlib.metadata.version('pycocotools')!=PINNED_VERSION:raise ValueError('Install pinned requirements-evaluation.txt')
    gtdata,plan,info,evidence=prepare_inputs(args.dataset,benchmark)
    out.mkdir(parents=True)
    dump(out/'evidence.json',evidence)
    dump(out/'execution_plan.json',{'argv':sys.argv,'kind':'preflight_only' if args.validate_only else 'saved_prediction_scoring',
        'methods':METHODS,'frames':len(plan),'main_protocol':{'IoU':'0.50:0.05:0.95','recall':'0:0.01:1',
        'maxDets':[1,10,100],'area_ranges':'official defaults','useCats':1,'iouType':'bbox'},
        'auxiliary_protocol':{'maxDets':[1,10,300],'standard_COCO':False},
        'prediction_changes':'xyxy-to-xywh and named class-ID mapping only; no confidence filtering, clipping or re-NMS',
        'annotation_changes':'positive annotation IDs in scoring copy; original IDs retained as source_annotation_id'})
    dump_gzip(out/'annotations.coco.json.gz',gtdata)
    src=out/'scorer_sources';src.mkdir()
    for file in [Path(__file__),Path(inspect.getfile(coco_metrics)),Path(inspect.getfile(legacy)),
                 Path(inspect.getfile(COCO)),Path(inspect.getfile(COCOeval))]:
        shutil.copy2(file,src/(file.parent.name+'__'+file.name))
    for file in Path(__file__).resolve().parents[3].joinpath('tests').glob('test_coco_scoring.py'):shutil.copy2(file,src/file.name)
    dump(out/'runtime.json',{'python':platform.python_version(),'platform':platform.platform(),
        'numpy':np.__version__,'pycocotools':importlib.metadata.version('pycocotools'),
        'package_source_hashes':{str(p):sha(p) for p in [Path(inspect.getfile(COCO)),Path(inspect.getfile(COCOeval)),Path(pycocotools._mask.__file__)]}})
    validations={};summary=[]
    legacy.AP_PROTOCOL='current'
    for method in METHODS:
        print(f'Validating {method}',flush=True)
        records,conversion,stats=load_predictions(benchmark/method,plan,info[method]['category_map'])
        old=legacy.ap_per_class(stats['tp'],stats['conf'],stats['pcls'],stats['tcls'],ignored=stats['ignored'])
        previous=json.loads((benchmark/method/'metrics.json').read_text())
        for k,v in [('mAP50',old['ap'][:,0].mean()),('mAP50-95',old['ap'].mean()),('precision',old['p'].mean()),('recall',old['r'].mean())]:
            np.testing.assert_allclose(v,previous['detection'][k],atol=1e-14,rtol=0)
        validations[method]={'conversion':conversion,'previous_custom_metrics_reproduce':True}
        if args.validate_only:continue
        directory=out/method;directory.mkdir()
        dump_gzip(directory/'predictions.coco.json.gz',records)
        gt,dt=make_coco(gtdata,records)
        print(f'Official COCO standard scoring {method}',flush=True)
        scorer,log=run_coco(gt,dt)
        (directory/'coco_standard_stdout.txt').write_text(log)
        standard=coco_metrics(scorer)
        point=fixed_operating_point(scorer)
        official_tp,official_ignore,considered=official_match_arrays(scorer,len(records))
        keep=cap_mask(stats['conf'],stats['pcls'],stats['n_pred'],100)
        np.testing.assert_array_equal(keep,considered)
        checked=sampled_ap(official_tp,stats['conf'],stats['pcls'],stats['tcls'],official_ignore,keep=keep)
        np.testing.assert_allclose([checked['AP'],checked['AP50']],[standard['AP'],standard['AP50']],atol=1e-14,rtol=0)
        integration=sampled_ap(stats['tp'],stats['conf'],stats['pcls'],stats['tcls'],stats['ignored'],stable=False)
        sort_cap=sampled_ap(stats['tp'],stats['conf'],stats['pcls'],stats['tcls'],stats['ignored'],keep=keep)
        decomposition={'old_custom_AP':float(old['ap'].mean()),'old_custom_AP50':float(old['ap'][:,0].mean()),
            'same_matches_COCO_accumulation_original_sort':integration,
            'same_matches_COCO_accumulation_stable_sort_cap100':sort_cap,
            'official_COCO_standard':{'AP':standard['AP'],'AP50':standard['AP50']},
            'note':'Sequential, order-dependent diagnostic differences; fixed original matching is not an official scorer',
            'predictions_excluded_by_per_image_category_cap100':int((~keep).sum()),
            'old_TP_considered_by_IoU':stats['tp'][keep].sum(0).tolist(),
            'official_TP_by_IoU':official_tp.sum(0).tolist(),
            'TP_flag_changes_considered_by_IoU':(stats['tp'][keep]!=official_tp[keep]).sum(0).tolist(),
            'ignore_flag_changes_considered_by_IoU':(stats['ignored'][keep]!=official_ignore[keep]).sum(0).tolist()}
        for box in conversion['zero_area_predictions_retained']:
            i=box['flat_index']
            if considered[i] and (official_tp[i].any() or official_ignore[i].any()):raise AssertionError('Unexpected zero-area box credit/ignore')
        np.savez_compressed(directory/'official_tensors100.npz',precision=scorer.eval['precision'],recall=scorer.eval['recall'],scores=scorer.eval['scores'],
                            category_ids=scorer.params.catIds,image_ids=scorer.params.imgIds,iou_thresholds=scorer.params.iouThrs,recall_thresholds=scorer.params.recThrs)
        np.savez_compressed(directory/'official_matches100.npz',tp=official_tp,ignored=official_ignore,considered=considered,
                            conf=stats['conf'],pcls=stats['pcls'],tcls=stats['tcls'],n_pred=stats['n_pred'],n_gt=stats['n_gt'])
        del scorer
        per_sequence={}
        for name in evidence['sequences']:
            ids=[r['image_id'] for r in plan if r['video']==name]
            seq,seqlog=run_coco(gt,dt,ids)
            per_sequence[name]={'frames':len(ids),'metrics':coco_metrics(seq)}
            (directory/(name+'_coco_stdout.txt')).write_text(seqlog)
            print(f'  scored {method}/{name}',flush=True)
            del seq
        aux,auxlog=run_coco(gt,dt,max_dets=(1,10,300))
        auxiliary=coco_metrics(aux,300)
        (directory/'coco_aux300_stdout.txt').write_text(auxlog)
        result={'official_COCO_max100':standard,'supplementary_fixed_conf25_IoU50':point,
            'auxiliary_COCO_rules_max300_nonstandard':auxiliary,'previous_custom':previous['detection'],
            'protocol_decomposition':decomposition,'per_sequence':per_sequence,
            'source_efficiency_unchanged_not_remeasured':previous['efficiency'],
            'conversion':conversion,'validation':{'official_tensor_summary_parity':True,'independent_accumulator_parity':True}}
        dump(directory/'metrics.json',result)
        line={'method':method,**{k:v for k,v in standard.items() if k!='per_class'},
              'previous_custom_AP':float(old['ap'].mean()),'AP_delta':standard['AP']-float(old['ap'].mean()),'auxiliary_AP300':auxiliary['AP']}
        summary.append(line)
        with (out/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(line));writer.writeheader();writer.writerows(summary)
        print(json.dumps(line),flush=True)
        del aux,gt,dt,records
    dump(out/'validation.json',validations)
    for path,digest in evidence['source_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Input changed during scoring: {path}')
    if 'ultralytics' in sys.modules or 'torch' in sys.modules:raise AssertionError('Unexpected model library import')
    dump(out/'COMPLETE.json',{'status':'validated_only' if args.validate_only else 'complete','methods':METHODS,
        'input_hashes_unchanged':True,'model_libraries_imported':False,'inference_or_pipeline_replay':False})


if __name__=='__main__':main()
